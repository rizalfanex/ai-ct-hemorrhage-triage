from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/processed/rsna_holdout_5000_manifest.csv")
DICOM_DIR = Path("data/processed/rsna_holdout_5000_dicom")
CHECKPOINT_PATH = Path("outputs/models/best_convnext_tiny_brain_only_rsna_10000.pt")

OUTPUT_DIR = Path("outputs/logs")
PRED_CSV = OUTPUT_DIR / "holdout_5000_predictions_convnext_tiny_brain_only_10000.csv"
THRESHOLD_CSV = OUTPUT_DIR / "holdout_5000_threshold_sweep_any_convnext_tiny_brain_only_10000.csv"
TRIAGE_CSV = OUTPUT_DIR / "holdout_5000_triage_simulation_convnext_tiny_brain_only_10000.csv"
METRICS_JSON = OUTPUT_DIR / "holdout_5000_eval_metrics_convnext_tiny_brain_only_10000.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32


class BrainWindowOnlyDataset(Dataset):
    """
    Uses existing RSNADicomDataset output, keeps only channel 0,
    and repeats it into 3 channels: [brain, brain, brain].
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item["image"]
        brain = image[0:1, :, :]
        item["image"] = brain.repeat(3, 1, 1)
        return item


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_ap(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def threshold_sweep(y_true, y_prob):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append({
            "threshold": round(float(threshold), 2),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "predicted_positive": int(tp + fp),
            "predicted_positive_rate": float((tp + fp) / len(y_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
        })

    return pd.DataFrame(rows)


def triage_simulation(pred_df):
    ranked = pred_df.sort_values("prob_any", ascending=False).reset_index(drop=True)

    total_cases = len(ranked)
    total_positive = int(ranked["true_any"].sum())

    rows = []

    for top_percent in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]:
        k = max(1, int(round(total_cases * top_percent / 100)))
        top_df = ranked.head(k)

        captured_positive = int(top_df["true_any"].sum())
        false_priority = int(k - captured_positive)

        rows.append({
            "top_percent_workload": top_percent,
            "cases_prioritized": k,
            "true_positive_captured": captured_positive,
            "false_priority_cases": false_priority,
            "recall_at_k": float(captured_positive / max(total_positive, 1)),
            "precision_at_k": float(captured_positive / max(k, 1)),
            "missed_positive": int(total_positive - captured_positive),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def main():
    print("===== EVALUATE CONVNEXT-TINY BRAIN-ONLY ON NATURAL HOLDOUT 5000 =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    manifest_df = pd.read_csv(MANIFEST_PATH)

    print("Manifest shape:", manifest_df.shape)
    print("Holdout label distribution:")
    print(manifest_df[LABEL_COLUMNS].sum())
    print("Holdout any prevalence:", float(manifest_df["any"].mean()))

    base_dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=IMAGE_SIZE,
    )

    dataset = BrainWindowOnlyDataset(base_dataset)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    label_columns = checkpoint["label_columns"]

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation any_auc:", checkpoint.get("any_auc"))
    print("Model:", model_name)
    print("Ablation:", checkpoint.get("ablation"))

    model = build_model(
        model_name=model_name,
        num_classes=len(label_columns),
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_image_ids = []
    all_probs = []
    all_labels = []

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_image_ids.extend(batch["image_id"])
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if step % 25 == 0 or step == 1:
            print(f"Inference step {step}/{len(loader)}")

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    pred_rows = []

    for i, image_id in enumerate(all_image_ids):
        row = {"image_id": image_id}

        for j, label in enumerate(label_columns):
            row[f"true_{label}"] = int(labels_np[i, j])
            row[f"prob_{label}"] = float(probs_np[i, j])

        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_CSV, index=False)

    print("\nSaved prediction CSV:", PRED_CSV)

    print("\n===== PER-LABEL AUC/AP ON NATURAL HOLDOUT =====")

    per_label = {}

    for j, label in enumerate(label_columns):
        y_true = labels_np[:, j].astype(int)
        y_prob = probs_np[:, j]

        result = {
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auc": safe_auc(y_true, y_prob),
            "average_precision": safe_ap(y_true, y_prob),
        }

        per_label[label] = result
        print(label, result)

    any_idx = label_columns.index("any")
    y_true_any = labels_np[:, any_idx].astype(int)
    y_prob_any = probs_np[:, any_idx]

    print("\n===== THRESHOLD SWEEP FOR ANY ON NATURAL HOLDOUT =====")
    threshold_df = threshold_sweep(y_true_any, y_prob_any)
    threshold_df.to_csv(THRESHOLD_CSV, index=False)
    print(threshold_df)

    print("\n===== TRIAGE SIMULATION =====")
    triage_df = triage_simulation(pred_df)
    triage_df.to_csv(TRIAGE_CSV, index=False)
    print(triage_df)

    best_f1 = threshold_df.sort_values("f1", ascending=False).iloc[0].to_dict()

    candidates_90 = threshold_df[threshold_df["recall"] >= 0.90].copy()
    candidates_95 = threshold_df[threshold_df["recall"] >= 0.95].copy()

    best_recall_90 = None
    if len(candidates_90) > 0:
        best_recall_90 = candidates_90.sort_values(
            ["precision", "f1"],
            ascending=False,
        ).iloc[0].to_dict()

    best_recall_95 = None
    if len(candidates_95) > 0:
        best_recall_95 = candidates_95.sort_values(
            ["precision", "f1"],
            ascending=False,
        ).iloc[0].to_dict()

    summary = {
        "manifest": str(MANIFEST_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_any_auc": checkpoint.get("any_auc"),
        "model_name": model_name,
        "experiment_name": checkpoint.get("experiment_name", "convnext_tiny_brain_only_rsna_10000"),
        "ablation": checkpoint.get("ablation", "brain_window_only_repeated_3_channels"),
        "parameters": checkpoint.get("parameters"),
        "holdout_size": len(pred_df),
        "holdout_any_prevalence": float(y_true_any.mean()),
        "per_label": per_label,
        "best_f1_threshold_any": best_f1,
        "best_threshold_with_recall_at_least_90": best_recall_90,
        "best_threshold_with_recall_at_least_95": best_recall_95,
        "outputs": {
            "predictions_csv": str(PRED_CSV),
            "threshold_csv": str(THRESHOLD_CSV),
            "triage_csv": str(TRIAGE_CSV),
        },
    }

    METRICS_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print("Best F1 threshold:")
    print(best_f1)

    print("\nBest threshold recall >= 0.90:")
    print(best_recall_90)

    print("\nBest threshold recall >= 0.95:")
    print(best_recall_95)

    print("\nSaved threshold CSV:", THRESHOLD_CSV)
    print("Saved triage CSV:", TRIAGE_CSV)
    print("Saved metrics JSON:", METRICS_JSON)

    print("\nSTATUS: NATURAL HOLDOUT 5000 EVALUATION BRAIN-ONLY PASSED")


if __name__ == "__main__":
    main()
