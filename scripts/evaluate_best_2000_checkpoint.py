from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Subset

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/processed/rsna_2000_manifest.csv")
DICOM_DIR = Path("data/processed/rsna_2000_dicom")
CHECKPOINT_PATH = Path("outputs/models/best_efficientnet_b0_rsna_2000.pt")

OUTPUT_DIR = Path("outputs/logs")
PRED_CSV = OUTPUT_DIR / "best_2000_val_predictions.csv"
THRESHOLD_CSV = OUTPUT_DIR / "best_2000_threshold_sweep_any.csv"
METRICS_JSON = OUTPUT_DIR / "best_2000_eval_metrics.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42
VAL_SIZE = 0.2


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


@torch.no_grad()
def main():
    print("===== EVALUATE BEST 2000 CHECKPOINT =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    manifest_df = pd.read_csv(MANIFEST_PATH)
    indices = np.arange(len(manifest_df))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=manifest_df["any"].values,
    )

    dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=IMAGE_SIZE,
    )

    val_dataset = Subset(dataset, val_idx.tolist())

    val_loader = DataLoader(
        val_dataset,
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
    print("Checkpoint any_auc:", checkpoint.get("any_auc"))
    print("Model:", model_name)
    print("Val size:", len(val_dataset))

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

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_image_ids.extend(batch["image_id"])
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

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

    print("Saved prediction CSV:", PRED_CSV)

    print("\n===== PER-LABEL AUC/AP =====")

    per_label = {}

    for j, label in enumerate(label_columns):
        y_true = labels_np[:, j].astype(int)
        y_prob = probs_np[:, j]

        auc = safe_auc(y_true, y_prob)
        ap = safe_ap(y_true, y_prob)

        per_label[label] = {
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "auc": auc,
            "average_precision": ap,
        }

        print(label, per_label[label])

    print("\n===== THRESHOLD SWEEP FOR ANY =====")

    any_idx = label_columns.index("any")
    y_true_any = labels_np[:, any_idx].astype(int)
    y_prob_any = probs_np[:, any_idx]

    sweep_rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob_any >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_any,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(y_true_any, y_pred, zero_division=0)
        recall = recall_score(y_true_any, y_pred, zero_division=0)
        f1 = f1_score(y_true_any, y_pred, zero_division=0)

        row = {
            "threshold": round(float(threshold), 2),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
        }

        sweep_rows.append(row)
        print(row)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(THRESHOLD_CSV, index=False)

    candidates_90 = sweep_df[sweep_df["recall"] >= 0.90].copy()
    candidates_95 = sweep_df[sweep_df["recall"] >= 0.95].copy()

    best_f1_row = sweep_df.sort_values("f1", ascending=False).iloc[0].to_dict()

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
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_size": len(val_dataset),
        "per_label": per_label,
        "best_f1_threshold_any": best_f1_row,
        "best_threshold_with_recall_at_least_90": best_recall_90,
        "best_threshold_with_recall_at_least_95": best_recall_95,
    }

    METRICS_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== BEST THRESHOLD SUMMARY =====")
    print("Best F1 threshold:")
    print(best_f1_row)

    print("\nBest threshold with recall >= 0.90:")
    print(best_recall_90)

    print("\nBest threshold with recall >= 0.95:")
    print(best_recall_95)

    print("\nSaved threshold sweep:", THRESHOLD_CSV)
    print("Saved metrics JSON:", METRICS_JSON)

    print("\nSTATUS: BEST CHECKPOINT EVALUATION PASSED")


if __name__ == "__main__":
    main()
