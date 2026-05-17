from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, random_split

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/sample/rsna_sample_manifest.csv")
DICOM_DIR = Path("data/sample/rsna_dicom")
CHECKPOINT_PATH = Path("outputs/models/smoke_efficientnet_b0_rsna_sample.pt")

OUTPUT_LOG_DIR = Path("outputs/logs")
PREDICTION_CSV = OUTPUT_LOG_DIR / "smoke_eval_predictions.csv"
METRICS_JSON = OUTPUT_LOG_DIR / "smoke_eval_metrics.json"
METRICS_CSV = OUTPUT_LOG_DIR / "smoke_eval_metrics.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42
THRESHOLD = 0.5


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


@torch.no_grad()
def main() -> None:
    print("===== RSNA SMOKE EVALUATION =====")

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=IMAGE_SIZE,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Dataset length:", len(dataset))
    print("Val length:", len(val_dataset))
    print("Labels:", LABEL_COLUMNS)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    label_columns = checkpoint["label_columns"]

    print("Checkpoint model:", model_name)
    print("Checkpoint labels:", label_columns)

    model = build_model(
        model_name=model_name,
        num_classes=len(label_columns),
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_probs = []
    all_labels = []
    all_image_ids = []

    total_loss = 0.0
    total_items = 0

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_image_ids.extend(batch["image_id"])

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    val_loss = total_loss / max(total_items, 1)

    print("Logits shape:", logits_np.shape)
    print("Probs shape:", probs_np.shape)
    print("Labels shape:", labels_np.shape)
    print("Val loss:", round(val_loss, 6))

    pred_np = (probs_np >= THRESHOLD).astype(int)

    prediction_rows = []

    for i, image_id in enumerate(all_image_ids):
        row = {
            "image_id": image_id,
        }

        for j, label in enumerate(label_columns):
            row[f"true_{label}"] = int(labels_np[i, j])
            row[f"prob_{label}"] = float(probs_np[i, j])
            row[f"pred_{label}"] = int(pred_np[i, j])

        prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(PREDICTION_CSV, index=False)

    metric_rows = []
    metrics = {
        "val_loss": float(val_loss),
        "threshold": THRESHOLD,
        "labels": {},
    }

    print("\n===== PER-LABEL METRICS =====")

    for j, label in enumerate(label_columns):
        y_true = labels_np[:, j].astype(int)
        y_prob = probs_np[:, j]
        y_pred = pred_np[:, j].astype(int)

        auc = safe_auc(y_true, y_prob)
        ap = safe_ap(y_true, y_prob)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        row = {
            "label": label,
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "auc": auc,
            "average_precision": ap,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        metric_rows.append(row)
        metrics["labels"][label] = row

        print(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    any_index = label_columns.index("any")

    y_true_any = labels_np[:, any_index].astype(int)
    y_pred_any = pred_np[:, any_index].astype(int)

    cm_any = confusion_matrix(
        y_true_any,
        y_pred_any,
        labels=[0, 1],
    )

    print("\n===== CONFUSION MATRIX ANY =====")
    print("Rows=true [normal, positive], Cols=pred [normal, positive]")
    print(cm_any)

    metrics["confusion_matrix_any"] = cm_any.tolist()

    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved predictions:", PREDICTION_CSV)
    print("Saved metrics CSV:", METRICS_CSV)
    print("Saved metrics JSON:", METRICS_JSON)

    print("\nSTATUS: SMOKE EVALUATION PASSED")


if __name__ == "__main__":
    main()
