from pathlib import Path
import sys
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/processed/rsna_train_10000_manifest.csv")
DICOM_DIR = Path("data/processed/rsna_train_10000_dicom")

OUTPUT_MODEL_DIR = Path("outputs/models")
OUTPUT_LOG_DIR = Path("outputs/logs")

MODEL_NAME = "densenet121"
EXPERIMENT_NAME = "densenet121_rsna_10000"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_SIZE = 0.2
THRESHOLD = 0.5
PRETRAINED = True
MAX_POS_WEIGHT = 20.0
USE_AMP = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(df: pd.DataFrame, label_columns: list[str]) -> torch.Tensor:
    labels = df[label_columns].values.astype(np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos

    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, MAX_POS_WEIGHT)

    return torch.tensor(pos_weight, dtype=torch.float32)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_items = 0

    all_probs = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    preds_np = (probs_np >= THRESHOLD).astype(int)

    metrics = {}
    metric_rows = []

    for j, label in enumerate(LABEL_COLUMNS):
        y_true = labels_np[:, j].astype(int)
        y_prob = probs_np[:, j]
        y_pred = preds_np[:, j].astype(int)

        row = {
            "label": label,
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "auc": safe_auc(y_true, y_prob),
            "average_precision": safe_ap(y_true, y_prob),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        metrics[label] = row
        metric_rows.append(row)

    val_loss = total_loss / max(total_items, 1)

    return {
        "val_loss": val_loss,
        "any_auc": metrics["any"]["auc"],
        "any_average_precision": metrics["any"]["average_precision"],
        "per_label": metrics,
        "metric_rows": metric_rows,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()

    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if step % 50 == 0 or step == 1:
            print(f"Epoch {epoch} | step {step}/{len(loader)} | loss={loss.item():.6f}")

    return total_loss / max(total_items, 1)


def main() -> None:
    print("===== RSNA DENSENET121 TRAINING 10000 =====")

    set_seed(SEED)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA AMP:", USE_AMP)

    manifest_df = pd.read_csv(MANIFEST_PATH)

    print("Manifest shape:", manifest_df.shape)
    print("Label distribution:")
    print(manifest_df[LABEL_COLUMNS].sum())

    indices = np.arange(len(manifest_df))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=manifest_df["any"].values,
    )

    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    val_df = manifest_df.iloc[val_idx].reset_index(drop=True)

    print("Train size:", len(train_idx))
    print("Val size:", len(val_idx))
    print("Train any distribution:")
    print(train_df["any"].value_counts().sort_index())
    print("Val any distribution:")
    print(val_df["any"].value_counts().sort_index())

    dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=IMAGE_SIZE,
    )

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    pos_weight = compute_pos_weight(train_df, LABEL_COLUMNS).to(device)

    print("Pos weight:")
    for label, weight in zip(LABEL_COLUMNS, pos_weight.detach().cpu().numpy().tolist()):
        print(f"  {label}: {weight:.4f}")

    try:
        model = build_model(
            model_name=MODEL_NAME,
            num_classes=len(LABEL_COLUMNS),
            pretrained=PRETRAINED,
        ).to(device)
        print("Pretrained:", PRETRAINED)
    except Exception as e:
        print("Pretrained model load failed. Falling back to pretrained=False.")
        print("Reason:", repr(e))
        model = build_model(
            model_name=MODEL_NAME,
            num_classes=len(LABEL_COLUMNS),
            pretrained=False,
        ).to(device)
        print("Pretrained: False")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())

    print("Model:", MODEL_NAME)
    print("Experiment:", EXPERIMENT_NAME)
    print("Parameters:", total_params)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    best_any_auc = -1.0
    best_checkpoint_path = OUTPUT_MODEL_DIR / f"best_{EXPERIMENT_NAME}.pt"
    last_checkpoint_path = OUTPUT_MODEL_DIR / f"last_{EXPERIMENT_NAME}.pt"

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        eval_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = eval_result["val_loss"]
        any_auc = eval_result["any_auc"]
        any_ap = eval_result["any_average_precision"]

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"any_auc={any_auc} | "
            f"any_ap={any_ap}"
        )

        print("Per-label summary:")
        for row in eval_result["metric_rows"]:
            print(row)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "any_auc": any_auc,
            "any_average_precision": any_ap,
            "per_label": eval_result["per_label"],
        }
        history.append(epoch_record)

        checkpoint = {
            "model_name": MODEL_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "model_state_dict": model.state_dict(),
            "label_columns": LABEL_COLUMNS,
            "image_size": IMAGE_SIZE,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "any_auc": any_auc,
            "any_average_precision": any_ap,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "use_amp": USE_AMP,
            "parameters": total_params,
        }

        torch.save(checkpoint, last_checkpoint_path)

        if any_auc is not None and any_auc > best_any_auc:
            best_any_auc = any_auc
            torch.save(checkpoint, best_checkpoint_path)
            print("Saved new best checkpoint:", best_checkpoint_path)

    elapsed = time.time() - start_time

    history_path = OUTPUT_LOG_DIR / f"{EXPERIMENT_NAME}_training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary_path = OUTPUT_LOG_DIR / f"{EXPERIMENT_NAME}_summary.json"
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_any_auc": best_any_auc,
        "training_time_seconds": round(elapsed, 2),
        "parameters": total_params,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== TRAINING DONE =====")
    print("Training time seconds:", round(elapsed, 2))
    print("Best any AUC:", best_any_auc)
    print("Saved best checkpoint:", best_checkpoint_path)
    print("Saved last checkpoint:", last_checkpoint_path)
    print("Saved history:", history_path)
    print("Saved summary:", summary_path)

    print("\nSTATUS: DENSENET121 TRAINING PASSED")


if __name__ == "__main__":
    main()
