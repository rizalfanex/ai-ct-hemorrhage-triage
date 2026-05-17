from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/sample/rsna_sample_manifest.csv")
DICOM_DIR = Path("data/sample/rsna_dicom")
OUTPUT_DIR = Path("outputs/models")

MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-4
SEED = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

        if step == 1:
            print(f"Epoch {epoch} step {step}")
            print("  images:", tuple(images.shape))
            print("  labels:", tuple(labels.shape))
            print("  logits:", tuple(logits.shape))
            print("  loss:", float(loss.item()))

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def main() -> None:
    print("===== RSNA BASELINE TRAINING SMOKE TEST =====")

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

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

    print("Dataset length:", len(dataset))
    print("Train length:", len(train_dataset))
    print("Val length:", len(val_dataset))
    print("Labels:", LABEL_COLUMNS)

    model = build_model(
        model_name=MODEL_NAME,
        num_classes=len(LABEL_COLUMNS),
        pretrained=False,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    print("Model:", MODEL_NAME)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    elapsed = time.time() - start_time

    checkpoint_path = OUTPUT_DIR / "smoke_efficientnet_b0_rsna_sample.pt"

    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "label_columns": LABEL_COLUMNS,
            "image_size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "train_size": train_size,
            "val_size": val_size,
        },
        checkpoint_path,
    )

    print("Training time seconds:", round(elapsed, 2))
    print("Saved checkpoint:", checkpoint_path)

    print("\nSTATUS: BASELINE TRAINING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
