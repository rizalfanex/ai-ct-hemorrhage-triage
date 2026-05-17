from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS


MANIFEST_PATH = Path("data/sample/rsna_sample_manifest.csv")
DICOM_DIR = Path("data/sample/rsna_dicom")


def main() -> None:
    print("===== RSNA DATASET LOADER TEST =====")

    dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=224,
    )

    print("Dataset length:", len(dataset))
    print("Label columns:", LABEL_COLUMNS)

    first_item = dataset[0]

    print("\n===== FIRST ITEM =====")
    print("image_id:", first_item["image_id"])
    print("image shape:", first_item["image"].shape)
    print("image dtype:", first_item["image"].dtype)
    print("image min:", float(first_item["image"].min()))
    print("image max:", float(first_item["image"].max()))
    print("labels:", first_item["labels"])
    print("labels shape:", first_item["labels"].shape)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    print("\n===== FIRST BATCH =====")
    print("batch image shape:", batch["image"].shape)
    print("batch labels shape:", batch["labels"].shape)
    print("batch image dtype:", batch["image"].dtype)
    print("batch labels dtype:", batch["labels"].dtype)
    print("batch image min:", float(batch["image"].min()))
    print("batch image max:", float(batch["image"].max()))
    print("batch image_ids:", batch["image_id"])

    if torch.cuda.is_available():
        images_gpu = batch["image"].cuda(non_blocking=True)
        labels_gpu = batch["labels"].cuda(non_blocking=True)

        print("\n===== GPU TRANSFER TEST =====")
        print("images gpu shape:", images_gpu.shape)
        print("labels gpu shape:", labels_gpu.shape)
        print("gpu:", torch.cuda.get_device_name(0))
        print("GPU transfer status: PASSED")
    else:
        print("\nCUDA not available. Skipping GPU transfer test.")

    assert batch["image"].shape == (8, 3, 224, 224)
    assert batch["labels"].shape == (8, 6)
    assert batch["image"].min() >= 0.0
    assert batch["image"].max() <= 1.0

    print("\nSTATUS: RSNA DATASET LOADER TEST PASSED")


if __name__ == "__main__":
    main()
