from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.dicom_utils import (
    create_multi_window_image,
    dicom_to_hu,
    read_dicom,
)


LABEL_COLUMNS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


class RSNADicomDataset(Dataset):
    """
    PyTorch Dataset for RSNA Intracranial Hemorrhage DICOM slices.

    Output:
        image: torch.FloatTensor, shape [3, image_size, image_size]
        labels: torch.FloatTensor, shape [6]
        image_id: str
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dicom_dir: str | Path,
        image_size: int = 224,
        label_columns: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dicom_dir = Path(dicom_dir)
        self.image_size = image_size
        self.label_columns = label_columns or LABEL_COLUMNS
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        if not self.dicom_dir.exists():
            raise FileNotFoundError(f"DICOM folder not found: {self.dicom_dir}")

        self.df = pd.read_csv(self.manifest_path)

        required_columns = ["image_id"] + self.label_columns
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in manifest: {missing_columns}")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> np.ndarray:
        dcm_path = self.dicom_dir / f"{image_id}.dcm"

        if not dcm_path.exists():
            raise FileNotFoundError(f"DICOM not found: {dcm_path}")

        dcm = read_dicom(dcm_path)
        image_hu = dicom_to_hu(dcm)
        image = create_multi_window_image(image_hu)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        return image

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        image = self._load_image(image_id)

        labels = row[self.label_columns].values.astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        labels = torch.from_numpy(labels)

        return {
            "image": image,
            "labels": labels,
            "image_id": image_id,
        }
