from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pydicom


CT_WINDOWS: Dict[str, Tuple[float, float]] = {
    "brain": (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "bone": (600.0, 2800.0),
}


def read_dicom(path: Union[str, Path]) -> pydicom.dataset.FileDataset:
    """
    Read a DICOM file safely.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    return pydicom.dcmread(str(path))


def dicom_to_hu(dicom: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert DICOM pixel array to Hounsfield Units (HU).

    HU = pixel_value * RescaleSlope + RescaleIntercept
    """
    image = dicom.pixel_array.astype(np.float32)

    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))

    hu_image = image * slope + intercept
    return hu_image.astype(np.float32)


def apply_window(
    image: np.ndarray,
    center: float,
    width: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply CT windowing to an HU image.

    Args:
        image: HU image.
        center: Window center.
        width: Window width.
        normalize: If True, output range becomes 0-1.

    Returns:
        Windowed image as float32.
    """
    lower = center - width / 2.0
    upper = center + width / 2.0

    windowed = np.clip(image, lower, upper)

    if normalize:
        windowed = (windowed - lower) / (upper - lower + 1e-8)

    return windowed.astype(np.float32)


def create_multi_window_image(image_hu: np.ndarray) -> np.ndarray:
    """
    Create 3-channel CT image using brain, subdural, and bone windows.

    Channel 0: brain window
    Channel 1: subdural window
    Channel 2: bone window
    """
    brain = apply_window(image_hu, *CT_WINDOWS["brain"])
    subdural = apply_window(image_hu, *CT_WINDOWS["subdural"])
    bone = apply_window(image_hu, *CT_WINDOWS["bone"])

    multi_window = np.stack([brain, subdural, bone], axis=-1)
    return multi_window.astype(np.float32)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized 0-1 image to uint8 0-255.
    """
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round()
    return image.astype(np.uint8)


def summarize_image(image: np.ndarray) -> dict:
    """
    Return basic statistics for debugging preprocessing.
    """
    return {
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }
