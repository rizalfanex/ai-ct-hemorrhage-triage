from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.dicom_utils import (
    CT_WINDOWS,
    apply_window,
    create_multi_window_image,
    normalize_to_uint8,
    summarize_image,
)


def create_synthetic_head_ct(size: int = 512) -> np.ndarray:
    """
    Create a synthetic CT-like HU image for testing windowing logic.
    This is not medical data. It only tests preprocessing code.
    """
    y, x = np.ogrid[:size, :size]
    center = size / 2

    skull_radius = size * 0.42
    brain_radius = size * 0.34
    bleed_radius = size * 0.045

    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    image = np.full((size, size), -1000.0, dtype=np.float32)
    image[distance <= skull_radius] = 900.0
    image[distance <= brain_radius] = 35.0

    bleed_x = int(size * 0.60)
    bleed_y = int(size * 0.45)
    bleed_distance = np.sqrt((x - bleed_x) ** 2 + (y - bleed_y) ** 2)
    image[bleed_distance <= bleed_radius] = 75.0

    noise = np.random.normal(loc=0.0, scale=4.0, size=(size, size)).astype(np.float32)
    image = image + noise

    return image.astype(np.float32)


def main() -> None:
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_hu = create_synthetic_head_ct()
    multi_window = create_multi_window_image(image_hu)
    multi_window_uint8 = normalize_to_uint8(multi_window)

    print("===== SYNTHETIC CT HU SUMMARY =====")
    print(summarize_image(image_hu))

    print("\n===== MULTI-WINDOW IMAGE SUMMARY =====")
    print(summarize_image(multi_window))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_hu, cmap="gray")
    axes[0].set_title("Synthetic HU")
    axes[0].axis("off")

    for idx, window_name in enumerate(["brain", "subdural", "bone"], start=1):
        center, width = CT_WINDOWS[window_name]
        windowed = apply_window(image_hu, center=center, width=width)
        axes[idx].imshow(windowed, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(f"{window_name}\nC={center}, W={width}")
        axes[idx].axis("off")

    fig.tight_layout()
    fig_path = output_dir / "test_ct_windowing.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    rgb_path = output_dir / "test_multi_window_rgb.png"
    plt.imsave(rgb_path, multi_window_uint8)

    print("\nSaved:")
    print(fig_path)
    print(rgb_path)

    assert image_hu.shape == (512, 512)
    assert multi_window.shape == (512, 512, 3)
    assert multi_window.min() >= 0.0
    assert multi_window.max() <= 1.0

    print("\nSTATUS: CT WINDOWING TEST PASSED")


if __name__ == "__main__":
    main()
