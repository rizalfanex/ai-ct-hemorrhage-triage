from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import pydicom

from src.preprocessing.dicom_utils import (
    CT_WINDOWS,
    apply_window,
    create_multi_window_image,
    dicom_to_hu,
    normalize_to_uint8,
    read_dicom,
    summarize_image,
)


SAMPLE_DIR = Path("data/sample/rsna_dicom")
MANIFEST_PATH = Path("data/sample/rsna_sample_manifest.csv")
OUTPUT_DIR = Path("outputs/figures/real_dicom_windowing")


LABEL_COLUMNS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def format_labels(row: pd.Series) -> str:
    active = [label for label in LABEL_COLUMNS if int(row[label]) == 1]
    if not active:
        return "normal"
    return ", ".join(active)


def inspect_dicom_metadata(dcm: pydicom.dataset.FileDataset) -> dict:
    return {
        "Rows": getattr(dcm, "Rows", None),
        "Columns": getattr(dcm, "Columns", None),
        "BitsStored": getattr(dcm, "BitsStored", None),
        "PixelRepresentation": getattr(dcm, "PixelRepresentation", None),
        "RescaleSlope": getattr(dcm, "RescaleSlope", None),
        "RescaleIntercept": getattr(dcm, "RescaleIntercept", None),
        "WindowCenter": getattr(dcm, "WindowCenter", None),
        "WindowWidth": getattr(dcm, "WindowWidth", None),
        "PhotometricInterpretation": getattr(dcm, "PhotometricInterpretation", None),
    }


def plot_case(row: pd.Series, index: int) -> None:
    image_id = row["image_id"]
    dcm_path = SAMPLE_DIR / f"{image_id}.dcm"

    if not dcm_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")

    dcm = read_dicom(dcm_path)
    image_hu = dicom_to_hu(dcm)

    brain = apply_window(image_hu, *CT_WINDOWS["brain"])
    subdural = apply_window(image_hu, *CT_WINDOWS["subdural"])
    bone = apply_window(image_hu, *CT_WINDOWS["bone"])
    multi = create_multi_window_image(image_hu)
    multi_uint8 = normalize_to_uint8(multi)

    label_text = format_labels(row)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image_hu, cmap="gray")
    axes[0].set_title("Raw HU")
    axes[0].axis("off")

    axes[1].imshow(brain, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Brain window\nC=40 W=80")
    axes[1].axis("off")

    axes[2].imshow(subdural, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Subdural window\nC=80 W=200")
    axes[2].axis("off")

    axes[3].imshow(bone, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Bone window\nC=600 W=2800")
    axes[3].axis("off")

    axes[4].imshow(multi_uint8)
    axes[4].set_title("3-channel multi-window")
    axes[4].axis("off")

    fig.suptitle(f"{image_id} | {label_text}", fontsize=11)
    fig.tight_layout()

    safe_label = "positive" if int(row["any"]) == 1 else "normal"
    out_path = OUTPUT_DIR / f"{index:02d}_{safe_label}_{image_id}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("Metadata:", inspect_dicom_metadata(dcm))
    print("HU summary:", summarize_image(image_hu))
    print("Multi-window summary:", summarize_image(multi))
    print("-" * 80)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    print("===== REAL DICOM WINDOWING TEST =====")
    print("Manifest shape:", manifest.shape)

    normal_cases = manifest[manifest["any"] == 0].head(5)
    positive_cases = manifest[manifest["any"] == 1].head(5)

    selected = pd.concat([normal_cases, positive_cases], axis=0).reset_index(drop=True)

    print("Selected cases:", len(selected))
    print(selected[["image_id", "any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]])

    for idx, row in selected.iterrows():
        plot_case(row, idx)

    print("\nSTATUS: REAL DICOM WINDOWING TEST PASSED")


if __name__ == "__main__":
    main()
