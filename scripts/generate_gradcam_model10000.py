from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.datasets.rsna_dataset import RSNADicomDataset, LABEL_COLUMNS
from src.models.classifier import build_model


MANIFEST_PATH = Path("data/processed/rsna_holdout_5000_manifest.csv")
DICOM_DIR = Path("data/processed/rsna_holdout_5000_dicom")
CHECKPOINT_PATH = Path("outputs/models/best_efficientnet_b0_rsna_10000.pt")
PRED_CSV = Path("outputs/logs/holdout_5000_predictions_model10000.csv")

OUTPUT_DIR = Path("outputs/figures/gradcam_model10000")

IMAGE_SIZE = 224
TARGET_LABEL = "any"
N_TRUE_POSITIVE = 4
N_FALSE_PRIORITY = 2
N_LOW_SCORE_POSITIVE = 2


def get_target_layer(model):
    """
    Try to locate the last convolutional layer for EfficientNet-style timm models.
    """
    backbone = model.model

    if hasattr(backbone, "conv_head"):
        return backbone.conv_head

    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]

    raise RuntimeError("Could not identify target layer for Grad-CAM.")


def build_case_selection(pred_df: pd.DataFrame) -> pd.DataFrame:
    true_positive_high = (
        pred_df[pred_df["true_any"] == 1]
        .sort_values("prob_any", ascending=False)
        .head(N_TRUE_POSITIVE)
        .copy()
    )
    true_positive_high["case_group"] = "high_score_true_positive"

    false_priority = (
        pred_df[pred_df["true_any"] == 0]
        .sort_values("prob_any", ascending=False)
        .head(N_FALSE_PRIORITY)
        .copy()
    )
    false_priority["case_group"] = "high_score_false_priority"

    low_score_positive = (
        pred_df[pred_df["true_any"] == 1]
        .sort_values("prob_any", ascending=True)
        .head(N_LOW_SCORE_POSITIVE)
        .copy()
    )
    low_score_positive["case_group"] = "low_score_positive"

    selected = pd.concat(
        [true_positive_high, false_priority, low_score_positive],
        axis=0,
    ).reset_index(drop=True)

    return selected


def tensor_to_rgb_float(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert [3, H, W] tensor in 0-1 range into [H, W, 3] numpy RGB float.
    """
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image


def make_single_gradcam_figure(
    model,
    cam,
    dataset,
    selected_row,
    manifest_lookup,
    device,
    index: int,
) -> dict:
    image_id = selected_row["image_id"]
    manifest_idx = manifest_lookup[image_id]

    item = dataset[manifest_idx]

    image_tensor = item["image"]
    labels = item["labels"]

    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    label_index = LABEL_COLUMNS.index(TARGET_LABEL)

    targets = [ClassifierOutputTarget(label_index)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]

    rgb_img = tensor_to_rgb_float(image_tensor)
    cam_overlay = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    true_labels = {
        label: int(labels[i].item())
        for i, label in enumerate(LABEL_COLUMNS)
    }

    pred_scores = {
        label: float(probs[i])
        for i, label in enumerate(LABEL_COLUMNS)
    }

    case_group = selected_row["case_group"]
    true_any = int(selected_row["true_any"])
    prob_any = float(selected_row["prob_any"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Multi-window CT")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(cam_overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    subtitle = (
        f"{image_id} | {case_group}\n"
        f"true_any={true_any} | prob_any={prob_any:.4f}"
    )
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{index:02d}_{case_group}_{image_id}_gradcam.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved:", out_path)
    print("  true labels:", true_labels)
    print("  pred scores:", pred_scores)

    return {
        "image_id": image_id,
        "case_group": case_group,
        "true_any": true_any,
        "prob_any": prob_any,
        "figure_path": str(out_path),
        **{f"true_{k}": v for k, v in true_labels.items()},
        **{f"prob_{k}": v for k, v in pred_scores.items()},
    }


def main():
    print("===== GRAD-CAM EXPLAINABILITY: MODEL 10000 =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {PRED_CSV}")

    pred_df = pd.read_csv(PRED_CSV)
    selected_df = build_case_selection(pred_df)

    print("\n===== SELECTED CASES =====")
    print(selected_df[["image_id", "case_group", "true_any", "prob_any"]])

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        model_name=checkpoint["model_name"],
        num_classes=len(checkpoint["label_columns"]),
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = get_target_layer(model)
    print("\nTarget layer:", target_layer)

    dataset = RSNADicomDataset(
        manifest_path=MANIFEST_PATH,
        dicom_dir=DICOM_DIR,
        image_size=IMAGE_SIZE,
    )

    manifest_lookup = {
        str(row["image_id"]): idx
        for idx, row in dataset.df.iterrows()
    }

    target_layers = [target_layer]

    results = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx, row in selected_df.iterrows():
            result = make_single_gradcam_figure(
                model=model,
                cam=cam,
                dataset=dataset,
                selected_row=row,
                manifest_lookup=manifest_lookup,
                device=device,
                index=idx,
            )
            results.append(result)

    result_df = pd.DataFrame(results)
    result_csv = OUTPUT_DIR / "gradcam_selected_cases.csv"
    result_df.to_csv(result_csv, index=False)

    print("\nSaved Grad-CAM case table:", result_csv)
    print("\nSTATUS: GRAD-CAM GENERATION PASSED")


if __name__ == "__main__":
    main()
