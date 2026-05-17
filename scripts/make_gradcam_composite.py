from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


GRADCAM_DIR = Path("outputs/figures/gradcam_model10000")
OUT_DIR = Path("outputs/figures/paper_ready")

CASE_CSV = GRADCAM_DIR / "gradcam_selected_cases.csv"

OUT_PNG = OUT_DIR / "figure_gradcam_composite_model10000.png"
OUT_PDF = OUT_DIR / "figure_gradcam_composite_model10000.pdf"
OUT_CASE_TABLE = OUT_DIR / "figure_gradcam_cases_table.csv"


def shorten_group(group: str) -> str:
    mapping = {
        "high_score_true_positive": "High-score TP",
        "high_score_false_priority": "High-score FP",
        "low_score_positive": "Low-score FN",
    }
    return mapping.get(group, group)


def main() -> None:
    print("===== BUILD PAPER-READY GRAD-CAM COMPOSITE =====")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CASE_CSV.exists():
        raise FileNotFoundError(f"Missing case CSV: {CASE_CSV}")

    case_df = pd.read_csv(CASE_CSV)

    print("Case table shape:", case_df.shape)
    print(case_df[["image_id", "case_group", "true_any", "prob_any", "figure_path"]])

    n_cases = len(case_df)

    n_cols = 2
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 4.8 * n_rows),
    )

    axes = axes.flatten()

    for idx, row in case_df.iterrows():
        ax = axes[idx]

        fig_path = Path(row["figure_path"])

        if not fig_path.exists():
            raise FileNotFoundError(f"Missing Grad-CAM image: {fig_path}")

        image = Image.open(fig_path).convert("RGB")

        ax.imshow(image)
        ax.axis("off")

        group = shorten_group(str(row["case_group"]))
        image_id = row["image_id"]
        true_any = int(row["true_any"])
        prob_any = float(row["prob_any"])

        title = f"{group} | {image_id} | true_any={true_any} | p(any)={prob_any:.3f}"
        ax.set_title(title, fontsize=10)

    for idx in range(n_cases, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Grad-CAM Explainability Examples for Intracranial Hemorrhage Triage Model",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    fig.savefig(OUT_PNG, dpi=250)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    summary_cols = [
        "image_id",
        "case_group",
        "true_any",
        "prob_any",
        "true_epidural",
        "true_intraparenchymal",
        "true_intraventricular",
        "true_subarachnoid",
        "true_subdural",
        "prob_epidural",
        "prob_intraparenchymal",
        "prob_intraventricular",
        "prob_subarachnoid",
        "prob_subdural",
    ]

    existing_cols = [col for col in summary_cols if col in case_df.columns]
    case_df[existing_cols].to_csv(OUT_CASE_TABLE, index=False)

    print("\nSaved composite PNG:", OUT_PNG)
    print("Saved composite PDF:", OUT_PDF)
    print("Saved case table:", OUT_CASE_TABLE)

    print("\nSTATUS: GRADCAM COMPOSITE FIGURE PASSED")


if __name__ == "__main__":
    main()
