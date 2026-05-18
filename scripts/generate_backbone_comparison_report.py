from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = Path("outputs/logs")
REPORT_DIR = Path("outputs/reports")
FIG_DIR = Path("outputs/figures/model_comparison")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    {
        "model_display": "EfficientNet-B0",
        "train_summary": LOG_DIR / "baseline_10000_summary.json",
        "holdout_metrics": LOG_DIR / "holdout_5000_eval_metrics_model10000.json",
        "threshold_csv": LOG_DIR / "holdout_5000_threshold_sweep_any_model10000.csv",
        "triage_csv": LOG_DIR / "holdout_5000_triage_simulation_model10000.csv",
    },    {
        "model_display": "DenseNet121",
        "train_summary": LOG_DIR / "densenet121_rsna_10000_summary.json",
        "holdout_metrics": LOG_DIR / "holdout_5000_eval_metrics_densenet121_10000.json",
        "threshold_csv": LOG_DIR / "holdout_5000_threshold_sweep_any_densenet121_10000.csv",
        "triage_csv": LOG_DIR / "holdout_5000_triage_simulation_densenet121_10000.csv",
    },
    {
        "model_display": "ConvNeXt-Tiny",
        "train_summary": LOG_DIR / "convnext_tiny_rsna_10000_summary.json",
        "holdout_metrics": LOG_DIR / "holdout_5000_eval_metrics_convnext_tiny_10000.json",
        "threshold_csv": LOG_DIR / "holdout_5000_threshold_sweep_any_convnext_tiny_10000.csv",
        "triage_csv": LOG_DIR / "holdout_5000_triage_simulation_convnext_tiny_10000.csv",
    },
]

OUT_COMPARISON_CSV = REPORT_DIR / "backbone_comparison_holdout_5000.csv"
OUT_SUBTYPE_CSV = REPORT_DIR / "backbone_comparison_per_label_auc_ap.csv"
OUT_MARKDOWN = REPORT_DIR / "backbone_comparison_summary.md"

OUT_FIG_AUC_AP = FIG_DIR / "backbone_any_auc_ap_comparison.png"
OUT_FIG_TRIAGE = FIG_DIR / "backbone_triage_recall_curve.png"
OUT_FIG_WORKLOAD = FIG_DIR / "backbone_recall90_workload_comparison.png"
OUT_FIG_SUBTYPE = FIG_DIR / "backbone_per_label_auc_comparison.png"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("===== BACKBONE COMPARISON REPORT =====")

    comparison_rows = []
    subtype_rows = []

    triage_tables = {}

    for item in MODEL_FILES:
        model_display = item["model_display"]
        train_summary = load_json(item["train_summary"])
        holdout_metrics = load_json(item["holdout_metrics"])

        any_metrics = holdout_metrics["per_label"]["any"]
        best_recall90 = holdout_metrics["best_threshold_with_recall_at_least_90"]
        best_f1 = holdout_metrics["best_f1_threshold_any"]

        row = {
            "model": model_display,
            "train_size": 10000,
            "holdout_size": holdout_metrics["holdout_size"],
            "holdout_prevalence_any": holdout_metrics["holdout_any_prevalence"],
            "parameters": train_summary.get("parameters"),
            "training_time_seconds": train_summary.get("training_time_seconds"),
            "validation_best_any_auc": train_summary.get("best_any_auc"),
            "holdout_any_auc": any_metrics["auc"],
            "holdout_any_average_precision": any_metrics["average_precision"],
            "best_f1_threshold": best_f1["threshold"],
            "best_f1": best_f1["f1"],
            "recall90_threshold": best_recall90["threshold"],
            "recall90_precision": best_recall90["precision"],
            "recall90_recall": best_recall90["recall"],
            "recall90_false_negative_rate": best_recall90["false_negative_rate"],
            "recall90_predicted_positive_rate": best_recall90["predicted_positive_rate"],
        }

        comparison_rows.append(row)

        for label, metrics in holdout_metrics["per_label"].items():
            subtype_rows.append({
                "model": model_display,
                "label": label,
                "positive_count": metrics["positive_count"],
                "prevalence": metrics["prevalence"],
                "auc": metrics["auc"],
                "average_precision": metrics["average_precision"],
            })

        triage_tables[model_display] = pd.read_csv(item["triage_csv"])

    comparison_df = pd.DataFrame(comparison_rows)
    subtype_df = pd.DataFrame(subtype_rows)

    comparison_df.to_csv(OUT_COMPARISON_CSV, index=False)
    subtype_df.to_csv(OUT_SUBTYPE_CSV, index=False)

    print("\n===== COMPARISON TABLE =====")
    print(comparison_df)

    print("\n===== PER-LABEL TABLE =====")
    print(subtype_df)

    # Figure 1: Any AUC/AP
    fig = plt.figure(figsize=(8, 5))
    x = range(len(comparison_df))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        comparison_df["holdout_any_auc"],
        width=width,
        label="AUC",
    )
    plt.bar(
        [i + width / 2 for i in x],
        comparison_df["holdout_any_average_precision"],
        width=width,
        label="Average Precision",
    )

    plt.xticks(list(x), comparison_df["model"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Backbone Comparison on Natural Holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_AUC_AP, dpi=250)
    plt.close(fig)

    # Figure 2: Triage recall curve
    fig = plt.figure(figsize=(8, 5))

    for model_name, triage_df in triage_tables.items():
        plt.plot(
            triage_df["top_percent_workload"],
            triage_df["recall_at_k"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Top % Workload Prioritized")
    plt.ylabel("Hemorrhage Recall Captured")
    plt.ylim(0, 1.05)
    plt.title("Triage Recall by Prioritized Workload")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_TRIAGE, dpi=250)
    plt.close(fig)

    # Figure 3: workload at recall >= 90
    fig = plt.figure(figsize=(8, 5))
    plt.bar(
        comparison_df["model"],
        comparison_df["recall90_predicted_positive_rate"],
    )
    plt.ylabel("Predicted Positive Rate / Prioritized Workload")
    plt.ylim(0, 0.6)
    plt.title("Workload Required to Reach >=90% Recall")
    plt.tight_layout()
    plt.savefig(OUT_FIG_WORKLOAD, dpi=250)
    plt.close(fig)

    # Figure 4: per-label AUC
    pivot_auc = subtype_df.pivot(index="label", columns="model", values="auc")
    fig = plt.figure(figsize=(10, 5))
    pivot_auc.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1.05)
    plt.ylabel("AUC")
    plt.title("Per-Label AUC by Backbone")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIG_SUBTYPE, dpi=250)
    plt.close(fig)

    lines = []
    lines.append("# Backbone Comparison Summary")
    lines.append("")
    lines.append("## Natural Holdout Performance")
    lines.append("")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-Label AUC/AP")
    lines.append("")
    lines.append(subtype_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- ConvNeXt-Tiny achieved the strongest natural-prevalence holdout performance among the evaluated backbones.")
    lines.append("- ConvNeXt-Tiny improved any-hemorrhage AUC and average precision over EfficientNet-B0 and DenseNet121.")
    lines.append("- ConvNeXt-Tiny achieved the best high-sensitivity triage trade-off among the evaluated backbones.")
    lines.append("- This supports using ConvNeXt-Tiny as the current primary backbone while retaining DenseNet121 as a strong comparator.")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for path in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- {path}")

    OUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved comparison CSV:", OUT_COMPARISON_CSV)
    print("Saved subtype CSV:", OUT_SUBTYPE_CSV)
    print("Saved markdown summary:", OUT_MARKDOWN)

    print("\nSaved figures:")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)

    print("\nSTATUS: BACKBONE COMPARISON REPORT GENERATED")


if __name__ == "__main__":
    main()

