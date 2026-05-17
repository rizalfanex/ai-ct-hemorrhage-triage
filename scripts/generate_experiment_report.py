from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = Path("outputs/logs")
FIG_DIR = Path("outputs/figures/experiment_summary")
REPORT_DIR = Path("outputs/reports")

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_2000_JSON = LOG_DIR / "holdout_5000_eval_metrics.json"
MODEL_10000_JSON = LOG_DIR / "holdout_5000_eval_metrics_model10000.json"

TRIAGE_2000_CSV = LOG_DIR / "holdout_5000_triage_simulation.csv"
TRIAGE_10000_CSV = LOG_DIR / "holdout_5000_triage_simulation_model10000.csv"

THRESHOLD_2000_CSV = LOG_DIR / "holdout_5000_threshold_sweep_any.csv"
THRESHOLD_10000_CSV = LOG_DIR / "holdout_5000_threshold_sweep_any_model10000.csv"

OUT_COMPARISON_CSV = REPORT_DIR / "model_comparison_holdout_5000.csv"
OUT_SUBTYPE_CSV = REPORT_DIR / "subtype_auc_ap_model10000.csv"
OUT_MARKDOWN = REPORT_DIR / "experiment_summary.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("===== BUILD EXPERIMENT SUMMARY =====")

    model_2000 = load_json(MODEL_2000_JSON)
    model_10000 = load_json(MODEL_10000_JSON)

    comparison_rows = []

    for name, data in [
        ("EfficientNet-B0 train_2000", model_2000),
        ("EfficientNet-B0 train_10000", model_10000),
    ]:
        any_metrics = data["per_label"]["any"]

        comparison_rows.append({
            "model": name,
            "train_size": 2000 if "2000" in name else 10000,
            "holdout_size": data["holdout_size"],
            "holdout_any_prevalence": data["holdout_any_prevalence"],
            "any_auc": any_metrics["auc"],
            "any_average_precision": any_metrics["average_precision"],
            "best_f1_threshold": data["best_f1_threshold_any"]["threshold"],
            "best_f1": data["best_f1_threshold_any"]["f1"],
            "best_recall90_threshold": data["best_threshold_with_recall_at_least_90"]["threshold"],
            "best_recall90_precision": data["best_threshold_with_recall_at_least_90"]["precision"],
            "best_recall90_recall": data["best_threshold_with_recall_at_least_90"]["recall"],
            "best_recall90_predicted_positive_rate": data["best_threshold_with_recall_at_least_90"]["predicted_positive_rate"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_COMPARISON_CSV, index=False)

    print("\n===== MODEL COMPARISON =====")
    print(comparison_df)

    subtype_rows = []
    for label, metrics in model_10000["per_label"].items():
        subtype_rows.append({
            "label": label,
            "positive_count": metrics["positive_count"],
            "prevalence": metrics["prevalence"],
            "auc": metrics["auc"],
            "average_precision": metrics["average_precision"],
        })

    subtype_df = pd.DataFrame(subtype_rows)
    subtype_df.to_csv(OUT_SUBTYPE_CSV, index=False)

    print("\n===== MODEL 10000 SUBTYPE METRICS =====")
    print(subtype_df)

    triage_2000 = pd.read_csv(TRIAGE_2000_CSV)
    triage_10000 = pd.read_csv(TRIAGE_10000_CSV)

    threshold_2000 = pd.read_csv(THRESHOLD_2000_CSV)
    threshold_10000 = pd.read_csv(THRESHOLD_10000_CSV)

    print("\n===== CREATE FIGURES =====")

    # Figure 1: AUC/AP comparison
    fig = plt.figure(figsize=(8, 5))
    x = range(len(comparison_df))
    width = 0.35

    plt.bar([i - width / 2 for i in x], comparison_df["any_auc"], width=width, label="AUC")
    plt.bar([i + width / 2 for i in x], comparison_df["any_average_precision"], width=width, label="Average Precision")

    plt.xticks(list(x), comparison_df["model"], rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Holdout Performance Comparison")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR / "model_auc_ap_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Figure 2: Triage recall curve
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        triage_2000["top_percent_workload"],
        triage_2000["recall_at_k"],
        marker="o",
        label="Train 2,000"
    )
    plt.plot(
        triage_10000["top_percent_workload"],
        triage_10000["recall_at_k"],
        marker="o",
        label="Train 10,000"
    )

    plt.xlabel("Top % Workload Prioritized")
    plt.ylabel("Hemorrhage Recall Captured")
    plt.ylim(0, 1.05)
    plt.title("AI Triage Simulation on Natural Holdout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIG_DIR / "triage_recall_curve.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Figure 3: Precision at top workload
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        triage_2000["top_percent_workload"],
        triage_2000["precision_at_k"],
        marker="o",
        label="Train 2,000"
    )
    plt.plot(
        triage_10000["top_percent_workload"],
        triage_10000["precision_at_k"],
        marker="o",
        label="Train 10,000"
    )

    plt.xlabel("Top % Workload Prioritized")
    plt.ylabel("Precision at Prioritized Workload")
    plt.ylim(0, 1.05)
    plt.title("Precision of AI-Prioritized Cases")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIG_DIR / "triage_precision_curve.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Figure 4: Threshold recall vs predicted workload
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        threshold_10000["threshold"],
        threshold_10000["recall"],
        marker="o",
        label="Recall"
    )
    plt.plot(
        threshold_10000["threshold"],
        threshold_10000["predicted_positive_rate"],
        marker="o",
        label="Predicted Positive Rate / Workload"
    )
    plt.plot(
        threshold_10000["threshold"],
        threshold_10000["precision"],
        marker="o",
        label="Precision"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.ylim(0, 1.05)
    plt.title("Threshold Trade-off: Model 10,000 on Natural Holdout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIG_DIR / "threshold_tradeoff_model10000.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Figure 5: subtype AUC/AP
    fig = plt.figure(figsize=(9, 5))

    plot_df = subtype_df.copy()
    x = range(len(plot_df))
    width = 0.35

    plt.bar([i - width / 2 for i in x], plot_df["auc"], width=width, label="AUC")
    plt.bar([i + width / 2 for i in x], plot_df["average_precision"], width=width, label="Average Precision")

    plt.xticks(list(x), plot_df["label"], rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Per-Label Performance: Model 10,000")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR / "subtype_auc_ap_model10000.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    lines = []
    lines.append("# Experiment Summary: RSNA Intracranial Hemorrhage Triage")
    lines.append("")
    lines.append("## Model Comparison on Natural Holdout 5,000")
    lines.append("")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Model 10,000 Per-Label Metrics")
    lines.append("")
    lines.append(subtype_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Key Interpretation")
    lines.append("")
    lines.append("- Increasing training data from 2,000 to 10,000 DICOM improved any-hemorrhage AUC on natural holdout.")
    lines.append("- The 10,000-sample model achieved stronger triage ranking performance.")
    lines.append("- At threshold 0.25, the 10,000 model reached recall above 0.90 with substantially lower priority workload than the 2,000 model.")
    lines.append("- The model is promising for research-grade triage simulation, but it is not clinically validated.")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for figure in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- {figure}")

    OUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved comparison CSV:", OUT_COMPARISON_CSV)
    print("Saved subtype CSV:", OUT_SUBTYPE_CSV)
    print("Saved markdown report:", OUT_MARKDOWN)
    print("Saved figures:")
    for figure in sorted(FIG_DIR.glob("*.png")):
        print(" ", figure)

    print("\nSTATUS: EXPERIMENT REPORT GENERATION PASSED")


if __name__ == "__main__":
    main()
