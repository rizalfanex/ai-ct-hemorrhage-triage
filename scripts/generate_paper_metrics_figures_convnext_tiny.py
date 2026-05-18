from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PRED_CSV = Path("outputs/logs/holdout_5000_predictions_convnext_tiny_10000.csv")
OUT_DIR = Path("outputs/figures/paper_metrics_convnext_tiny")
REPORT_DIR = Path("outputs/reports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

THRESHOLD_ANY_TRIAGE = 0.10
THRESHOLD_ANY_BEST_F1 = 0.65


def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def plot_any_roc_pr(df: pd.DataFrame) -> dict:
    y_true = df["true_any"].values.astype(int)
    y_prob = df["prob_any"].values.astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    roc_auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Any Hemorrhage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = OUT_DIR / "roc_curve_any_convnext_tiny_10000.png"
    plt.savefig(roc_path, dpi=250)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Any Hemorrhage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = OUT_DIR / "pr_curve_any_convnext_tiny_10000.png"
    plt.savefig(pr_path, dpi=250)
    plt.close(fig)

    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "roc_path": str(roc_path),
        "pr_path": str(pr_path),
    }


def plot_per_label_auc_ap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in LABELS:
        y_true = df[f"true_{label}"].values.astype(int)
        y_prob = df[f"prob_{label}"].values.astype(float)

        rows.append({
            "label": label,
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auc": safe_roc_auc(y_true, y_prob),
            "average_precision": safe_ap(y_true, y_prob),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_path = REPORT_DIR / "paper_metrics_per_label_auc_ap_convnext_tiny_10000.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fig = plt.figure(figsize=(9, 5))
    x = np.arange(len(metrics_df))
    width = 0.35

    plt.bar(x - width / 2, metrics_df["auc"], width=width, label="AUC")
    plt.bar(x + width / 2, metrics_df["average_precision"], width=width, label="Average Precision")
    plt.xticks(x, metrics_df["label"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Per-Label AUC and Average Precision")
    plt.legend()
    plt.tight_layout()

    fig_path = OUT_DIR / "per_label_auc_ap_convnext_tiny_10000.png"
    plt.savefig(fig_path, dpi=250)
    plt.close(fig)

    return metrics_df


def plot_calibration(df: pd.DataFrame) -> dict:
    y_true = df["true_any"].values.astype(int)
    y_prob = df["prob_any"].values.astype(float)

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_probability": prob_pred,
        "fraction_of_positives": prob_true,
    })

    calibration_path = REPORT_DIR / "calibration_curve_any_convnext_tiny_10000.csv"
    calibration_df.to_csv(calibration_path, index=False)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve: Any Hemorrhage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = OUT_DIR / "calibration_curve_any_convnext_tiny_10000.png"
    plt.savefig(fig_path, dpi=250)
    plt.close(fig)

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "brier_score": brier,
        "calibration_csv": str(calibration_path),
        "calibration_figure": str(fig_path),
    }


def plot_confusion_matrices(df: pd.DataFrame) -> dict:
    y_true = df["true_any"].values.astype(int)
    y_prob = df["prob_any"].values.astype(float)

    output = {}

    thresholds = {
        "triage_recall90_threshold_010": THRESHOLD_ANY_TRIAGE,
        "best_f1_threshold_065": THRESHOLD_ANY_BEST_F1,
    }

    for name, threshold in thresholds.items():
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fig = plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix: Any Hemorrhage\nThreshold = {threshold:.2f}")
        plt.xticks([0, 1], ["Pred Normal", "Pred Positive"])
        plt.yticks([0, 1], ["True Normal", "True Positive"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

        plt.colorbar()
        plt.tight_layout()

        fig_path = OUT_DIR / f"confusion_matrix_any_{name}.png"
        plt.savefig(fig_path, dpi=250)
        plt.close(fig)

        output[name] = {
            "threshold": threshold,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "false_negative_rate": float(fn / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "figure": str(fig_path),
        }

    return output


def main() -> None:
    print("===== PAPER METRICS FIGURES =====")

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    print("Prediction shape:", df.shape)
    print("Columns:", df.columns.tolist())

    any_result = plot_any_roc_pr(df)
    print("\nAny ROC/PR:")
    print(any_result)

    per_label_df = plot_per_label_auc_ap(df)
    print("\nPer-label metrics:")
    print(per_label_df)

    calibration_result = plot_calibration(df)
    print("\nCalibration:")
    print(calibration_result)

    cm_result = plot_confusion_matrices(df)
    print("\nConfusion matrices:")
    print(cm_result)

    summary = {
        "any_roc_pr": any_result,
        "calibration": calibration_result,
        "confusion_matrices": cm_result,
    }

    summary_path = REPORT_DIR / "paper_metrics_summary_convnext_tiny_10000.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved summary:", summary_path)

    print("\nGenerated figures:")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)

    print("\nSTATUS: PAPER METRICS FIGURES GENERATED")


if __name__ == "__main__":
    main()

