from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PRED_CSV = Path("outputs/logs/holdout_5000_predictions_model10000.csv")

OUT_DIR = Path("outputs/error_analysis")
PRIVATE_DIR = OUT_DIR / "private"
FIG_DIR = Path("outputs/figures/error_analysis")
REPORT_DIR = Path("outputs/reports")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.25

LABELS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

OUT_SUMMARY_JSON = REPORT_DIR / "error_analysis_summary_model10000_threshold025.json"
OUT_PUBLIC_SUMMARY_CSV = REPORT_DIR / "error_analysis_public_summary_threshold025.csv"
OUT_SUBTYPE_MISS_CSV = REPORT_DIR / "error_analysis_false_negative_subtype_summary_threshold025.csv"

PRIVATE_CASES_CSV = PRIVATE_DIR / "error_cases_with_image_ids_threshold025.csv"


def classify_error(row: pd.Series) -> str:
    y_true = int(row["true_any"])
    y_pred = int(row["pred_any"])

    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    if y_true == 1 and y_pred == 0:
        return "false_negative"

    raise ValueError("Unexpected label combination")


def build_probability_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group_name, group_df in df.groupby("error_group"):
        probs = group_df["prob_any"].values.astype(float)

        rows.append({
            "error_group": group_name,
            "count": int(len(group_df)),
            "prob_mean": float(np.mean(probs)),
            "prob_std": float(np.std(probs)),
            "prob_min": float(np.min(probs)),
            "prob_p25": float(np.percentile(probs, 25)),
            "prob_median": float(np.percentile(probs, 50)),
            "prob_p75": float(np.percentile(probs, 75)),
            "prob_max": float(np.max(probs)),
        })

    return pd.DataFrame(rows)


def build_subtype_miss_summary(df: pd.DataFrame) -> pd.DataFrame:
    fn_df = df[df["error_group"] == "false_negative"].copy()
    tp_df = df[df["error_group"] == "true_positive"].copy()
    positive_df = df[df["true_any"] == 1].copy()

    rows = []

    for label in LABELS:
        total_positive_subtype = int(positive_df[f"true_{label}"].sum())
        missed_subtype = int(fn_df[f"true_{label}"].sum())
        captured_subtype = int(tp_df[f"true_{label}"].sum())

        rows.append({
            "subtype": label,
            "positive_count_in_holdout": total_positive_subtype,
            "captured_count_at_threshold": captured_subtype,
            "missed_count_at_threshold": missed_subtype,
            "miss_rate_within_subtype": float(missed_subtype / max(total_positive_subtype, 1)),
            "capture_rate_within_subtype": float(captured_subtype / max(total_positive_subtype, 1)),
        })

    return pd.DataFrame(rows)


def plot_probability_histogram(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 6))

    for group in ["true_positive", "false_positive", "true_negative", "false_negative"]:
        group_df = df[df["error_group"] == group]
        if len(group_df) == 0:
            continue

        plt.hist(
            group_df["prob_any"].values,
            bins=30,
            alpha=0.45,
            label=f"{group} (n={len(group_df)})",
        )

    plt.axvline(THRESHOLD, linestyle="--", label=f"Threshold = {THRESHOLD}")
    plt.xlabel("Predicted probability for any hemorrhage")
    plt.ylabel("Case count")
    plt.title("Probability Distribution by Error Group")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "probability_distribution_by_error_group_threshold025.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_subtype_miss(subtype_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8, 5))

    x = np.arange(len(subtype_df))

    plt.bar(x, subtype_df["miss_rate_within_subtype"])
    plt.xticks(x, subtype_df["subtype"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Miss rate at threshold 0.25")
    plt.title("False Negative Rate by Hemorrhage Subtype")
    plt.tight_layout()

    out_path = FIG_DIR / "false_negative_rate_by_subtype_threshold025.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 5))

    plt.imshow(cm)
    plt.title("Error Analysis Confusion Matrix\nAny Hemorrhage, Threshold 0.25")
    plt.xticks([0, 1], ["Pred Normal", "Pred Positive"])
    plt.yticks([0, 1], ["True Normal", "True Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    plt.colorbar()
    plt.tight_layout()

    out_path = FIG_DIR / "error_analysis_confusion_matrix_threshold025.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    print("===== ERROR ANALYSIS MODEL 10000 THRESHOLD 0.25 =====")

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    print("Prediction shape:", df.shape)

    df["pred_any"] = (df["prob_any"] >= THRESHOLD).astype(int)
    df["error_group"] = df.apply(classify_error, axis=1)

    y_true = df["true_any"].values.astype(int)
    y_pred = df["pred_any"].values.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix:")
    print(cm)

    error_counts = df["error_group"].value_counts().to_dict()
    print("Error group counts:")
    print(error_counts)

    probability_df = build_probability_distribution(df)
    print("\\nProbability distribution summary:")
    print(probability_df)

    subtype_miss_df = build_subtype_miss_summary(df)
    print("\\nSubtype false negative summary:")
    print(subtype_miss_df)

    private_cols = [
        "image_id",
        "true_any",
        "pred_any",
        "prob_any",
        "error_group",
    ]

    for label in LABELS:
        private_cols.append(f"true_{label}")
        private_cols.append(f"prob_{label}")

    df[private_cols].sort_values(
        ["error_group", "prob_any"],
        ascending=[True, False],
    ).to_csv(PRIVATE_CASES_CSV, index=False)

    public_summary_rows = [
        {
            "threshold": THRESHOLD,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "false_negative_rate": float(fn / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "prioritized_workload_rate": float((tp + fp) / len(df)),
        }
    ]

    public_summary_df = pd.DataFrame(public_summary_rows)
    public_summary_df.to_csv(OUT_PUBLIC_SUMMARY_CSV, index=False)
    subtype_miss_df.to_csv(OUT_SUBTYPE_MISS_CSV, index=False)

    summary = {
        "threshold": THRESHOLD,
        "prediction_csv": str(PRED_CSV),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": public_summary_rows[0],
        "error_group_counts": {
            key: int(value)
            for key, value in error_counts.items()
        },
        "probability_distribution_by_group": probability_df.to_dict(orient="records"),
        "false_negative_subtype_summary": subtype_miss_df.to_dict(orient="records"),
        "private_case_file_not_for_github": str(PRIVATE_CASES_CSV),
    }

    OUT_SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_probability_histogram(df)
    plot_subtype_miss(subtype_miss_df)
    plot_confusion_matrix(cm)

    print("\\nSaved public summary CSV:", OUT_PUBLIC_SUMMARY_CSV)
    print("Saved subtype miss CSV:", OUT_SUBTYPE_MISS_CSV)
    print("Saved summary JSON:", OUT_SUMMARY_JSON)
    print("Saved private case-level CSV:", PRIVATE_CASES_CSV)
    print("Saved figures:")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)

    print("\\nSTATUS: ERROR ANALYSIS GENERATED")


if __name__ == "__main__":
    main()
