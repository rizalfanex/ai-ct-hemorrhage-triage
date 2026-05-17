from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PRED_CSV = Path("outputs/logs/holdout_5000_predictions_model10000.csv")

REPORT_DIR = Path("outputs/reports")
FIG_DIR = Path("outputs/figures/paper_metrics")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CI_CSV = REPORT_DIR / "bootstrap_ci_model10000.csv"
OUT_CI_JSON = REPORT_DIR / "bootstrap_ci_model10000.json"
OUT_THRESHOLD_CI_CSV = REPORT_DIR / "bootstrap_threshold_ci_any_model10000.csv"
OUT_FIG = FIG_DIR / "bootstrap_ci_auc_ap_model10000.png"

LABELS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

N_BOOTSTRAP = 2000
RANDOM_STATE = 2026

TRIAGE_THRESHOLD = 0.25
BEST_F1_THRESHOLD = 0.90


def percentile_ci(values: list[float], alpha: float = 0.95) -> tuple[float, float]:
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(values, lower_q)), float(np.percentile(values, upper_q))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)

    sampled = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled)

    return sampled


def bootstrap_auc_ap(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    print("===== BOOTSTRAP AUC/AP =====")

    for label in LABELS:
        y_true = df[f"true_{label}"].values.astype(int)
        y_prob = df[f"prob_{label}"].values.astype(float)

        point_auc = safe_auc(y_true, y_prob)
        point_ap = safe_ap(y_true, y_prob)

        auc_values = []
        ap_values = []

        for _ in range(N_BOOTSTRAP):
            idx = stratified_bootstrap_indices(y_true, rng)

            y_true_b = y_true[idx]
            y_prob_b = y_prob[idx]

            auc_b = safe_auc(y_true_b, y_prob_b)
            ap_b = safe_ap(y_true_b, y_prob_b)

            if auc_b is not None:
                auc_values.append(auc_b)
            if ap_b is not None:
                ap_values.append(ap_b)

        auc_low, auc_high = percentile_ci(auc_values)
        ap_low, ap_high = percentile_ci(ap_values)

        row = {
            "label": label,
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auc": point_auc,
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            "average_precision": point_ap,
            "average_precision_ci_low": ap_low,
            "average_precision_ci_high": ap_high,
            "n_bootstrap": N_BOOTSTRAP,
        }

        rows.append(row)
        print(row)

    return pd.DataFrame(rows)


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "predicted_positive_rate": float((tp + fp) / len(y_true)),
    }


def bootstrap_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 99)

    y_true = df["true_any"].values.astype(int)
    y_prob = df["prob_any"].values.astype(float)

    rows = []

    for threshold_name, threshold in [
        ("triage_recall90", TRIAGE_THRESHOLD),
        ("best_f1", BEST_F1_THRESHOLD),
    ]:
        point = compute_threshold_metrics(y_true, y_prob, threshold)

        boot_values = {
            "precision": [],
            "recall": [],
            "f1": [],
            "false_negative_rate": [],
            "false_positive_rate": [],
            "predicted_positive_rate": [],
        }

        for _ in range(N_BOOTSTRAP):
            idx = stratified_bootstrap_indices(y_true, rng)

            y_true_b = y_true[idx]
            y_prob_b = y_prob[idx]

            metrics_b = compute_threshold_metrics(y_true_b, y_prob_b, threshold)

            for key in boot_values:
                boot_values[key].append(metrics_b[key])

        row = {
            "threshold_name": threshold_name,
            "threshold": threshold,
            "tn": point["tn"],
            "fp": point["fp"],
            "fn": point["fn"],
            "tp": point["tp"],
        }

        for key, values in boot_values.items():
            low, high = percentile_ci(values)
            row[key] = point[key]
            row[f"{key}_ci_low"] = low
            row[f"{key}_ci_high"] = high

        rows.append(row)
        print(row)

    return pd.DataFrame(rows)


def plot_ci(ci_df: pd.DataFrame) -> None:
    labels = ci_df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    auc_values = ci_df["auc"].values
    auc_err_low = auc_values - ci_df["auc_ci_low"].values
    auc_err_high = ci_df["auc_ci_high"].values - auc_values

    ap_values = ci_df["average_precision"].values
    ap_err_low = ap_values - ci_df["average_precision_ci_low"].values
    ap_err_high = ci_df["average_precision_ci_high"].values - ap_values

    fig = plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        auc_values,
        width=width,
        yerr=[auc_err_low, auc_err_high],
        capsize=4,
        label="AUC",
    )

    plt.bar(
        x + width / 2,
        ap_values,
        width=width,
        yerr=[ap_err_low, ap_err_high],
        capsize=4,
        label="Average Precision",
    )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score with 95% Bootstrap CI")
    plt.title("Holdout Performance with Bootstrap Confidence Intervals")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=250)
    plt.close(fig)


def main() -> None:
    print("===== BOOTSTRAP CONFIDENCE INTERVALS: MODEL 10000 =====")

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    print("Prediction shape:", df.shape)
    print("N bootstrap:", N_BOOTSTRAP)

    ci_df = bootstrap_auc_ap(df)
    ci_df.to_csv(OUT_CI_CSV, index=False)

    print("\n===== BOOTSTRAP THRESHOLD METRICS =====")
    threshold_ci_df = bootstrap_threshold_metrics(df)
    threshold_ci_df.to_csv(OUT_THRESHOLD_CI_CSV, index=False)

    plot_ci(ci_df)

    summary = {
        "prediction_csv": str(PRED_CSV),
        "n_bootstrap": N_BOOTSTRAP,
        "auc_ap_ci_csv": str(OUT_CI_CSV),
        "threshold_ci_csv": str(OUT_THRESHOLD_CI_CSV),
        "figure": str(OUT_FIG),
        "labels": ci_df.to_dict(orient="records"),
        "thresholds": threshold_ci_df.to_dict(orient="records"),
    }

    OUT_CI_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved CI CSV:", OUT_CI_CSV)
    print("Saved threshold CI CSV:", OUT_THRESHOLD_CI_CSV)
    print("Saved CI JSON:", OUT_CI_JSON)
    print("Saved CI figure:", OUT_FIG)

    print("\nSTATUS: BOOTSTRAP CONFIDENCE INTERVALS GENERATED")


if __name__ == "__main__":
    main()
