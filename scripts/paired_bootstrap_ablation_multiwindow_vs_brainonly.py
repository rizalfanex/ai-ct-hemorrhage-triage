from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


LOG_DIR = Path("outputs/logs")
REPORT_DIR = Path("outputs/reports")
FIG_DIR = Path("outputs/figures/ablation_study")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRED_MULTI = LOG_DIR / "holdout_5000_predictions_convnext_tiny_10000.csv"
PRED_BRAIN = LOG_DIR / "holdout_5000_predictions_convnext_tiny_brain_only_10000.csv"

METRICS_MULTI = LOG_DIR / "holdout_5000_eval_metrics_convnext_tiny_10000.json"
METRICS_BRAIN = LOG_DIR / "holdout_5000_eval_metrics_convnext_tiny_brain_only_10000.json"

OUT_CSV = REPORT_DIR / "paired_bootstrap_ablation_multiwindow_vs_brainonly.csv"
OUT_JSON = REPORT_DIR / "paired_bootstrap_ablation_multiwindow_vs_brainonly.json"
OUT_FIG = FIG_DIR / "paired_bootstrap_multiwindow_vs_brainonly.png"

N_BOOTSTRAP = 2000
RANDOM_STATE = 2026


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def metric_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def percentile_ci(values: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(values, lower_q)), float(np.percentile(values, upper_q))


def bootstrap_p_value(diff_values: np.ndarray) -> float:
    frac_le_zero = float(np.mean(diff_values <= 0))
    frac_ge_zero = float(np.mean(diff_values >= 0))
    return float(min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero)))


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)

    sampled = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled)

    return sampled


def load_predictions() -> pd.DataFrame:
    if not PRED_MULTI.exists():
        raise FileNotFoundError(f"Missing multi-window predictions: {PRED_MULTI}")

    if not PRED_BRAIN.exists():
        raise FileNotFoundError(f"Missing brain-only predictions: {PRED_BRAIN}")

    multi_df = pd.read_csv(PRED_MULTI)[["image_id", "true_any", "prob_any"]].copy()
    brain_df = pd.read_csv(PRED_BRAIN)[["image_id", "true_any", "prob_any"]].copy()

    multi_df = multi_df.rename(columns={"prob_any": "prob_any_multi_window"})
    brain_df = brain_df.rename(columns={"prob_any": "prob_any_brain_only"})

    merged = multi_df.merge(
        brain_df,
        on=["image_id", "true_any"],
        how="inner",
    )

    return merged


def summarize_thresholds() -> dict:
    multi = load_json(METRICS_MULTI)
    brain = load_json(METRICS_BRAIN)

    return {
        "multi_window": {
            "any_auc": multi["per_label"]["any"]["auc"],
            "any_average_precision": multi["per_label"]["any"]["average_precision"],
            "best_recall90": multi["best_threshold_with_recall_at_least_90"],
            "best_f1": multi["best_f1_threshold_any"],
        },
        "brain_only": {
            "any_auc": brain["per_label"]["any"]["auc"],
            "any_average_precision": brain["per_label"]["any"]["average_precision"],
            "best_recall90": brain["best_threshold_with_recall_at_least_90"],
            "best_f1": brain["best_f1_threshold_any"],
        },
    }


def main() -> None:
    print("===== PAIRED BOOTSTRAP ABLATION: MULTI-WINDOW VS BRAIN-ONLY =====")

    pred_df = load_predictions()

    print("Merged prediction shape:", pred_df.shape)
    print("Columns:", pred_df.columns.tolist())

    y_true = pred_df["true_any"].values.astype(int)
    prob_multi = pred_df["prob_any_multi_window"].values.astype(float)
    prob_brain = pred_df["prob_any_brain_only"].values.astype(float)

    print("Positive count:", int(y_true.sum()))
    print("Negative count:", int(len(y_true) - y_true.sum()))
    print("Prevalence:", float(y_true.mean()))

    point = {
        "multi_window_auc": metric_auc(y_true, prob_multi),
        "brain_only_auc": metric_auc(y_true, prob_brain),
        "multi_window_average_precision": metric_ap(y_true, prob_multi),
        "brain_only_average_precision": metric_ap(y_true, prob_brain),
    }

    point["auc_difference_multi_minus_brain"] = point["multi_window_auc"] - point["brain_only_auc"]
    point["average_precision_difference_multi_minus_brain"] = (
        point["multi_window_average_precision"] - point["brain_only_average_precision"]
    )

    print("\n===== POINT METRICS =====")
    print(point)

    rng = np.random.default_rng(RANDOM_STATE)

    auc_diffs = []
    ap_diffs = []

    for _ in range(N_BOOTSTRAP):
        idx = stratified_bootstrap_indices(y_true, rng)

        y_b = y_true[idx]
        multi_b = prob_multi[idx]
        brain_b = prob_brain[idx]

        auc_diffs.append(metric_auc(y_b, multi_b) - metric_auc(y_b, brain_b))
        ap_diffs.append(metric_ap(y_b, multi_b) - metric_ap(y_b, brain_b))

    auc_diffs = np.array(auc_diffs)
    ap_diffs = np.array(ap_diffs)

    auc_low, auc_high = percentile_ci(auc_diffs)
    ap_low, ap_high = percentile_ci(ap_diffs)

    rows = [
        {
            "comparison": "multi_window_minus_brain_only",
            "metric": "auc",
            "multi_window_value": point["multi_window_auc"],
            "brain_only_value": point["brain_only_auc"],
            "difference": point["auc_difference_multi_minus_brain"],
            "difference_ci_low": auc_low,
            "difference_ci_high": auc_high,
            "bootstrap_p_value": bootstrap_p_value(auc_diffs),
            "n_bootstrap": N_BOOTSTRAP,
        },
        {
            "comparison": "multi_window_minus_brain_only",
            "metric": "average_precision",
            "multi_window_value": point["multi_window_average_precision"],
            "brain_only_value": point["brain_only_average_precision"],
            "difference": point["average_precision_difference_multi_minus_brain"],
            "difference_ci_low": ap_low,
            "difference_ci_high": ap_high,
            "bootstrap_p_value": bootstrap_p_value(ap_diffs),
            "n_bootstrap": N_BOOTSTRAP,
        },
    ]

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)

    print("\n===== BOOTSTRAP DIFFERENCE RESULT =====")
    print(result_df)

    threshold_summary = summarize_thresholds()

    summary = {
        "n_bootstrap": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "holdout_size": int(len(pred_df)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "point_metrics": point,
        "bootstrap_results": result_df.to_dict(orient="records"),
        "threshold_summary": threshold_summary,
        "output_csv": str(OUT_CSV),
        "figure": str(OUT_FIG),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x_labels = ["AUC", "Average Precision"]
    y = result_df["difference"].values
    yerr_low = y - result_df["difference_ci_low"].values
    yerr_high = result_df["difference_ci_high"].values - y

    fig = plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(y)), y, yerr=[yerr_low, yerr_high], capsize=4)
    plt.axhline(0, linestyle="--")
    plt.xticks(np.arange(len(y)), x_labels)
    plt.ylabel("Difference: Multi-window minus Brain-only")
    plt.title("Paired Bootstrap Ablation Comparison")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=250)
    plt.close(fig)

    print("\nSaved CSV:", OUT_CSV)
    print("Saved JSON:", OUT_JSON)
    print("Saved figure:", OUT_FIG)

    print("\nSTATUS: PAIRED BOOTSTRAP ABLATION COMPARISON GENERATED")


if __name__ == "__main__":
    main()
