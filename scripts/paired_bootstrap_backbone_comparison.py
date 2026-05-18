from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


LOG_DIR = Path("outputs/logs")
REPORT_DIR = Path("outputs/reports")
FIG_DIR = Path("outputs/figures/statistical_comparison")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILES = {
    "EfficientNet-B0": LOG_DIR / "holdout_5000_predictions_model10000.csv",
    "DenseNet121": LOG_DIR / "holdout_5000_predictions_densenet121_10000.csv",
    "ConvNeXt-Tiny": LOG_DIR / "holdout_5000_predictions_convnext_tiny_10000.csv",
}

OUT_CSV = REPORT_DIR / "paired_bootstrap_backbone_comparison.csv"
OUT_JSON = REPORT_DIR / "paired_bootstrap_backbone_comparison.json"
OUT_FIG = FIG_DIR / "paired_bootstrap_auc_ap_difference.png"

N_BOOTSTRAP = 2000
RANDOM_STATE = 2026

PRIMARY_MODEL = "ConvNeXt-Tiny"

COMPARISONS = [
    ("ConvNeXt-Tiny", "DenseNet121"),
    ("ConvNeXt-Tiny", "EfficientNet-B0"),
    ("DenseNet121", "EfficientNet-B0"),
]


def metric_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def metric_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def percentile_ci(values: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(values, lower_q)), float(np.percentile(values, upper_q))


def bootstrap_p_value(diff_values: np.ndarray) -> float:
    """
    Two-sided bootstrap sign-style p-value.
    If almost all bootstrap differences are > 0, p becomes small.
    This is not DeLong, but it is a useful paired bootstrap test.
    """
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
    merged = None

    for model_name, path in PRED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {model_name}: {path}")

        df = pd.read_csv(path)

        required = ["image_id", "true_any", "prob_any"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = df[required].copy()
        df = df.rename(columns={"prob_any": f"prob_any_{model_name}"})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[["image_id", "true_any", f"prob_any_{model_name}"]],
                on=["image_id", "true_any"],
                how="inner",
            )

    if merged is None:
        raise RuntimeError("No predictions loaded.")

    return merged


def main() -> None:
    print("===== PAIRED BOOTSTRAP BACKBONE COMPARISON =====")

    pred_df = load_predictions()

    print("Merged prediction shape:", pred_df.shape)
    print("Columns:", pred_df.columns.tolist())

    y_true = pred_df["true_any"].values.astype(int)

    print("Positive count:", int(y_true.sum()))
    print("Negative count:", int(len(y_true) - y_true.sum()))
    print("Prevalence:", float(y_true.mean()))

    rng = np.random.default_rng(RANDOM_STATE)

    point_metrics = {}

    for model_name in PRED_FILES:
        y_prob = pred_df[f"prob_any_{model_name}"].values.astype(float)

        point_metrics[model_name] = {
            "auc": metric_auc(y_true, y_prob),
            "average_precision": metric_ap(y_true, y_prob),
        }

    print("\n===== POINT METRICS =====")
    for model_name, metrics in point_metrics.items():
        print(model_name, metrics)

    rows = []

    for model_a, model_b in COMPARISONS:
        prob_a = pred_df[f"prob_any_{model_a}"].values.astype(float)
        prob_b = pred_df[f"prob_any_{model_b}"].values.astype(float)

        point_auc_diff = point_metrics[model_a]["auc"] - point_metrics[model_b]["auc"]
        point_ap_diff = point_metrics[model_a]["average_precision"] - point_metrics[model_b]["average_precision"]

        auc_diffs = []
        ap_diffs = []

        for _ in range(N_BOOTSTRAP):
            idx = stratified_bootstrap_indices(y_true, rng)

            y_b = y_true[idx]
            prob_a_b = prob_a[idx]
            prob_b_b = prob_b[idx]

            auc_diffs.append(metric_auc(y_b, prob_a_b) - metric_auc(y_b, prob_b_b))
            ap_diffs.append(metric_ap(y_b, prob_a_b) - metric_ap(y_b, prob_b_b))

        auc_diffs = np.array(auc_diffs)
        ap_diffs = np.array(ap_diffs)

        auc_low, auc_high = percentile_ci(auc_diffs)
        ap_low, ap_high = percentile_ci(ap_diffs)

        row_auc = {
            "comparison": f"{model_a} - {model_b}",
            "model_a": model_a,
            "model_b": model_b,
            "metric": "auc",
            "model_a_value": point_metrics[model_a]["auc"],
            "model_b_value": point_metrics[model_b]["auc"],
            "difference": point_auc_diff,
            "difference_ci_low": auc_low,
            "difference_ci_high": auc_high,
            "bootstrap_p_value": bootstrap_p_value(auc_diffs),
            "n_bootstrap": N_BOOTSTRAP,
        }

        row_ap = {
            "comparison": f"{model_a} - {model_b}",
            "model_a": model_a,
            "model_b": model_b,
            "metric": "average_precision",
            "model_a_value": point_metrics[model_a]["average_precision"],
            "model_b_value": point_metrics[model_b]["average_precision"],
            "difference": point_ap_diff,
            "difference_ci_low": ap_low,
            "difference_ci_high": ap_high,
            "bootstrap_p_value": bootstrap_p_value(ap_diffs),
            "n_bootstrap": N_BOOTSTRAP,
        }

        rows.extend([row_auc, row_ap])

        print("\nComparison:", model_a, "vs", model_b)
        print("AUC diff:", row_auc)
        print("AP diff:", row_ap)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)

    summary = {
        "n_bootstrap": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "holdout_size": int(len(pred_df)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "point_metrics": point_metrics,
        "comparisons": result_df.to_dict(orient="records"),
        "output_csv": str(OUT_CSV),
        "figure": str(OUT_FIG),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figure
    plot_df = result_df.copy()
    plot_df["label"] = plot_df["comparison"] + "\n" + plot_df["metric"]

    x = np.arange(len(plot_df))
    y = plot_df["difference"].values
    yerr_low = y - plot_df["difference_ci_low"].values
    yerr_high = plot_df["difference_ci_high"].values - y

    fig = plt.figure(figsize=(12, 6))
    plt.bar(x, y, yerr=[yerr_low, yerr_high], capsize=4)
    plt.axhline(0, linestyle="--")
    plt.xticks(x, plot_df["label"], rotation=35, ha="right")
    plt.ylabel("Paired bootstrap difference")
    plt.title("Paired Bootstrap Comparison of Backbone Performance")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=250)
    plt.close(fig)

    print("\nSaved CSV:", OUT_CSV)
    print("Saved JSON:", OUT_JSON)
    print("Saved figure:", OUT_FIG)

    print("\nSTATUS: PAIRED BOOTSTRAP BACKBONE COMPARISON GENERATED")


if __name__ == "__main__":
    main()
