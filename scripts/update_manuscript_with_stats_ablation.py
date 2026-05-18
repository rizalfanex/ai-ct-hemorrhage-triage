from pathlib import Path
import pandas as pd


PAPER_DIR = Path("paper")
REPORT_DIR = Path("outputs/reports")

MANUSCRIPT_PATH = PAPER_DIR / "manuscript_draft.md"
MANUSCRIPT_V2_PATH = PAPER_DIR / "manuscript_draft_v2.md"
STAT_SUMMARY_PATH = PAPER_DIR / "statistical_and_ablation_summary.md"

BACKBONE_PAIR_CSV = REPORT_DIR / "paired_bootstrap_backbone_comparison.csv"
ABLATION_CSV = REPORT_DIR / "paired_bootstrap_ablation_multiwindow_vs_brainonly.csv"


def main() -> None:
    print("===== UPDATE MANUSCRIPT WITH STATISTICAL AND ABLATION RESULTS =====")

    if not MANUSCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing manuscript: {MANUSCRIPT_PATH}")

    if not BACKBONE_PAIR_CSV.exists():
        raise FileNotFoundError(f"Missing paired backbone CSV: {BACKBONE_PAIR_CSV}")

    if not ABLATION_CSV.exists():
        raise FileNotFoundError(f"Missing ablation CSV: {ABLATION_CSV}")

    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")

    backbone_df = pd.read_csv(BACKBONE_PAIR_CSV)
    ablation_df = pd.read_csv(ABLATION_CSV)

    conv_vs_dense_auc = backbone_df[
        (backbone_df["comparison"] == "ConvNeXt-Tiny - DenseNet121") &
        (backbone_df["metric"] == "auc")
    ].iloc[0]

    conv_vs_dense_ap = backbone_df[
        (backbone_df["comparison"] == "ConvNeXt-Tiny - DenseNet121") &
        (backbone_df["metric"] == "average_precision")
    ].iloc[0]

    conv_vs_eff_auc = backbone_df[
        (backbone_df["comparison"] == "ConvNeXt-Tiny - EfficientNet-B0") &
        (backbone_df["metric"] == "auc")
    ].iloc[0]

    conv_vs_eff_ap = backbone_df[
        (backbone_df["comparison"] == "ConvNeXt-Tiny - EfficientNet-B0") &
        (backbone_df["metric"] == "average_precision")
    ].iloc[0]

    abl_auc = ablation_df[ablation_df["metric"] == "auc"].iloc[0]
    abl_ap = ablation_df[ablation_df["metric"] == "average_precision"].iloc[0]

    statistical_section = f"""
## 3.7 Paired Bootstrap Statistical Comparison

Because all evaluated backbones were tested on the same 5,000-image natural-prevalence holdout set, paired bootstrap analysis was performed to estimate whether observed performance differences were stable across resampled holdout distributions.

ConvNeXt-Tiny outperformed DenseNet121 in any-hemorrhage AUC by {conv_vs_dense_auc["difference"]:.6f} with 95% bootstrap CI {conv_vs_dense_auc["difference_ci_low"]:.6f} to {conv_vs_dense_auc["difference_ci_high"]:.6f} and bootstrap p-value {conv_vs_dense_auc["bootstrap_p_value"]:.4f}. ConvNeXt-Tiny also improved AP over DenseNet121 by {conv_vs_dense_ap["difference"]:.6f} with 95% bootstrap CI {conv_vs_dense_ap["difference_ci_low"]:.6f} to {conv_vs_dense_ap["difference_ci_high"]:.6f}.

Compared with EfficientNet-B0, ConvNeXt-Tiny improved AUC by {conv_vs_eff_auc["difference"]:.6f} with 95% bootstrap CI {conv_vs_eff_auc["difference_ci_low"]:.6f} to {conv_vs_eff_auc["difference_ci_high"]:.6f}. It improved AP by {conv_vs_eff_ap["difference"]:.6f} with 95% bootstrap CI {conv_vs_eff_ap["difference_ci_low"]:.6f} to {conv_vs_eff_ap["difference_ci_high"]:.6f}.

These paired bootstrap results support ConvNeXt-Tiny as the strongest evaluated backbone in this experimental protocol.

## 3.8 CT Window Ablation Study

To test whether multi-window CT input contributed measurable benefit, ConvNeXt-Tiny was additionally trained using only the brain window replicated into three channels. The brain-only model achieved any-hemorrhage AUC {abl_auc["brain_only_value"]:.6f}, compared with {abl_auc["multi_window_value"]:.6f} for the multi-window model. The paired AUC difference for multi-window minus brain-only was {abl_auc["difference"]:.6f}, with bootstrap p-value {abl_auc["bootstrap_p_value"]:.4f}.

Average precision was {abl_ap["multi_window_value"]:.6f} for the multi-window model and {abl_ap["brain_only_value"]:.6f} for the brain-only model. The paired AP difference was {abl_ap["difference"]:.6f}, with bootstrap p-value {abl_ap["bootstrap_p_value"]:.4f}.

These results do not provide evidence that multi-window input significantly improves slice-level any-hemorrhage triage over brain-window-only input in this experimental setup. This suggests that most discriminative signal for the any-hemorrhage task may already be captured by the brain window, while subtype-specific or patient-level tasks may still benefit from additional window settings.
"""

    # Insert before Discussion if not already inserted.
    marker = "# 4. Discussion"
    if marker not in manuscript:
        raise ValueError("Could not find Discussion marker in manuscript.")

    if "## 3.7 Paired Bootstrap Statistical Comparison" not in manuscript:
        manuscript_v2 = manuscript.replace(marker, statistical_section + "\n" + marker)
    else:
        manuscript_v2 = manuscript

    # Strengthen discussion wording.
    manuscript_v2 = manuscript_v2.replace(
        "ConvNeXt-Tiny outperformed EfficientNet-B0 and DenseNet121 in AUC, average precision, and high-sensitivity workload trade-off.",
        "ConvNeXt-Tiny outperformed EfficientNet-B0 and DenseNet121 in AUC, average precision, and high-sensitivity workload trade-off. Paired bootstrap analysis indicated that these improvements were stable over resampled holdout distributions."
    )

    manuscript_v2 = manuscript_v2.replace(
        "Calibration remains important.",
        "The CT-window ablation showed that brain-window-only input performed comparably to the multi-window representation. Therefore, the benefit of multi-window preprocessing should not be overstated for the slice-level any-hemorrhage endpoint. Calibration remains important."
    )

    MANUSCRIPT_V2_PATH.write_text(manuscript_v2, encoding="utf-8")

    stat_summary = f"""# Statistical and Ablation Summary

## Paired Backbone Comparison

{backbone_df.to_markdown(index=False)}

## CT Window Ablation

{ablation_df.to_markdown(index=False)}

## Main Interpretation

ConvNeXt-Tiny significantly outperformed DenseNet121 and EfficientNet-B0 under paired bootstrap comparison on the same natural-prevalence holdout set.

The CT window ablation did not show a statistically meaningful difference between multi-window and brain-window-only input. Brain-only was slightly higher in AUC, while multi-window was slightly higher in AP, but both differences were small and not statistically robust.

## Manuscript Implication

The paper should claim that ConvNeXt-Tiny is the strongest evaluated backbone. However, it should not claim that multi-window preprocessing clearly outperforms brain-only preprocessing for the any-hemorrhage endpoint.
"""

    STAT_SUMMARY_PATH.write_text(stat_summary, encoding="utf-8")

    print("Saved updated manuscript:", MANUSCRIPT_V2_PATH)
    print("Saved statistical summary:", STAT_SUMMARY_PATH)

    print("\nSTATUS: MANUSCRIPT UPDATED WITH STATISTICAL AND ABLATION RESULTS")


if __name__ == "__main__":
    main()
