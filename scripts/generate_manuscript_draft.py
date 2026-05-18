from pathlib import Path
import json
import pandas as pd


PAPER_DIR = Path("paper")
REPORT_DIR = Path("outputs/reports")
LOG_DIR = Path("outputs/logs")

PAPER_DIR.mkdir(parents=True, exist_ok=True)

MANUSCRIPT_PATH = PAPER_DIR / "manuscript_draft.md"
FIGURE_PLAN_PATH = PAPER_DIR / "figure_plan.md"
CHECKLIST_PATH = PAPER_DIR / "q1_submission_gap_checklist.md"

BEST_MODEL_METRICS = LOG_DIR / "holdout_5000_eval_metrics_convnext_tiny_10000.json"
BEST_MODEL_TRAIN = LOG_DIR / "convnext_tiny_rsna_10000_summary.json"
BACKBONE_COMPARISON = REPORT_DIR / "backbone_comparison_holdout_5000.csv"
PER_LABEL_COMPARISON = REPORT_DIR / "backbone_comparison_per_label_auc_ap.csv"
BOOTSTRAP_CI = REPORT_DIR / "bootstrap_ci_convnext_tiny_10000.csv"
BOOTSTRAP_THRESHOLD_CI = REPORT_DIR / "bootstrap_threshold_ci_any_convnext_tiny_10000.csv"
PAPER_METRICS = REPORT_DIR / "paper_metrics_summary_convnext_tiny_10000.json"
ERROR_ANALYSIS = REPORT_DIR / "error_analysis_summary_model10000_threshold025.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def get_ci_row(ci_df: pd.DataFrame, label: str) -> pd.Series:
    rows = ci_df[ci_df["label"] == label]
    if len(rows) != 1:
        raise ValueError(f"Could not find CI row for label={label}")
    return rows.iloc[0]


def main() -> None:
    print("===== GENERATE Q1-STYLE MANUSCRIPT DRAFT =====")

    best_metrics = load_json(BEST_MODEL_METRICS)
    best_train = load_json(BEST_MODEL_TRAIN)
    paper_metrics = load_json(PAPER_METRICS)

    backbone_df = read_csv(BACKBONE_COMPARISON)
    per_label_df = read_csv(PER_LABEL_COMPARISON)
    ci_df = read_csv(BOOTSTRAP_CI)
    threshold_ci_df = read_csv(BOOTSTRAP_THRESHOLD_CI)

    any_metrics = best_metrics["per_label"]["any"]
    any_ci = get_ci_row(ci_df, "any")

    triage = best_metrics["best_threshold_with_recall_at_least_90"]
    best_f1 = best_metrics["best_f1_threshold_any"]

    triage_ci = threshold_ci_df[threshold_ci_df["threshold_name"] == "triage_recall90"].iloc[0]
    best_f1_ci = threshold_ci_df[threshold_ci_df["threshold_name"] == "best_f1"].iloc[0]

    brier = paper_metrics["calibration"]["brier_score"]

    convnext_row = backbone_df[backbone_df["model"] == "ConvNeXt-Tiny"].iloc[0]
    densenet_row = backbone_df[backbone_df["model"] == "DenseNet121"].iloc[0]
    efficientnet_row = backbone_df[backbone_df["model"] == "EfficientNet-B0"].iloc[0]

    best_per_label = per_label_df[per_label_df["model"] == "ConvNeXt-Tiny"].copy()

    manuscript = f"""# Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT Images: A Natural-Prevalence Holdout Study

## Abstract

### Background

Intracranial hemorrhage (ICH) is a time-sensitive radiological finding where delayed interpretation of head computed tomography (CT) may affect clinical workflow. Deep learning systems may assist radiology triage by prioritizing scans with high predicted probability of hemorrhage, while leaving final interpretation to radiologists.

### Objective

This study evaluates whether convolutional neural network backbones can prioritize head CT images for ICH triage using multi-window DICOM preprocessing, natural-prevalence holdout evaluation, threshold analysis, calibration analysis, bootstrap confidence intervals, and error analysis.

### Methods

We used the RSNA Intracranial Hemorrhage Detection dataset and converted DICOM images into a three-channel CT representation using brain, subdural, and bone windows. Three CNN backbones were compared: EfficientNet-B0, DenseNet121, and ConvNeXt-Tiny. Each model was trained on 10,000 balanced CT images consisting of 5,000 normal and 5,000 hemorrhage-positive samples. Generalization was evaluated on a non-overlapping natural-prevalence holdout set of 5,000 CT images containing 701 hemorrhage-positive images and 4,299 normal images. Primary outcomes were any-hemorrhage area under the ROC curve (AUC) and average precision (AP). Secondary analyses included threshold sweep, triage simulation, calibration, bootstrap 95% confidence intervals, and error analysis.

### Results

ConvNeXt-Tiny achieved the best natural-holdout performance among the evaluated backbones, with any-hemorrhage AUC {any_metrics["auc"]:.4f} (95% CI {any_ci["auc_ci_low"]:.4f}–{any_ci["auc_ci_high"]:.4f}) and AP {any_metrics["average_precision"]:.4f} (95% CI {any_ci["average_precision_ci_low"]:.4f}–{any_ci["average_precision_ci_high"]:.4f}). EfficientNet-B0 achieved AUC {efficientnet_row["holdout_any_auc"]:.4f} and AP {efficientnet_row["holdout_any_average_precision"]:.4f}, while DenseNet121 achieved AUC {densenet_row["holdout_any_auc"]:.4f} and AP {densenet_row["holdout_any_average_precision"]:.4f}. At a high-sensitivity triage threshold of {triage["threshold"]:.2f}, ConvNeXt-Tiny reached recall {triage["recall"]:.4f} (95% CI {triage_ci["recall_ci_low"]:.4f}–{triage_ci["recall_ci_high"]:.4f}), precision {triage["precision"]:.4f} (95% CI {triage_ci["precision_ci_low"]:.4f}–{triage_ci["precision_ci_high"]:.4f}), and prioritized {triage["predicted_positive_rate"] * 100:.2f}% of the workload. Calibration analysis yielded a Brier score of {brier:.4f}.

### Conclusion

A ConvNeXt-Tiny model trained with multi-window CT inputs achieved strong slice-level ICH triage performance on a natural-prevalence holdout set. The system showed potential for prioritizing likely hemorrhage-positive CT images, but external validation, patient-level aggregation, calibration refinement, and prospective clinical workflow evaluation are required before clinical use.

## Keywords

Intracranial hemorrhage; computed tomography; radiology triage; deep learning; ConvNeXt; medical image analysis; explainable AI.

---

# 1. Introduction

Intracranial hemorrhage is an urgent radiological finding requiring timely review of head CT examinations. In busy radiology workflows, an automated triage system may help prioritize likely abnormal studies for earlier radiologist review. The goal of such a system is not to replace radiologists, but to improve queue prioritization and reduce time-to-review for potentially critical findings.

Public CT datasets have enabled reproducible research in ICH detection. The RSNA Intracranial Hemorrhage Detection dataset provides annotated cranial CT data for detecting acute hemorrhage and hemorrhage subtypes. However, high model performance on balanced validation splits may overestimate real-world triage utility because clinical prevalence is typically imbalanced. Therefore, natural-prevalence evaluation, threshold trade-off analysis, calibration, and error analysis are needed to judge clinical workflow relevance.

This study builds a reproducible research prototype for slice-level ICH triage. The contribution is fourfold:

1. A complete DICOM-to-model pipeline using Hounsfield Unit conversion and multi-window CT preprocessing.
2. A fair comparison of EfficientNet-B0, DenseNet121, and ConvNeXt-Tiny on the same training and holdout protocol.
3. Natural-prevalence holdout evaluation with ROC, PR, calibration, bootstrap confidence intervals, and triage workload simulation.
4. Error analysis and explainability-oriented outputs to support discussion of failure modes and clinical limitations.

# 2. Materials and Methods

## 2.1 Dataset

The RSNA Intracranial Hemorrhage Detection dataset was used. Labels were converted from the original long format into a wide multi-label table with six binary targets:

- any
- epidural
- intraparenchymal
- intraventricular
- subarachnoid
- subdural

After preprocessing labels, 752,803 unique training images were available. Of these, 107,933 were positive for any hemorrhage. The natural any-hemorrhage prevalence was approximately 14.34%.

Experimental subsets were constructed as follows:

| Subset | Size | Distribution |
|---|---:|---|
| Balanced training set | 10,000 | 5,000 normal / 5,000 any-hemorrhage positive |
| Natural holdout set | 5,000 | 4,299 normal / 701 any-hemorrhage positive |

The holdout set was sampled without overlap with the training subset.

## 2.2 DICOM Preprocessing

DICOM pixel arrays were converted to Hounsfield Units using rescale slope and intercept metadata. A three-channel CT image representation was generated using:

| Channel | Window | Center | Width |
|---|---|---:|---:|
| 1 | Brain | 40 | 80 |
| 2 | Subdural | 80 | 200 |
| 3 | Bone | 600 | 2800 |

Each windowed image was clipped, normalized to the range 0–1, stacked into a three-channel tensor, and resized to 224 × 224.

## 2.3 Models

Three CNN backbones were evaluated:

| Backbone | Rationale |
|---|---|
| EfficientNet-B0 | Lightweight compound-scaled CNN baseline |
| DenseNet121 | Strong classic medical-imaging CNN comparator |
| ConvNeXt-Tiny | Modern CNN backbone with transformer-inspired design choices |

Each model used a six-output multi-label classification head.

## 2.4 Training Protocol

Models were trained on the same 10,000-image balanced subset. The loss function was binary cross entropy with logits and positive class weighting. AdamW optimization was used with learning rate 1e-4 and weight decay 1e-4. Mixed precision training was enabled on CUDA.

Training configuration:

| Parameter | Value |
|---|---:|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Primary validation metric | any-hemorrhage AUC |

## 2.5 Evaluation

Evaluation was performed on the 5,000-image natural-prevalence holdout set. Primary metrics were:

- ROC AUC for any hemorrhage
- Average precision for any hemorrhage

Secondary metrics included:

- Per-label AUC and AP
- Precision, recall, and F1 across thresholds
- Confusion matrices at selected thresholds
- Triage recall at top workload percentages
- Calibration curve and Brier score
- Bootstrap 95% confidence intervals
- Error analysis by false-positive and false-negative groups

# 3. Results

## 3.1 Backbone Comparison

{backbone_df.to_markdown(index=False)}

ConvNeXt-Tiny achieved the strongest performance among the evaluated backbones, with any-hemorrhage AUC {convnext_row["holdout_any_auc"]:.4f} and AP {convnext_row["holdout_any_average_precision"]:.4f}. It also required the lowest prioritized workload to achieve recall above 90%.

## 3.2 Best Model Performance

ConvNeXt-Tiny achieved:

| Metric | Value |
|---|---:|
| Any AUC | {any_metrics["auc"]:.6f} |
| Any AUC 95% CI low | {any_ci["auc_ci_low"]:.6f} |
| Any AUC 95% CI high | {any_ci["auc_ci_high"]:.6f} |
| Any AP | {any_metrics["average_precision"]:.6f} |
| Any AP 95% CI low | {any_ci["average_precision_ci_low"]:.6f} |
| Any AP 95% CI high | {any_ci["average_precision_ci_high"]:.6f} |
| Brier score | {brier:.6f} |

## 3.3 Per-Label Performance of ConvNeXt-Tiny

{best_per_label.to_markdown(index=False)}

## 3.4 High-Sensitivity Triage Threshold

At threshold {triage["threshold"]:.2f}, ConvNeXt-Tiny achieved:

| Metric | Value |
|---|---:|
| TP | {int(triage["tp"])} |
| FP | {int(triage["fp"])} |
| FN | {int(triage["fn"])} |
| TN | {int(triage["tn"])} |
| Recall | {triage["recall"]:.6f} |
| Recall 95% CI low | {triage_ci["recall_ci_low"]:.6f} |
| Recall 95% CI high | {triage_ci["recall_ci_high"]:.6f} |
| Precision | {triage["precision"]:.6f} |
| Precision 95% CI low | {triage_ci["precision_ci_low"]:.6f} |
| Precision 95% CI high | {triage_ci["precision_ci_high"]:.6f} |
| False negative rate | {triage["false_negative_rate"]:.6f} |
| Prioritized workload rate | {triage["predicted_positive_rate"]:.6f} |

This operating point captured 653 of 701 hemorrhage-positive images while prioritizing 32.44% of the workload.

## 3.5 Best-F1 Threshold

At threshold {best_f1["threshold"]:.2f}, ConvNeXt-Tiny achieved F1 {best_f1["f1"]:.6f}, precision {best_f1["precision"]:.6f}, and recall {best_f1["recall"]:.6f}. This threshold is less appropriate for high-sensitivity emergency triage because recall is lower than the selected triage threshold.

## 3.6 Figures

The following figures are generated by the repository:

| Figure | Path |
|---|---|
| Backbone AUC/AP comparison | outputs/figures/model_comparison/backbone_any_auc_ap_comparison.png |
| Backbone triage recall curve | outputs/figures/model_comparison/backbone_triage_recall_curve.png |
| Per-label AUC comparison | outputs/figures/model_comparison/backbone_per_label_auc_comparison.png |
| ROC curve | outputs/figures/paper_metrics_convnext_tiny/roc_curve_any_convnext_tiny_10000.png |
| PR curve | outputs/figures/paper_metrics_convnext_tiny/pr_curve_any_convnext_tiny_10000.png |
| Calibration curve | outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png |
| ConvNeXt bootstrap CI | outputs/figures/paper_metrics/bootstrap_ci_auc_ap_convnext_tiny_10000.png |
| Triage confusion matrix | outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_triage_recall90_threshold_010.png |

# 4. Discussion

This study demonstrates that a modern CNN backbone trained with multi-window CT preprocessing can achieve strong slice-level ICH triage performance on a natural-prevalence holdout set. ConvNeXt-Tiny outperformed EfficientNet-B0 and DenseNet121 in AUC, average precision, and high-sensitivity workload trade-off.

The natural-prevalence holdout evaluation is important because balanced validation sets do not reflect deployment-like workload. In triage settings, recall is often prioritized because false negatives may delay review of urgent cases. The selected threshold of 0.10 achieved recall above 93% while prioritizing approximately one-third of the workload. This is not a replacement for radiologist interpretation, but it may support queue prioritization.

Calibration remains important. The Brier score of {brier:.4f} suggests that probability quality should be further analyzed and potentially improved using post-hoc calibration methods such as temperature scaling or isotonic regression. Additionally, subtype AP varied substantially, particularly for epidural hemorrhage because the holdout contained only 17 positive epidural cases.

# 5. Limitations

This study has several limitations:

1. The model is evaluated at slice level, not full patient-study level.
2. The holdout set is internal to the RSNA dataset, not an external hospital dataset.
3. Clinical workflow impact was simulated, not prospectively measured.
4. Model calibration has not yet been optimized.
5. Rare subtype evaluation is limited by small positive counts.
6. Grad-CAM outputs are qualitative and should not be overinterpreted as causal explanations.
7. The model is not clinically validated and is not intended for diagnosis.

# 6. Conclusion

ConvNeXt-Tiny achieved the strongest performance among three evaluated CNN backbones for slice-level ICH triage, with natural-holdout any-hemorrhage AUC {any_metrics["auc"]:.4f} and AP {any_metrics["average_precision"]:.4f}. At a high-sensitivity threshold, the model captured 93.15% of hemorrhage-positive images while prioritizing 32.44% of the workload. These results support further investigation of ConvNeXt-based CT triage systems with external validation, patient-level aggregation, calibration, and prospective workflow evaluation.

# References

1. RSNA Intracranial Hemorrhage Detection Challenge. Radiological Society of North America / Kaggle, 2019.
2. RSNA Intracranial Hemorrhage Detection Open Data Registry. AWS Open Data Registry.
3. Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
4. Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely Connected Convolutional Networks. CVPR, 2017.
5. Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. CVPR, 2022.
"""

    figure_plan = """# Figure Plan

## Main Figures

1. **Pipeline figure**
   - DICOM input
   - HU conversion
   - Multi-window preprocessing
   - CNN backbone
   - Multi-label outputs
   - Triage ranking

2. **Backbone comparison**
   - `outputs/figures/model_comparison/backbone_any_auc_ap_comparison.png`

3. **Triage recall curve**
   - `outputs/figures/model_comparison/backbone_triage_recall_curve.png`

4. **ROC and PR curves for best model**
   - `outputs/figures/paper_metrics_convnext_tiny/roc_curve_any_convnext_tiny_10000.png`
   - `outputs/figures/paper_metrics_convnext_tiny/pr_curve_any_convnext_tiny_10000.png`

5. **Calibration curve**
   - `outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png`

6. **Bootstrap confidence interval**
   - `outputs/figures/paper_metrics/bootstrap_ci_auc_ap_convnext_tiny_10000.png`

7. **Confusion matrix at triage threshold**
   - `outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_triage_recall90_threshold_010.png`

## Supplementary Figures

1. Per-label AUC/AP
2. Error analysis probability distribution
3. False negative rate by subtype
4. Grad-CAM qualitative examples, only if publication permissions are confirmed
"""

    checklist = """# Q1 Submission Gap Checklist

## Already Completed

- [x] DICOM preprocessing pipeline
- [x] Multi-window CT input
- [x] Balanced training subset
- [x] Natural-prevalence holdout
- [x] EfficientNet-B0 baseline
- [x] DenseNet121 baseline
- [x] ConvNeXt-Tiny baseline
- [x] Backbone comparison report
- [x] ROC / PR / calibration figures for best model
- [x] Bootstrap confidence intervals
- [x] Triage threshold analysis
- [x] Error analysis
- [x] GitHub repository without medical data/checkpoints

## Still Needed Before Serious Submission

- [ ] External validation dataset, ideally from a different institution
- [ ] Patient-level or study-level aggregation
- [ ] Calibration improvement: temperature scaling or isotonic regression
- [ ] Statistical comparison between model AUCs, e.g. DeLong or bootstrap paired difference
- [ ] More rigorous ablation study:
  - single-window vs multi-window
  - pretrained vs non-pretrained
  - balanced vs natural training
- [ ] Full manuscript language polishing
- [ ] Journal-specific formatting
- [ ] Ethical/data usage statement
- [ ] Data availability statement
- [ ] Code availability statement
- [ ] Clinical expert review of qualitative examples
"""

    MANUSCRIPT_PATH.write_text(manuscript, encoding="utf-8")
    FIGURE_PLAN_PATH.write_text(figure_plan, encoding="utf-8")
    CHECKLIST_PATH.write_text(checklist, encoding="utf-8")

    print("Saved manuscript:", MANUSCRIPT_PATH)
    print("Saved figure plan:", FIGURE_PLAN_PATH)
    print("Saved Q1 checklist:", CHECKLIST_PATH)

    print("\nSTATUS: MANUSCRIPT DRAFT GENERATED")


if __name__ == "__main__":
    main()
