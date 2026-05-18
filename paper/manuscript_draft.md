# Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT Images: A Natural-Prevalence Holdout Study

## Abstract

### Background

Intracranial hemorrhage (ICH) is a time-sensitive radiological finding where delayed interpretation of head computed tomography (CT) may affect clinical workflow. Deep learning systems may assist radiology triage by prioritizing scans with high predicted probability of hemorrhage, while leaving final interpretation to radiologists.

### Objective

This study evaluates whether convolutional neural network backbones can prioritize head CT images for ICH triage using multi-window DICOM preprocessing, natural-prevalence holdout evaluation, threshold analysis, calibration analysis, bootstrap confidence intervals, and error analysis.

### Methods

We used the RSNA Intracranial Hemorrhage Detection dataset and converted DICOM images into a three-channel CT representation using brain, subdural, and bone windows. Three CNN backbones were compared: EfficientNet-B0, DenseNet121, and ConvNeXt-Tiny. Each model was trained on 10,000 balanced CT images consisting of 5,000 normal and 5,000 hemorrhage-positive samples. Generalization was evaluated on a non-overlapping natural-prevalence holdout set of 5,000 CT images containing 701 hemorrhage-positive images and 4,299 normal images. Primary outcomes were any-hemorrhage area under the ROC curve (AUC) and average precision (AP). Secondary analyses included threshold sweep, triage simulation, calibration, bootstrap 95% confidence intervals, and error analysis.

### Results

ConvNeXt-Tiny achieved the best natural-holdout performance among the evaluated backbones, with any-hemorrhage AUC 0.9458 (95% CI 0.9370–0.9538) and AP 0.8160 (95% CI 0.7922–0.8397). EfficientNet-B0 achieved AUC 0.9164 and AP 0.7097, while DenseNet121 achieved AUC 0.9365 and AP 0.7816. At a high-sensitivity triage threshold of 0.10, ConvNeXt-Tiny reached recall 0.9315 (95% CI 0.9130–0.9501), precision 0.4026 (95% CI 0.3894–0.4179), and prioritized 32.44% of the workload. Calibration analysis yielded a Brier score of 0.0640.

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

| model           |   train_size |   holdout_size |   holdout_prevalence_any |    parameters |   training_time_seconds |   validation_best_any_auc |   holdout_any_auc |   holdout_any_average_precision |   best_f1_threshold |   best_f1 |   recall90_threshold |   recall90_precision |   recall90_recall |   recall90_false_negative_rate |   recall90_predicted_positive_rate |
|:----------------|-------------:|---------------:|-------------------------:|--------------:|------------------------:|--------------------------:|------------------:|--------------------------------:|--------------------:|----------:|---------------------:|---------------------:|------------------:|-------------------------------:|-----------------------------------:|
| EfficientNet-B0 |        10000 |           5000 |                   0.1402 | nan           |                 1144.32 |                  0.904689 |          0.916446 |                        0.709731 |                0.9  |  0.64409  |                 0.25 |             0.359076 |          0.908702 |                      0.0912981 |                             0.3548 |
| DenseNet121     |        10000 |           5000 |                   0.1402 |   6.96001e+06 |                 1163.08 |                  0.925171 |          0.936476 |                        0.781555 |                0.85 |  0.708696 |                 0.35 |             0.373974 |          0.910128 |                      0.0898716 |                             0.3412 |
| ConvNeXt-Tiny   |        10000 |           5000 |                   0.1402 |   2.78247e+07 |                 1145.25 |                  0.930487 |          0.945789 |                        0.816036 |                0.65 |  0.742898 |                 0.1  |             0.402589 |          0.931526 |                      0.0684736 |                             0.3244 |

ConvNeXt-Tiny achieved the strongest performance among the evaluated backbones, with any-hemorrhage AUC 0.9458 and AP 0.8160. It also required the lowest prioritized workload to achieve recall above 90%.

## 3.2 Best Model Performance

ConvNeXt-Tiny achieved:

| Metric | Value |
|---|---:|
| Any AUC | 0.945789 |
| Any AUC 95% CI low | 0.936999 |
| Any AUC 95% CI high | 0.953849 |
| Any AP | 0.816036 |
| Any AP 95% CI low | 0.792165 |
| Any AP 95% CI high | 0.839746 |
| Brier score | 0.063986 |

## 3.3 Per-Label Performance of ConvNeXt-Tiny

| model         | label            |   positive_count |   prevalence |      auc |   average_precision |
|:--------------|:-----------------|-----------------:|-------------:|---------:|--------------------:|
| ConvNeXt-Tiny | any              |              701 |       0.1402 | 0.945789 |           0.816036  |
| ConvNeXt-Tiny | epidural         |               17 |       0.0034 | 0.938839 |           0.0562137 |
| ConvNeXt-Tiny | intraparenchymal |              254 |       0.0508 | 0.970937 |           0.770624  |
| ConvNeXt-Tiny | intraventricular |              160 |       0.032  | 0.990244 |           0.847204  |
| ConvNeXt-Tiny | subarachnoid     |              234 |       0.0468 | 0.92674  |           0.498531  |
| ConvNeXt-Tiny | subdural         |              301 |       0.0602 | 0.914619 |           0.597648  |

## 3.4 High-Sensitivity Triage Threshold

At threshold 0.10, ConvNeXt-Tiny achieved:

| Metric | Value |
|---|---:|
| TP | 653 |
| FP | 969 |
| FN | 48 |
| TN | 3330 |
| Recall | 0.931526 |
| Recall 95% CI low | 0.912981 |
| Recall 95% CI high | 0.950071 |
| Precision | 0.402589 |
| Precision 95% CI low | 0.389417 |
| Precision 95% CI high | 0.417873 |
| False negative rate | 0.068474 |
| Prioritized workload rate | 0.324400 |

This operating point captured 653 of 701 hemorrhage-positive images while prioritizing 32.44% of the workload.

## 3.5 Best-F1 Threshold

At threshold 0.65, ConvNeXt-Tiny achieved F1 0.742898, precision 0.739745, and recall 0.746077. This threshold is less appropriate for high-sensitivity emergency triage because recall is lower than the selected triage threshold.

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

Calibration remains important. The Brier score of 0.0640 suggests that probability quality should be further analyzed and potentially improved using post-hoc calibration methods such as temperature scaling or isotonic regression. Additionally, subtype AP varied substantially, particularly for epidural hemorrhage because the holdout contained only 17 positive epidural cases.

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

ConvNeXt-Tiny achieved the strongest performance among three evaluated CNN backbones for slice-level ICH triage, with natural-holdout any-hemorrhage AUC 0.9458 and AP 0.8160. At a high-sensitivity threshold, the model captured 93.15% of hemorrhage-positive images while prioritizing 32.44% of the workload. These results support further investigation of ConvNeXt-based CT triage systems with external validation, patient-level aggregation, calibration, and prospective workflow evaluation.

# References

1. RSNA Intracranial Hemorrhage Detection Challenge. Radiological Society of North America / Kaggle, 2019.
2. RSNA Intracranial Hemorrhage Detection Open Data Registry. AWS Open Data Registry.
3. Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
4. Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely Connected Convolutional Networks. CVPR, 2017.
5. Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. CVPR, 2022.
