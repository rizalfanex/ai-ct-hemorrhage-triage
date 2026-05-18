# Explainable ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT
# Natural-Prevalence Evaluation, Paired Statistical Backbone Comparison, CT Window Ablation, Calibration, and Explainability

> **Q1-oriented research repository and paper companion for AI-assisted intracranial hemorrhage triage from non-contrast head CT images.**  
> Repository: <https://github.com/rizalfanex/ai-ct-hemorrhage-triage>

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
  - [1.1 Final Research Position](#11-final-research-position)
  - [1.2 Main Result](#12-main-result)
  - [1.3 Why This Repository Looks Q1-Oriented](#13-why-this-repository-looks-q1-oriented)
- [2. Recommended Q1 Paper Title](#2-recommended-q1-paper-title)
  - [2.1 Primary Title](#21-primary-title)
  - [2.2 Alternative Titles](#22-alternative-titles)
- [3. Research Motivation](#3-research-motivation)
  - [3.1 Clinical Problem](#31-clinical-problem)
  - [3.2 AI Triage Opportunity](#32-ai-triage-opportunity)
  - [3.3 Research Question](#33-research-question)
- [4. Completed Work Summary](#4-completed-work-summary)
  - [4.1 Engineering Work Completed](#41-engineering-work-completed)
  - [4.2 Experimental Work Completed](#42-experimental-work-completed)
  - [4.3 Paper-Readiness Work Completed](#43-paper-readiness-work-completed)
- [5. Literature Positioning](#5-literature-positioning)
  - [5.1 RSNA ICH Challenge Context](#51-rsna-ich-challenge-context)
  - [5.2 Relevant ICH Detection Papers](#52-relevant-ich-detection-papers)
  - [5.3 Backbone Literature](#53-backbone-literature)
  - [5.4 Explainability Literature](#54-explainability-literature)
- [6. Dataset and Experimental Splits](#6-dataset-and-experimental-splits)
  - [6.1 Dataset Source](#61-dataset-source)
  - [6.2 Label Schema](#62-label-schema)
  - [6.3 Training and Holdout Splits](#63-training-and-holdout-splits)
- [7. Methodology](#7-methodology)
  - [7.1 Pipeline Overview](#71-pipeline-overview)
  - [7.2 DICOM to Hounsfield Unit Conversion](#72-dicom-to-hounsfield-unit-conversion)
  - [7.3 CT Windowing](#73-ct-windowing)
  - [7.4 Multi-Label Prediction](#74-multi-label-prediction)
  - [7.5 Loss Function](#75-loss-function)
  - [7.6 Triage Thresholding](#76-triage-thresholding)
- [8. Model Backbones](#8-model-backbones)
  - [8.1 EfficientNet-B0](#81-efficientnet-b0)
  - [8.2 DenseNet121](#82-densenet121)
  - [8.3 ConvNeXt-Tiny](#83-convnext-tiny)
- [9. Training Protocol](#9-training-protocol)
  - [9.1 Shared Training Setup](#91-shared-training-setup)
  - [9.2 Hardware](#92-hardware)
- [10. Evaluation Protocol](#10-evaluation-protocol)
  - [10.1 Primary Metrics](#101-primary-metrics)
  - [10.2 Secondary Metrics](#102-secondary-metrics)
  - [10.3 Statistical Testing](#103-statistical-testing)
- [11. Quantitative Results](#11-quantitative-results)
  - [11.1 Backbone Comparison](#111-backbone-comparison)
  - [11.2 Best Model: ConvNeXt-Tiny](#112-best-model-convnext-tiny)
  - [11.3 Per-Label Results](#113-per-label-results)
- [12. Paired Statistical Backbone Comparison](#12-paired-statistical-backbone-comparison)
- [13. CT Window Ablation Study](#13-ct-window-ablation-study)
- [14. Triage Simulation](#14-triage-simulation)
- [15. Calibration Analysis](#15-calibration-analysis)
- [16. Error Analysis](#16-error-analysis)
- [17. Qualitative Explainability](#17-qualitative-explainability)
- [18. Embedded Figures for Q1-Style Review](#18-embedded-figures-for-q1-style-review)
  - [18.1 Backbone Comparison Figures](#181-backbone-comparison-figures)
  - [18.2 ConvNeXt-Tiny Main Paper Figures](#182-convnext-tiny-main-paper-figures)
  - [18.3 Bootstrap and Statistical Figures](#183-bootstrap-and-statistical-figures)
  - [18.4 Ablation Study Figure](#184-ablation-study-figure)
  - [18.5 Error Analysis Figures](#185-error-analysis-figures)
  - [18.6 Supplementary EfficientNet-B0 Baseline Figures](#186-supplementary-efficientnet-b0-baseline-figures)
  - [18.7 Experiment Summary Figures](#187-experiment-summary-figures)
- [19. Generated Reports](#19-generated-reports)
- [20. Repository Structure](#20-repository-structure)
- [21. Reproducibility Workflow](#21-reproducibility-workflow)
- [22. GitHub Data Governance](#22-github-data-governance)
- [23. Q1-Readiness Assessment](#23-q1-readiness-assessment)
- [24. Remaining Gaps Before Strong Q1 Submission](#24-remaining-gaps-before-strong-q1-submission)
- [25. Limitations](#25-limitations)
- [26. Future Work](#26-future-work)
- [27. Medical and Ethical Disclaimer](#27-medical-and-ethical-disclaimer)
- [28. Suggested Citation](#28-suggested-citation)
- [29. References](#29-references)

---

## 1. Executive Summary

### 1.1 Final Research Position

This repository presents a **research-grade deep learning framework** for slice-level intracranial hemorrhage triage on non-contrast head CT images.

The project is designed to support a Q1-style manuscript because it does not stop at basic training accuracy. It includes:

1. DICOM preprocessing;
2. Hounsfield Unit conversion;
3. CT windowing;
4. multi-label hemorrhage prediction;
5. natural-prevalence holdout evaluation;
6. three-backbone comparison;
7. bootstrap confidence intervals;
8. paired bootstrap statistical testing;
9. CT window ablation;
10. calibration analysis;
11. triage threshold analysis;
12. error analysis;
13. explainability-oriented Grad-CAM generation;
14. manuscript draft and figure plan;
15. GitHub-safe public release without raw medical data.

### 1.2 Main Result

The best-performing backbone is **ConvNeXt-Tiny**.

| Metric | ConvNeXt-Tiny Result |
|---|---:|
| Natural holdout size | 5,000 CT images |
| Any-hemorrhage positive cases | 701 |
| Any-hemorrhage prevalence | 14.02% |
| Any AUC | **0.945789** |
| AUC 95% bootstrap CI | **0.936999 – 0.953849** |
| Any Average Precision | **0.816036** |
| AP 95% bootstrap CI | **0.792165 – 0.839746** |
| Brier score | **0.063986** |
| High-sensitivity threshold | **0.10** |
| Recall at threshold 0.10 | **0.931526** |
| Precision at threshold 0.10 | **0.402589** |
| Prioritized workload | **32.44%** |
| False negative rate | **6.85%** |

### 1.3 Why This Repository Looks Q1-Oriented

A Q1-oriented medical AI paper is expected to include more than model accuracy. This project includes:

| Q1 Expectation | Status |
|---|---|
| Natural-prevalence evaluation | Completed |
| Multiple backbone comparison | Completed |
| Confidence intervals | Completed |
| Statistical comparison | Completed |
| Ablation study | Completed |
| Calibration analysis | Completed |
| Error analysis | Completed |
| Explainability component | Generated locally |
| Reproducible code | Completed |
| Data governance | Completed |
| Manuscript draft | Completed |
| External validation | Not yet completed |
| Patient-level aggregation | Not yet completed |

---

## 2. Recommended Q1 Paper Title

### 2.1 Primary Title

> **Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT: Natural-Prevalence Evaluation, Paired Statistical Backbone Comparison, and CT Window Ablation**

This is the recommended title because it communicates:

| Title Component | Reason |
|---|---|
| Explainable | Signals Grad-CAM / interpretability |
| ConvNeXt-Based | Highlights the best-performing modern CNN |
| Intracranial Hemorrhage | Clear clinical target |
| Head CT | Clear imaging modality |
| Natural-Prevalence Evaluation | Signals realistic evaluation |
| Paired Statistical Backbone Comparison | Signals statistical rigor |
| CT Window Ablation | Signals experimental depth |

### 2.2 Alternative Titles

1. **Natural-Prevalence Evaluation of ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT**
2. **A Reproducible Deep Learning Framework for Intracranial Hemorrhage Triage with Bootstrap Evaluation and CT Window Ablation**
3. **ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT: Calibration, Statistical Comparison, and Explainability**
4. **AI-Assisted Head CT Triage for Intracranial Hemorrhage Using Modern Convolutional Networks**

---

## 3. Research Motivation

### 3.1 Clinical Problem

Intracranial hemorrhage is a potentially life-threatening radiological finding. In high-volume emergency and radiology workflows, head CT studies may queue before expert review. A triage model can support radiologists by ranking likely hemorrhage-positive images earlier.

### 3.2 AI Triage Opportunity

This project focuses on **triage**, not autonomous diagnosis.

The intended use is:

```text
Model prediction → priority score → radiology queue assistance → radiologist final interpretation
```

The model should therefore prioritize:

1. high recall;
2. low false negative rate;
3. manageable workload;
4. interpretable output;
5. robust evaluation under realistic prevalence.

### 3.3 Research Question

> Can a modern CNN backbone prioritize hemorrhage-positive head CT images under natural-prevalence holdout conditions while maintaining high recall and manageable workload?

Sub-questions:

1. Which backbone performs best: EfficientNet-B0, DenseNet121, or ConvNeXt-Tiny?
2. Is ConvNeXt-Tiny statistically better under paired bootstrap comparison?
3. Does multi-window CT input improve over brain-window-only input?
4. What triage threshold provides high sensitivity?
5. What workload is required to capture most hemorrhage-positive cases?
6. How calibrated are the predicted probabilities?
7. What failure modes remain?

---

## 4. Completed Work Summary

### 4.1 Engineering Work Completed

| Work Item | Status |
|---|---|
| Project folder setup | Completed |
| Conda environment used | Completed |
| CUDA/GPU verification | Completed |
| Dependency/import check | Completed |
| Kaggle authentication | Completed |
| RSNA dataset download | Completed |
| DICOM extraction | Completed |
| Label preprocessing | Completed |
| CT windowing utilities | Completed |
| Dataset loader | Completed |
| Model builder | Completed |
| Training scripts | Completed |
| Evaluation scripts | Completed |
| GitHub repository | Completed |
| `.gitignore` medical-data protection | Completed |

### 4.2 Experimental Work Completed

| Experiment | Status |
|---|---|
| Synthetic CT windowing test | Completed |
| Real DICOM window visualization | Completed locally |
| Smoke training | Completed |
| EfficientNet-B0 2k training | Completed |
| EfficientNet-B0 10k training | Completed |
| DenseNet121 10k training | Completed |
| ConvNeXt-Tiny 10k training | Completed |
| ConvNeXt-Tiny brain-only ablation | Completed |
| Natural holdout evaluation | Completed |
| ROC / PR / calibration generation | Completed |
| Bootstrap confidence intervals | Completed |
| Paired backbone comparison | Completed |
| Paired ablation comparison | Completed |
| Error analysis | Completed |
| Grad-CAM generation | Completed locally |

### 4.3 Paper-Readiness Work Completed

| Paper Component | Status |
|---|---|
| README / project report | Completed |
| Manuscript draft | Completed |
| Figure plan | Completed |
| Q1 checklist | Completed |
| Statistical summary | Completed |
| Public-safe GitHub figures | Completed |
| Public-safe GitHub reports | Completed |

---

## 5. Literature Positioning

### 5.1 RSNA ICH Challenge Context

The RSNA Intracranial Hemorrhage Detection Challenge was created to support AI research for detecting acute intracranial hemorrhage and hemorrhage subtypes from head CT images. RSNA describes the challenge as using more than 25,000 annotated cranial CT exams, created with ASNR, MD.ai, and Kaggle.

This project uses the RSNA challenge data but evaluates a compact, reproducible experimental pipeline under a natural-prevalence holdout design.

### 5.2 Relevant ICH Detection Papers

| Study | Key Point | Relation to This Project |
|---|---|---|
| Flanders et al., 2020 | Describes construction of RSNA 2019 Brain CT Hemorrhage Challenge dataset | Dataset context |
| Wang et al., 2021 | RSNA first-place solution; reported AUCs around 0.99 for ICH subtype classification and external validation | Strong literature benchmark; not directly comparable because this project uses a smaller 10k training subset |
| Chilamkurthy et al., 2018 | Critical findings detection on head CT, including ICH | Supports clinical relevance of AI triage |
| Arbabshirani et al., 2018 | 3D deep learning for ICH detection and workflow prioritization | Supports triage workflow motivation |
| Nguyen et al., 2020 | CNN-LSTM approach with multi-window CT input | Related to windowing and slice-sequence modeling |
| Chen et al., 2024 | Efficient model for ICH classification, validated on RSNA and external datasets | Relevant external-validation direction |

Important comparison note:

> This repository should not claim state-of-the-art against full RSNA challenge winners because the training scale, split design, objective, and external validation protocol are different.

### 5.3 Backbone Literature

| Backbone | Literature Basis | Why Used Here |
|---|---|---|
| EfficientNet-B0 | EfficientNet compound scaling | Lightweight baseline |
| DenseNet121 | Dense feature reuse | Common strong medical imaging comparator |
| ConvNeXt-Tiny | Modernized pure ConvNet design | Best-performing backbone in this project |

### 5.4 Explainability Literature

Grad-CAM is used as the qualitative interpretability technique. It generates class-discriminative localization maps from gradients flowing into convolutional feature maps. In this project, Grad-CAM outputs were generated locally but not uploaded publicly because they are derived from medical images.

---

## 6. Dataset and Experimental Splits

### 6.1 Dataset Source

Dataset:

```text
RSNA Intracranial Hemorrhage Detection
```

Task:

```text
Multi-label hemorrhage classification from head CT images
```

### 6.2 Label Schema

| Label | Meaning |
|---|---|
| `any` | Any intracranial hemorrhage |
| `epidural` | Epidural hemorrhage |
| `intraparenchymal` | Intraparenchymal hemorrhage |
| `intraventricular` | Intraventricular hemorrhage |
| `subarachnoid` | Subarachnoid hemorrhage |
| `subdural` | Subdural hemorrhage |

### 6.3 Training and Holdout Splits

| Split | Size | Distribution |
|---|---:|---|
| Full processed label table | 752,803 unique images | RSNA processed labels |
| Any-positive images | 107,933 | Any hemorrhage positive |
| Balanced train subset | 10,000 | 5,000 normal / 5,000 positive |
| Natural holdout | 5,000 | 4,299 normal / 701 positive |
| Holdout prevalence | 14.02% | Natural-prevalence evaluation |

---

## 7. Methodology

### 7.1 Pipeline Overview

```text
DICOM CT image
    ↓
Pixel array extraction
    ↓
Hounsfield Unit conversion
    ↓
CT windowing
    ↓
3-channel tensor
    ↓
CNN backbone
    ↓
Six-label sigmoid outputs
    ↓
Thresholding
    ↓
Triage ranking
    ↓
Statistical evaluation
```

### 7.2 DICOM to Hounsfield Unit Conversion

Raw DICOM pixels are converted to Hounsfield Units:

```math
HU(x) = x \cdot s + b
```

where:

- \(x\) = raw pixel value;
- \(s\) = `RescaleSlope`;
- \(b\) = `RescaleIntercept`.

### 7.3 CT Windowing

For window center \(c\) and width \(w\):

```math
I_w =
\frac{
\text{clip}(HU, c - \frac{w}{2}, c + \frac{w}{2}) - (c - \frac{w}{2})
}{w}
```

Multi-window input:

```math
X = [I_{brain}, I_{subdural}, I_{bone}]
```

| Channel | Window | Center | Width |
|---|---|---:|---:|
| 1 | Brain | 40 | 80 |
| 2 | Subdural | 80 | 200 |
| 3 | Bone | 600 | 2800 |

Brain-only ablation:

```math
X_{brain-only} = [I_{brain}, I_{brain}, I_{brain}]
```

### 7.4 Multi-Label Prediction

The model outputs six logits:

```math
z = f_{\theta}(X) \in \mathbb{R}^{6}
```

Sigmoid probability:

```math
\hat{y}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}
```

### 7.5 Loss Function

Weighted binary cross-entropy:

```math
\mathcal{L} =
-\frac{1}{K}
\sum_{k=1}^{K}
\left[
w_k y_k \log(\hat{y}_k)
+
(1-y_k)\log(1-\hat{y}_k)
\right]
```

where \(K=6\).

### 7.6 Triage Thresholding

```math
\hat{y}_{any}^{binary} =
\begin{cases}
1, & \hat{p}_{any} \geq \tau \\
0, & \hat{p}_{any} < \tau
\end{cases}
```

For the best model:

```text
High-sensitivity threshold: τ = 0.10
```

---

## 8. Model Backbones

### 8.1 EfficientNet-B0

Role:

```text
Lightweight baseline
```

Holdout result:

| Metric | Value |
|---|---:|
| Any AUC | 0.916446 |
| Any AP | 0.709731 |

### 8.2 DenseNet121

Role:

```text
Strong classic medical imaging comparator
```

Holdout result:

| Metric | Value |
|---|---:|
| Any AUC | 0.936476 |
| Any AP | 0.781555 |

### 8.3 ConvNeXt-Tiny

Role:

```text
Best-performing modern CNN backbone
```

Holdout result:

| Metric | Value |
|---|---:|
| Any AUC | 0.945789 |
| Any AP | 0.816036 |

---

## 9. Training Protocol

### 9.1 Shared Training Setup

| Parameter | Value |
|---|---:|
| Training subset | 10,000 CT images |
| Training distribution | 5,000 normal / 5,000 hemorrhage-positive |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Mixed precision | Enabled |
| Primary validation metric | Any AUC |

### 9.2 Hardware

```text
GPU: NVIDIA GeForce RTX 5060
CUDA available: True
```

---

## 10. Evaluation Protocol

### 10.1 Primary Metrics

1. ROC AUC;
2. Average Precision.

### 10.2 Secondary Metrics

1. Precision;
2. Recall;
3. F1;
4. False negative rate;
5. False positive rate;
6. Prioritized workload rate;
7. Brier score;
8. Calibration curve.

### 10.3 Statistical Testing

Bootstrap procedures:

1. model-level bootstrap confidence interval;
2. paired bootstrap backbone comparison;
3. paired bootstrap CT-window ablation comparison.

Paired metric difference:

```math
\Delta = M_A - M_B
```

Bootstrap 95% confidence interval:

```math
CI_{95\%} =
[Q_{2.5\%}(\Delta^{(b)}), Q_{97.5\%}(\Delta^{(b)})]
```

---

## 11. Quantitative Results

### 11.1 Backbone Comparison

| Model | Any AUC | Any AP | Recall ≥90% Workload |
|---|---:|---:|---:|
| EfficientNet-B0 | 0.916446 | 0.709731 | 35.48% |
| DenseNet121 | 0.936476 | 0.781555 | 34.12% |
| ConvNeXt-Tiny | **0.945789** | **0.816036** | **32.44%** |

### 11.2 Best Model: ConvNeXt-Tiny

| Metric | Value |
|---|---:|
| Any AUC | 0.945789 |
| Any AUC 95% CI | 0.936999 – 0.953849 |
| Any AP | 0.816036 |
| Any AP 95% CI | 0.792165 – 0.839746 |
| Brier score | 0.063986 |

### 11.3 Per-Label Results

| Label | Positive Count | Prevalence | AUC | AP |
|---|---:|---:|---:|---:|
| any | 701 | 0.1402 | 0.945789 | 0.816036 |
| epidural | 17 | 0.0034 | 0.938839 | 0.056214 |
| intraparenchymal | 254 | 0.0508 | 0.970937 | 0.770624 |
| intraventricular | 160 | 0.0320 | 0.990244 | 0.847204 |
| subarachnoid | 234 | 0.0468 | 0.926740 | 0.498531 |
| subdural | 301 | 0.0602 | 0.914619 | 0.597648 |

---

## 12. Paired Statistical Backbone Comparison

### 12.1 ConvNeXt-Tiny vs DenseNet121

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.009313 | 0.001840 – 0.016482 | 0.009 |
| AP | +0.034482 | 0.017094 – 0.051272 | 0.000 |

### 12.2 ConvNeXt-Tiny vs EfficientNet-B0

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.029343 | 0.019851 – 0.039057 | 0.000 |
| AP | +0.106306 | 0.081276 – 0.132075 | 0.000 |

### 12.3 DenseNet121 vs EfficientNet-B0

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.020030 | 0.011076 – 0.029448 | 0.000 |
| AP | +0.071824 | 0.046238 – 0.097875 | 0.000 |

Interpretation:

> ConvNeXt-Tiny produced a statistically stable improvement over DenseNet121 and EfficientNet-B0 under paired bootstrap evaluation.

---

## 13. CT Window Ablation Study

### 13.1 Ablation Design

Two ConvNeXt-Tiny variants were compared:

| Variant | Input |
|---|---|
| Multi-window | Brain + subdural + bone |
| Brain-only | Brain channel repeated three times |

### 13.2 Ablation Result

| Input | Any AUC | Any AP |
|---|---:|---:|
| Multi-window | 0.945789 | 0.816036 |
| Brain-only | 0.947194 | 0.815377 |

### 13.3 Paired Bootstrap Ablation

| Metric | Multi-window Minus Brain-only | Bootstrap p-value |
|---|---:|---:|
| AUC | -0.001405 | 0.646 |
| AP | +0.000659 | 0.929 |

Interpretation:

> Multi-window input did not show statistically meaningful improvement over brain-window-only input for slice-level any-hemorrhage triage.

This is a scientifically important negative result.

---

## 14. Triage Simulation

### 14.1 High-Sensitivity Operating Point

For ConvNeXt-Tiny:

```text
Threshold = 0.10
```

| Metric | Value |
|---|---:|
| TP | 653 |
| FP | 969 |
| FN | 48 |
| TN | 3330 |
| Recall | 0.931526 |
| Precision | 0.402589 |
| False negative rate | 0.068474 |
| Prioritized workload | 32.44% |

### 14.2 Clinical Workflow Interpretation

The model captured:

```text
653 / 701 hemorrhage-positive images
```

while prioritizing:

```text
32.44% of the CT workload
```

This supports the triage objective but does not establish clinical readiness.

---

## 15. Calibration Analysis

ConvNeXt-Tiny Brier score:

```text
0.063986
```

Calibration figure:

```text
outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png
```

Calibration remains important because real triage workflows may depend on stable probability thresholds.

---

## 16. Error Analysis

Error analysis was performed to inspect:

1. true positives;
2. false positives;
3. false negatives;
4. true negatives;
5. subtype-specific missed cases;
6. probability distribution by error group.

Public aggregate error-analysis figures are included. Private case-level CSV files with image IDs are intentionally excluded from GitHub.

---

## 17. Qualitative Explainability

Grad-CAM was generated locally for selected cases.

Generated locally:

```text
outputs/figures/gradcam_model10000/
outputs/figures/paper_ready/figure_gradcam_composite_model10000.png
outputs/figures/paper_ready/figure_gradcam_composite_model10000.pdf
```

These files are not publicly embedded because they may contain patient-derived CT visualizations.

---

## 18. Embedded Figures for Q1-Style Review

### 18.1 Backbone Comparison Figures

#### 18.1.1 Backbone AUC/AP Comparison

![Backbone AUC/AP Comparison](outputs/figures/model_comparison/backbone_any_auc_ap_comparison.png)

#### 18.1.2 Backbone Triage Recall Curve

![Backbone Triage Recall Curve](outputs/figures/model_comparison/backbone_triage_recall_curve.png)

#### 18.1.3 Backbone Per-Label AUC Comparison

![Backbone Per-Label AUC Comparison](outputs/figures/model_comparison/backbone_per_label_auc_comparison.png)

#### 18.1.4 Workload Required for Recall ≥90%

![Backbone Recall90 Workload Comparison](outputs/figures/model_comparison/backbone_recall90_workload_comparison.png)

### 18.2 ConvNeXt-Tiny Main Paper Figures

#### 18.2.1 ROC Curve

![ConvNeXt-Tiny ROC Curve](outputs/figures/paper_metrics_convnext_tiny/roc_curve_any_convnext_tiny_10000.png)

#### 18.2.2 Precision-Recall Curve

![ConvNeXt-Tiny PR Curve](outputs/figures/paper_metrics_convnext_tiny/pr_curve_any_convnext_tiny_10000.png)

#### 18.2.3 Calibration Curve

![ConvNeXt-Tiny Calibration Curve](outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png)

#### 18.2.4 Per-Label AUC/AP

![ConvNeXt-Tiny Per-Label AUC/AP](outputs/figures/paper_metrics_convnext_tiny/per_label_auc_ap_convnext_tiny_10000.png)

#### 18.2.5 Confusion Matrix at High-Sensitivity Threshold 0.10

![ConvNeXt-Tiny Triage Confusion Matrix](outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_triage_recall90_threshold_010.png)

#### 18.2.6 Confusion Matrix at Best-F1 Threshold 0.65

![ConvNeXt-Tiny Best-F1 Confusion Matrix](outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_best_f1_threshold_065.png)

### 18.3 Bootstrap and Statistical Figures

#### 18.3.1 ConvNeXt-Tiny Bootstrap CI

![ConvNeXt-Tiny Bootstrap CI](outputs/figures/paper_metrics/bootstrap_ci_auc_ap_convnext_tiny_10000.png)

#### 18.3.2 Paired Bootstrap Backbone Comparison

![Paired Bootstrap Backbone Comparison](outputs/figures/statistical_comparison/paired_bootstrap_auc_ap_difference.png)

### 18.4 Ablation Study Figure

#### 18.4.1 Multi-Window vs Brain-Only Paired Bootstrap

![Ablation Study Paired Bootstrap](outputs/figures/ablation_study/paired_bootstrap_multiwindow_vs_brainonly.png)

### 18.5 Error Analysis Figures

#### 18.5.1 Error Analysis Confusion Matrix

![Error Analysis Confusion Matrix](outputs/figures/error_analysis/error_analysis_confusion_matrix_threshold025.png)

#### 18.5.2 False Negative Rate by Subtype

![False Negative Rate by Subtype](outputs/figures/error_analysis/false_negative_rate_by_subtype_threshold025.png)

#### 18.5.3 Probability Distribution by Error Group

![Probability Distribution by Error Group](outputs/figures/error_analysis/probability_distribution_by_error_group_threshold025.png)

### 18.6 Supplementary EfficientNet-B0 Baseline Figures

#### 18.6.1 EfficientNet-B0 ROC Curve

![EfficientNet-B0 ROC Curve](outputs/figures/paper_metrics/roc_curve_any_model10000.png)

#### 18.6.2 EfficientNet-B0 PR Curve

![EfficientNet-B0 PR Curve](outputs/figures/paper_metrics/pr_curve_any_model10000.png)

#### 18.6.3 EfficientNet-B0 Calibration Curve

![EfficientNet-B0 Calibration Curve](outputs/figures/paper_metrics/calibration_curve_any_model10000.png)

#### 18.6.4 EfficientNet-B0 Per-Label AUC/AP

![EfficientNet-B0 Per-Label AUC/AP](outputs/figures/paper_metrics/per_label_auc_ap_model10000.png)

#### 18.6.5 EfficientNet-B0 Triage Confusion Matrix

![EfficientNet-B0 Triage Confusion Matrix](outputs/figures/paper_metrics/confusion_matrix_any_triage_recall90_threshold_025.png)

#### 18.6.6 EfficientNet-B0 Best-F1 Confusion Matrix

![EfficientNet-B0 Best-F1 Confusion Matrix](outputs/figures/paper_metrics/confusion_matrix_any_best_f1_threshold_090.png)

#### 18.6.7 EfficientNet-B0 Bootstrap CI

![EfficientNet-B0 Bootstrap CI](outputs/figures/paper_metrics/bootstrap_ci_auc_ap_model10000.png)

### 18.7 Experiment Summary Figures

#### 18.7.1 Model AUC/AP Comparison

![Model AUC/AP Comparison](outputs/figures/experiment_summary/model_auc_ap_comparison.png)

#### 18.7.2 Subtype AUC/AP Model 10000

![Subtype AUC/AP](outputs/figures/experiment_summary/subtype_auc_ap_model10000.png)

#### 18.7.3 Threshold Trade-Off

![Threshold Trade-Off](outputs/figures/experiment_summary/threshold_tradeoff_model10000.png)

#### 18.7.4 Triage Precision Curve

![Triage Precision Curve](outputs/figures/experiment_summary/triage_precision_curve.png)

#### 18.7.5 Triage Recall Curve

![Triage Recall Curve](outputs/figures/experiment_summary/triage_recall_curve.png)

---

## 19. Generated Reports

Key report files:

```text
outputs/reports/backbone_comparison_holdout_5000.csv
outputs/reports/backbone_comparison_per_label_auc_ap.csv
outputs/reports/backbone_comparison_summary.md
outputs/reports/bootstrap_ci_convnext_tiny_10000.csv
outputs/reports/bootstrap_ci_convnext_tiny_10000.json
outputs/reports/bootstrap_threshold_ci_any_convnext_tiny_10000.csv
outputs/reports/paired_bootstrap_backbone_comparison.csv
outputs/reports/paired_bootstrap_backbone_comparison.json
outputs/reports/paired_bootstrap_ablation_multiwindow_vs_brainonly.csv
outputs/reports/paired_bootstrap_ablation_multiwindow_vs_brainonly.json
outputs/reports/paper_metrics_summary_convnext_tiny_10000.json
outputs/reports/calibration_curve_any_convnext_tiny_10000.csv
outputs/reports/paper_metrics_per_label_auc_ap_convnext_tiny_10000.csv
```

---

## 20. Repository Structure

```text
ai-ct-hemorrhage-triage/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── datasets/
│   ├── models/
│   ├── preprocessing/
│   ├── evaluation/
│   ├── training/
│   ├── visualization/
│   └── utils/
├── scripts/
│   ├── train_baseline_10000.py
│   ├── train_densenet121_10000.py
│   ├── train_convnext_tiny_10000.py
│   ├── train_convnext_tiny_brain_only_10000.py
│   ├── evaluate_holdout_5000_convnext_tiny.py
│   ├── generate_backbone_comparison_report.py
│   ├── generate_paper_metrics_figures_convnext_tiny.py
│   ├── bootstrap_confidence_intervals_convnext_tiny.py
│   ├── paired_bootstrap_backbone_comparison.py
│   ├── paired_bootstrap_ablation_multiwindow_vs_brainonly.py
│   └── generate_manuscript_draft.py
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── reports/
└── paper/
    ├── manuscript_draft.md
    ├── manuscript_draft_v2.md
    ├── figure_plan.md
    ├── q1_submission_gap_checklist.md
    └── statistical_and_ablation_summary.md
```

---

## 21. Reproducibility Workflow

### 21.1 Environment

```powershell
conda activate main
pip install -r requirements.txt
```

### 21.2 Dataset Download

```powershell
kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p data\raw\rsna
```

### 21.3 Label Preparation

```powershell
python scripts\prepare_rsna_labels.py
```

### 21.4 Data Extraction

```powershell
python scripts\extract_rsna_train_10000.py
python scripts\extract_rsna_holdout_5000.py
```

### 21.5 Model Training

```powershell
python scripts\train_baseline_10000.py
python scripts\train_densenet121_10000.py
python scripts\train_convnext_tiny_10000.py
python scripts\train_convnext_tiny_brain_only_10000.py
```

### 21.6 Evaluation

```powershell
python scripts\evaluate_holdout_5000_model10000.py
python scripts\evaluate_holdout_5000_densenet121.py
python scripts\evaluate_holdout_5000_convnext_tiny.py
python scripts\evaluate_holdout_5000_convnext_tiny_brain_only.py
```

### 21.7 Report Generation

```powershell
python scripts\generate_backbone_comparison_report.py
python scripts\generate_paper_metrics_figures_convnext_tiny.py
python scripts\bootstrap_confidence_intervals_convnext_tiny.py
python scripts\paired_bootstrap_backbone_comparison.py
python scripts\paired_bootstrap_ablation_multiwindow_vs_brainonly.py
```

---

## 22. GitHub Data Governance

### 22.1 Included in GitHub

This repository includes:

1. source code;
2. training scripts;
3. evaluation scripts;
4. aggregate metrics;
5. public-safe aggregate figures;
6. paper draft;
7. statistical summaries.

### 22.2 Excluded from GitHub

This repository intentionally excludes:

1. raw DICOM files;
2. RSNA zip files;
3. model checkpoints;
4. prediction CSV files with image-level identifiers;
5. patient-derived Grad-CAM CT overlays;
6. processed medical images.

This is controlled through `.gitignore`.

---

## 23. Q1-Readiness Assessment

### 23.1 Completed

- [x] DICOM preprocessing
- [x] HU conversion
- [x] CT windowing
- [x] Natural-prevalence holdout
- [x] EfficientNet-B0 baseline
- [x] DenseNet121 baseline
- [x] ConvNeXt-Tiny baseline
- [x] Brain-only ablation
- [x] Bootstrap confidence interval
- [x] Paired bootstrap backbone comparison
- [x] Paired bootstrap ablation comparison
- [x] Calibration curve
- [x] Brier score
- [x] Error analysis
- [x] Manuscript draft
- [x] GitHub-safe public repo

### 23.2 Still Needed

- [ ] External validation
- [ ] Patient-level aggregation
- [ ] Study-level triage simulation
- [ ] Calibration improvement
- [ ] Radiologist review of qualitative cases
- [ ] Full academic editing
- [ ] Journal-specific formatting

---

## 24. Remaining Gaps Before Strong Q1 Submission

The largest remaining scientific gaps are:

1. **External validation**  
   Current holdout is internal to RSNA.

2. **Patient-level aggregation**  
   Current model is slice-level.

3. **Prospective workflow validation**  
   Current triage impact is simulated.

4. **Calibration optimization**  
   Calibration is measured but not improved.

5. **Clinical expert review**  
   Grad-CAM and failure cases should be reviewed by radiologists.

---

## 25. Limitations

1. Slice-level evaluation only.
2. Internal RSNA holdout only.
3. No external hospital validation.
4. No prospective clinical workflow test.
5. Rare subtype AP is unstable, especially epidural due to only 17 positive holdout cases.
6. Grad-CAM is qualitative, not causal evidence.
7. The model is not a medical device.

---

## 26. Future Work

1. External validation on independent hospital data.
2. Patient-level aggregation.
3. Study-level triage simulation.
4. Temperature scaling or isotonic calibration.
5. Full RSNA-scale training.
6. Transformer and hybrid architecture comparison.
7. Radiologist-in-the-loop evaluation.
8. Prospective workflow trial.
9. Deployment latency analysis.
10. Journal-formatted manuscript preparation.

---

## 27. Medical and Ethical Disclaimer

This repository is for **research and educational purposes only**.

The model is:

- not clinically validated;
- not regulatory approved;
- not a medical device;
- not intended for diagnosis;
- not intended for patient management;
- not a replacement for radiologists.

Clinical translation requires institutional review, external validation, prospective evaluation, and regulatory assessment.

---

## 28. Suggested Citation

```bibtex
@misc{rizal2026convnext_ich_triage,
  title={Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT: Natural-Prevalence Evaluation, Paired Statistical Backbone Comparison, and CT Window Ablation},
  author={Rizal Fanex},
  year={2026},
  note={Research prototype and reproducible GitHub implementation}
}
```

---

## 29. References

1. Flanders AE, Prevedello LM, Shih G, et al. **Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge.** *Radiology: Artificial Intelligence*, 2020.  
   <https://pubmed.ncbi.nlm.nih.gov/33937827/>

2. RSNA. **RSNA Intracranial Hemorrhage Detection Challenge.**  
   <https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019>

3. Wang X, Shen T, Yang S, et al. **A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans.** *NeuroImage: Clinical*, 2021.  
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC8377493/>

4. Chilamkurthy S, Ghosh R, Tanamala S, et al. **Deep learning algorithms for detection of critical findings in head CT scans.** *The Lancet*, 2018.  
   <https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(18)31645-3/abstract>

5. Tan M, Le QV. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML, 2019.  
   <https://arxiv.org/abs/1905.11946>

6. Huang G, Liu Z, van der Maaten L, Weinberger KQ. **Densely Connected Convolutional Networks.** CVPR, 2017.  
   <https://arxiv.org/abs/1608.06993>

7. Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. **A ConvNet for the 2020s.** CVPR, 2022.  
   <https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html>

8. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.** ICCV, 2017.  
   <https://arxiv.org/abs/1610.02391>

9. Nguyen NT, Tran DQ, Nguyen NT, Nguyen HQ. **A CNN-LSTM Architecture for Detection of Intracranial Hemorrhage on CT scans.** 2020.  
   <https://arxiv.org/abs/2005.10992>

10. Chen YR, Chen CC, Kuo CF, Lin CH. **An efficient deep neural network for automatic classification of acute intracranial hemorrhages in brain CT scans.** *Computers in Biology and Medicine*, 2024.  
   <https://doi.org/10.1016/j.compbiomed.2024.108587>
