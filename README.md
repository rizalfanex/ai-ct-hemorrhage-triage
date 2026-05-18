# Explainable ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT  
## Natural-Prevalence Evaluation, Statistical Backbone Comparison, CT Window Ablation, Calibration, and Explainability

> **Q1-oriented research repository for AI-assisted intracranial hemorrhage (ICH) triage from non-contrast head CT images.**  
> This repository documents a complete experimental workflow: RSNA DICOM preprocessing, Hounsfield Unit conversion, CT windowing, CNN backbone comparison, ConvNeXt-Tiny best-model evaluation, statistical bootstrap testing, CT window ablation, calibration analysis, triage simulation, Grad-CAM qualitative interpretation, manuscript draft generation, and GitHub-safe public release without medical images or checkpoints.

---

## Table of Contents

1. [Executive Summary](#executive-summary)  
2. [Most Suitable Q1-Style Paper Title](#most-suitable-q1-style-paper-title)  
3. [Research Problem](#research-problem)  
4. [Why This Research Matters](#why-this-research-matters)  
5. [Project Status](#project-status)  
6. [What Has Been Completed](#what-has-been-completed)  
7. [Relevant Literature and Positioning](#relevant-literature-and-positioning)  
8. [Dataset and Experimental Splits](#dataset-and-experimental-splits)  
9. [Methodological Pipeline](#methodological-pipeline)  
10. [Mathematical Formulation](#mathematical-formulation)  
11. [Model Backbones](#model-backbones)  
12. [Training Configuration](#training-configuration)  
13. [Evaluation Protocol](#evaluation-protocol)  
14. [Primary Quantitative Results](#primary-quantitative-results)  
15. [Best Model: ConvNeXt-Tiny](#best-model-convnext-tiny)  
16. [Per-Label Performance](#per-label-performance)  
17. [Backbone Comparison](#backbone-comparison)  
18. [Paired Bootstrap Statistical Comparison](#paired-bootstrap-statistical-comparison)  
19. [CT Window Ablation Study](#ct-window-ablation-study)  
20. [Triage Simulation](#triage-simulation)  
21. [Calibration Analysis](#calibration-analysis)  
22. [Error Analysis](#error-analysis)  
23. [Qualitative Explainability](#qualitative-explainability)  
24. [Generated Figures](#generated-figures)  
25. [Generated Reports](#generated-reports)  
26. [Repository Structure](#repository-structure)  
27. [Reproducibility Workflow](#reproducibility-workflow)  
28. [GitHub Data Governance](#github-data-governance)  
29. [Q1-Readiness Assessment](#q1-readiness-assessment)  
30. [Remaining Gaps Before Serious Q1 Submission](#remaining-gaps-before-serious-q1-submission)  
31. [Limitations](#limitations)  
32. [Future Work](#future-work)  
33. [Medical and Ethical Disclaimer](#medical-and-ethical-disclaimer)  
34. [Suggested Citation](#suggested-citation)  
35. [References](#references)

---

## Executive Summary

This repository presents a **research-grade AI pipeline** for slice-level intracranial hemorrhage triage on head CT images.

The current best model is **ConvNeXt-Tiny**, evaluated on a **5,000-image natural-prevalence holdout set**. The model achieved:

| Metric | Value |
|---|---:|
| Any-hemorrhage AUC | **0.945789** |
| AUC 95% bootstrap CI | **0.936999 – 0.953849** |
| Any-hemorrhage Average Precision | **0.816036** |
| AP 95% bootstrap CI | **0.792165 – 0.839746** |
| Brier score | **0.063986** |
| High-sensitivity threshold | **0.10** |
| Recall at threshold 0.10 | **0.931526** |
| Precision at threshold 0.10 | **0.402589** |
| Prioritized workload at threshold 0.10 | **32.44%** |
| False negative rate at threshold 0.10 | **6.85%** |

The project includes:

- DICOM-to-Hounsfield Unit preprocessing;
- CT windowing;
- multi-label classification;
- EfficientNet-B0, DenseNet121, and ConvNeXt-Tiny comparison;
- paired bootstrap statistical testing;
- bootstrap confidence intervals;
- calibration analysis;
- CT window ablation;
- error analysis;
- Grad-CAM qualitative interpretation;
- manuscript draft and submission checklist;
- public GitHub release with data governance safeguards.

---

## Most Suitable Q1-Style Paper Title

Recommended title:

> **Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT: Natural-Prevalence Evaluation, Paired Statistical Backbone Comparison, and CT Window Ablation**

Why this title is strong:

| Title Element | Reason |
|---|---|
| Explainable | Signals interpretability / Grad-CAM component |
| ConvNeXt-Based | Highlights the best-performing modern CNN backbone |
| Intracranial Hemorrhage | Defines the clinical target |
| Head CT | Defines modality |
| Natural-Prevalence Evaluation | Signals realistic evaluation beyond balanced validation |
| Paired Statistical Backbone Comparison | Signals methodological rigor |
| CT Window Ablation | Signals experimental depth and honest ablation |

Alternative titles:

1. **Natural-Prevalence Evaluation of ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT**
2. **A Reproducible Deep Learning Framework for Intracranial Hemorrhage Triage with Bootstrap Evaluation and CT Window Ablation**
3. **ConvNeXt-Based Intracranial Hemorrhage Triage on Head CT: Calibration, Statistical Comparison, and Explainability**
4. **AI-Assisted Head CT Triage for Intracranial Hemorrhage Using Modern Convolutional Networks**

---

## Research Problem

Intracranial hemorrhage is a potentially life-threatening condition that requires rapid radiological assessment. In high-volume emergency or radiology settings, a triage model may help prioritize CT images with high hemorrhage probability.

The research question:

> Can a modern convolutional neural network prioritize hemorrhage-positive head CT images under natural-prevalence holdout conditions while maintaining high recall and manageable workload?

Sub-questions:

1. Which CNN backbone performs best under identical training/evaluation protocol?
2. Does ConvNeXt-Tiny significantly outperform EfficientNet-B0 and DenseNet121?
3. Does multi-window CT input outperform brain-window-only input?
4. What threshold gives high-sensitivity triage?
5. What workload is required to capture most hemorrhage-positive cases?
6. How calibrated are the model probabilities?
7. What failure modes remain?

---

## Why This Research Matters

A triage model is not designed to replace radiologists. It is designed to assist workflow prioritization.

A clinically meaningful triage model should:

- prioritize high-risk images earlier;
- maintain high sensitivity;
- reduce false negatives;
- avoid excessive false-positive workload;
- report uncertainty and confidence intervals;
- be interpretable enough for expert review;
- be evaluated under realistic prevalence.

This repository focuses on **triage-oriented evaluation**, not just classification accuracy.

---

## Project Status

| Area | Status |
|---|---|
| Dataset download | Completed locally |
| Label preprocessing | Completed |
| DICOM preprocessing | Completed |
| CT windowing | Completed |
| EfficientNet-B0 baseline | Completed |
| DenseNet121 baseline | Completed |
| ConvNeXt-Tiny baseline | Completed |
| Natural holdout evaluation | Completed |
| ROC / PR / calibration figures | Completed |
| Bootstrap confidence intervals | Completed |
| Paired bootstrap backbone comparison | Completed |
| Brain-window-only ablation | Completed |
| Paired bootstrap ablation test | Completed |
| Grad-CAM qualitative outputs | Generated locally |
| GitHub-safe public release | Completed |
| Manuscript draft | Completed |
| Q1 submission | Not yet final; still requires external validation and manuscript polishing |

---

## What Has Been Completed

This section records the actual project work already completed.

### 1. Environment and Dependency Setup

Completed:

- Conda environment: `main`
- Core libraries verified:
  - PyTorch
  - CUDA
  - NumPy
  - Pandas
  - scikit-learn
  - pydicom
  - OpenCV
  - Albumentations
  - timm
  - torchmetrics
  - MONAI
- GPU test passed on NVIDIA GeForce RTX 5060.

### 2. RSNA Dataset Access

Completed:

- Kaggle authentication configured.
- RSNA Intracranial Hemorrhage Detection dataset downloaded locally.
- Full dataset zip stored locally, not uploaded to GitHub.
- Raw medical data excluded via `.gitignore`.

### 3. Data Governance

Completed:

- `data/` ignored.
- `.dcm`, `.dicom`, `.zip`, `.pt`, `.pth`, `.ckpt`, `.onnx` ignored.
- Prediction CSV files with image-level identifiers ignored.
- Grad-CAM patient-derived visualizations ignored.
- GitHub public repository contains only code, aggregate metrics, aggregate reports, non-patient-derived figures, and manuscript files.

### 4. DICOM and CT Windowing

Completed:

- DICOM loading.
- Pixel-to-Hounsfield Unit conversion.
- Multi-window CT preprocessing:
  - brain window;
  - subdural window;
  - bone window.
- Synthetic CT preprocessing test passed.
- Real DICOM visualization generated locally but excluded from GitHub.

### 5. Dataset Construction

Completed:

- RSNA label pivot from long to wide multi-label format.
- Balanced training subset of 10,000 images:
  - 5,000 normal;
  - 5,000 hemorrhage-positive.
- Natural-prevalence holdout of 5,000 images:
  - 4,299 normal;
  - 701 any-hemorrhage positive.

### 6. Baseline and Backbone Training

Completed:

| Model | Training Size | Status |
|---|---:|---|
| EfficientNet-B0 | 2,000 | Completed |
| EfficientNet-B0 | 10,000 | Completed |
| DenseNet121 | 10,000 | Completed |
| ConvNeXt-Tiny | 10,000 | Completed |
| ConvNeXt-Tiny brain-only | 10,000 | Completed |

### 7. Natural Holdout Evaluation

Completed for:

- EfficientNet-B0;
- DenseNet121;
- ConvNeXt-Tiny;
- ConvNeXt-Tiny brain-only.

### 8. Paper Metrics

Completed:

- ROC curve;
- PR curve;
- calibration curve;
- confusion matrix;
- per-label AUC/AP;
- Brier score;
- threshold sweep;
- triage simulation.

### 9. Statistical Testing

Completed:

- bootstrap confidence intervals for ConvNeXt-Tiny;
- paired bootstrap backbone comparison;
- paired bootstrap multi-window vs brain-only ablation.

### 10. Manuscript Preparation

Completed:

- manuscript draft;
- manuscript draft v2;
- figure plan;
- Q1 submission gap checklist;
- statistical and ablation summary.

---

## Relevant Literature and Positioning

This project is positioned within four literature categories:

1. public ICH CT datasets and challenge benchmarks;
2. deep learning for ICH detection;
3. CNN backbone architecture research;
4. explainability and clinical triage research.

### Literature Summary Table

| Topic | Reference | Relevance to This Project |
|---|---|---|
| RSNA ICH dataset | RSNA Intracranial Hemorrhage Detection Challenge | Provides the main dataset and subtype labels |
| ICH challenge winner | Wang et al., 2021 | Strong reference for high-performing RSNA-based ICH detection |
| ICH DL review | Cortés-Ferre et al., 2023 | Supports broader DL relevance for ICH classification |
| ICH DL meta-analysis | Karamian et al., 2025 | Supports diagnostic promise of DL for NCCT ICH detection |
| Brain CT triage simulation | Lee et al., 2022 | Relevant to workflow and emergency triage framing |
| Clinical simulation of DL ICH detection | Choi et al., 2024 | Relevant to clinical decision-making simulation |
| EfficientNet | Tan and Le, 2019 | Lightweight scaling-efficient CNN baseline |
| DenseNet | Huang et al., 2017 | Dense feature-reuse CNN comparator |
| ConvNeXt | Liu et al., 2022 | Modern CNN backbone selected as primary model |
| Grad-CAM | Selvaraju et al., 2017 | Explainability method used for qualitative analysis |

### Comparative Literature Context

Important caution:

> The results in this repository are not directly comparable to all published papers because datasets, splits, training scale, patient-level aggregation, preprocessing, evaluation units, and external validation protocols may differ.

However, relevant positioning is still useful.

| Study / Source | Dataset / Evaluation | Reported Focus | Relation to This Project |
|---|---|---|---|
| RSNA Challenge | >25,000 cranial CT exams shared for acute ICH and subtype detection | Public challenge benchmark | This project uses the RSNA challenge data |
| Wang et al., 2021 | 2019 RSNA Brain CT Hemorrhage Challenge dataset | Automatic detection and classification of acute ICH and five subtypes | Shows high-performing challenge-level ICH detection is feasible |
| Lee et al., 2022 | Brain CT triage system evaluated in ED workflow simulation and external screening cohorts | Emergency triage workflow | Supports triage framing beyond pure classification |
| Choi et al., 2024 | DL-based ICH detection algorithm in simulated clinical environment | Clinical decision support simulation | Supports workflow-impact motivation |
| This project | 5,000-image natural-prevalence holdout | Slice-level triage, backbone comparison, bootstrap CI, ablation | Focuses on reproducible experimental rigor and Q1-style reporting |

---

## Dataset and Experimental Splits

The RSNA Intracranial Hemorrhage Detection challenge dataset was used.

### Label Columns

| Label | Description |
|---|---|
| `any` | Any intracranial hemorrhage |
| `epidural` | Epidural hemorrhage |
| `intraparenchymal` | Intraparenchymal hemorrhage |
| `intraventricular` | Intraventricular hemorrhage |
| `subarachnoid` | Subarachnoid hemorrhage |
| `subdural` | Subdural hemorrhage |

### Processed Label Summary

| Item | Count |
|---|---:|
| Unique processed images | 752,803 |
| Any hemorrhage positive | 107,933 |
| Approximate natural any prevalence | 14.34% |

### Experimental Splits

| Split | Size | Distribution | Purpose |
|---|---:|---|---|
| Smoke test | Small sample | Sanity check | Pipeline validation |
| Training subset 2k | 2,000 | 1,000 normal / 1,000 positive | Initial baseline |
| Training subset 10k | 10,000 | 5,000 normal / 5,000 positive | Main training |
| Natural holdout | 5,000 | 4,299 normal / 701 positive | Main evaluation |

---

## Methodological Pipeline

```text
RSNA DICOM Image
    ↓
Read DICOM metadata and pixel array
    ↓
Convert raw pixels to Hounsfield Units
    ↓
Apply CT windows
    ↓
Construct 3-channel tensor
    ↓
Resize to 224 × 224
    ↓
CNN backbone
    ↓
Six-label sigmoid output
    ↓
Any-hemorrhage triage probability
    ↓
Threshold selection and workload simulation
```

### Multi-Window Input

```text
Channel 1: brain window
Channel 2: subdural window
Channel 3: bone window
```

### Brain-Only Ablation

```text
Channel 1: brain window
Channel 2: brain window
Channel 3: brain window
```

---

## Mathematical Formulation

### 1. Hounsfield Unit Conversion

For raw DICOM pixel \(x\):

```math
HU(x) = s x + b
```

where:

- \(s\) is `RescaleSlope`;
- \(b\) is `RescaleIntercept`.

### 2. CT Windowing

For window center \(c\) and width \(w\):

```math
L = c - rac{w}{2}
```

```math
U = c + rac{w}{2}
```

```math
I_{c,w}(x) =
rac{\min(\max(HU(x), L), U) - L}{U-L}
```

### 3. Multi-Window Tensor

```math
X =
[I_{40,80}, I_{80,200}, I_{600,2800}]
```

where:

- \(I_{40,80}\) is the brain window;
- \(I_{80,200}\) is the subdural window;
- \(I_{600,2800}\) is the bone window.

### 4. Model Prediction

A CNN model \(f_{	heta}\) maps the input tensor \(X\) to six logits:

```math
z = f_{	heta}(X), \quad z \in \mathbb{R}^{6}
```

The probability for label \(k\) is:

```math
\hat{p}_k = \sigma(z_k) = rac{1}{1 + e^{-z_k}}
```

### 5. Multi-Label Loss

Weighted binary cross-entropy:

```math
\mathcal{L}
=
-rac{1}{K}
\sum_{k=1}^{K}
\left[
w_k y_k \log(\hat{p}_k)
+
(1-y_k)\log(1-\hat{p}_k)

ight]
```

where:

- \(K=6\);
- \(w_k\) is the positive class weight;
- \(y_k\in\{0,1\}\);
- \(\hat{p}_k\in[0,1]\).

### 6. Triage Decision Rule

For any-hemorrhage triage:

```math
\hat{y}_{any} =
egin{cases}
1, & \hat{p}_{any} \geq 	au \
0, & \hat{p}_{any} < 	au
\end{cases}
```

For ConvNeXt-Tiny high-sensitivity triage:

```math
    au = 0.10
```

### 7. Triage Metrics

Recall:

```math
Recall = rac{TP}{TP + FN}
```

Precision:

```math
Precision = rac{TP}{TP + FP}
```

False negative rate:

```math
FNR = rac{FN}{TP + FN}
```

False positive rate:

```math
FPR = rac{FP}{FP + TN}
```

Prioritized workload:

```math
W = rac{TP + FP}{N}
```

### 8. Brier Score

```math
Brier = rac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2
```

### 9. Paired Bootstrap Difference

For models \(A\) and \(B\):

```math
\Delta_M = M_A - M_B
```

For bootstrap sample \(b\):

```math
\Delta_M^{(b)} = M_A^{(b)} - M_B^{(b)}
```

95% percentile interval:

```math
CI_{95\%}
=
[
Q_{2.5\%}(\Delta_M^{(b)}),
Q_{97.5\%}(\Delta_M^{(b)})
]
```

---

## Model Backbones

### EfficientNet-B0

Used as a lightweight CNN baseline. EfficientNet is based on compound scaling of depth, width, and resolution.

### DenseNet121

Used as a strong classic medical imaging comparator. DenseNet uses dense connectivity to improve feature propagation and encourage feature reuse.

### ConvNeXt-Tiny

Used as the modern CNN backbone. ConvNeXt modernizes convolutional networks using design principles inspired by contemporary vision architectures while remaining a pure ConvNet family.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Image size | 224 × 224 |
| Training subset | 10,000 images |
| Training distribution | 5,000 normal / 5,000 positive |
| Validation split | 20% |
| Batch size | 32 |
| Epochs | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Class weighting | Positive class weighting |
| Mixed precision | Enabled |
| GPU | NVIDIA GeForce RTX 5060 |

---

## Evaluation Protocol

Evaluation used the same 5,000-image natural-prevalence holdout for all models.

| Holdout Statistic | Value |
|---|---:|
| Total images | 5,000 |
| Normal | 4,299 |
| Any hemorrhage positive | 701 |
| Any prevalence | 14.02% |
| Epidural positives | 17 |
| Intraparenchymal positives | 254 |
| Intraventricular positives | 160 |
| Subarachnoid positives | 234 |
| Subdural positives | 301 |

---

## Primary Quantitative Results

### Internal Validation AUC

| Model | Best Validation Any AUC |
|---|---:|
| EfficientNet-B0 | 0.904689 |
| DenseNet121 | 0.925171 |
| ConvNeXt-Tiny | 0.930487 |
| ConvNeXt-Tiny brain-only | 0.933614 |

### Natural Holdout Any-Hemorrhage Performance

| Model | Any AUC | Any AP | Best F1 |
|---|---:|---:|---:|
| EfficientNet-B0 | 0.916446 | 0.709731 | 0.644090 |
| DenseNet121 | 0.936476 | 0.781555 | 0.708696 |
| ConvNeXt-Tiny | **0.945789** | **0.816036** | 0.742898 |
| ConvNeXt-Tiny brain-only | 0.947194 | 0.815377 | **0.743767** |

Interpretation:

- ConvNeXt-Tiny was the strongest among the three primary multi-window backbones.
- Brain-only ablation was very competitive and slightly exceeded multi-window AUC, but paired bootstrap showed no meaningful statistical difference.

---

## Best Model: ConvNeXt-Tiny

### ConvNeXt-Tiny Holdout Summary

| Metric | Value |
|---|---:|
| AUC | 0.945789 |
| AUC 95% CI | 0.936999 – 0.953849 |
| AP | 0.816036 |
| AP 95% CI | 0.792165 – 0.839746 |
| Brier score | 0.063986 |
| Best F1 threshold | 0.65 |
| Best F1 | 0.742898 |
| High-sensitivity threshold | 0.10 |
| Recall at 0.10 | 0.931526 |
| Precision at 0.10 | 0.402589 |
| Prioritized workload at 0.10 | 32.44% |

---

## Per-Label Performance

### ConvNeXt-Tiny Per-Label Results

| Label | Positive Count | Prevalence | AUC | AP |
|---|---:|---:|---:|---:|
| any | 701 | 0.1402 | 0.945789 | 0.816036 |
| epidural | 17 | 0.0034 | 0.938839 | 0.056214 |
| intraparenchymal | 254 | 0.0508 | 0.970937 | 0.770624 |
| intraventricular | 160 | 0.0320 | 0.990244 | 0.847204 |
| subarachnoid | 234 | 0.0468 | 0.926740 | 0.498531 |
| subdural | 301 | 0.0602 | 0.914619 | 0.597648 |

Key interpretation:

- Intraventricular hemorrhage achieved the highest AUC and AP.
- Epidural AUC was high, but AP was low due to only 17 positive holdout cases.
- Rare subtype evaluation should be interpreted cautiously.

---

## Backbone Comparison

### Main Comparison

| Model | Any AUC | Any AP | Recall ≥90% Threshold | Recall | Precision | Workload |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.916446 | 0.709731 | 0.25 | 0.908702 | 0.359076 | 35.48% |
| DenseNet121 | 0.936476 | 0.781555 | 0.35 | 0.910128 | 0.373974 | 34.12% |
| ConvNeXt-Tiny | **0.945789** | **0.816036** | 0.10 | **0.931526** | **0.402589** | **32.44%** |

ConvNeXt-Tiny achieved:

- highest AUC;
- highest AP;
- highest high-sensitivity recall;
- highest precision at the selected ≥90% recall operating point;
- lowest workload among the three main backbones.

---

## Paired Bootstrap Statistical Comparison

All three backbone models were evaluated on the exact same holdout images. This allows paired bootstrap comparison.

### ConvNeXt-Tiny vs DenseNet121

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.009313 | 0.001840 – 0.016482 | 0.009 |
| AP | +0.034482 | 0.017094 – 0.051272 | 0.000 |

### ConvNeXt-Tiny vs EfficientNet-B0

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.029343 | 0.019851 – 0.039057 | 0.000 |
| AP | +0.106306 | 0.081276 – 0.132075 | 0.000 |

### DenseNet121 vs EfficientNet-B0

| Metric | Difference | 95% CI | Bootstrap p-value |
|---|---:|---:|---:|
| AUC | +0.020030 | 0.011076 – 0.029448 | 0.000 |
| AP | +0.071824 | 0.046238 – 0.097875 | 0.000 |

Interpretation:

> ConvNeXt-Tiny outperformed both DenseNet121 and EfficientNet-B0 with paired bootstrap confidence intervals that did not cross zero.

---

## CT Window Ablation Study

### Goal

The ablation tested whether the 3-channel multi-window input improves any-hemorrhage triage compared with using only the brain window.

### Ablation Configurations

| Configuration | Channel 1 | Channel 2 | Channel 3 |
|---|---|---|---|
| Multi-window | Brain | Subdural | Bone |
| Brain-only | Brain | Brain | Brain |

### Ablation Result

| Input | AUC | AP | Recall ≥90% Threshold | Recall | Precision | Workload |
|---|---:|---:|---:|---:|---:|---:|
| Multi-window | 0.945789 | 0.816036 | 0.10 | 0.931526 | 0.402589 | 32.44% |
| Brain-only | 0.947194 | 0.815377 | 0.60 | 0.904422 | 0.433356 | 29.26% |

### Paired Bootstrap Ablation

| Metric | Multi-window Minus Brain-only | Bootstrap p-value |
|---|---:|---:|
| AUC | -0.001405 | 0.646 |
| AP | +0.000659 | 0.929 |

Interpretation:

> Multi-window input did not show statistically meaningful improvement over brain-window-only input for slice-level any-hemorrhage triage.

This is a valuable ablation result because it prevents overclaiming.

---

## Triage Simulation

### ConvNeXt-Tiny Threshold 0.10

| Confusion Matrix Component | Count |
|---|---:|
| TP | 653 |
| FP | 969 |
| FN | 48 |
| TN | 3330 |

| Metric | Value |
|---|---:|
| Recall | 0.931526 |
| Recall 95% CI | 0.912981 – 0.950071 |
| Precision | 0.402589 |
| Precision 95% CI | 0.389417 – 0.417873 |
| False negative rate | 0.068474 |
| False positive rate | 0.225401 |
| Prioritized workload | 0.3244 |

Clinical-style interpretation:

> At threshold 0.10, ConvNeXt-Tiny captured 653 of 701 hemorrhage-positive images while prioritizing 32.44% of the workload.

---

## Calibration Analysis

ConvNeXt-Tiny calibration:

| Metric | Value |
|---|---:|
| Brier score | 0.063986 |

Generated calibration artifact:

```text
outputs/reports/calibration_curve_any_convnext_tiny_10000.csv
outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png
```

Interpretation:

- Ranking performance is strong.
- Calibration is measured but not yet optimized.
- Future work should evaluate temperature scaling or isotonic regression.

---

## Error Analysis

Error analysis was performed for the EfficientNet-B0 model at threshold 0.25 earlier in the project. The same structure can be extended to ConvNeXt-Tiny.

Existing error-analysis outputs:

```text
outputs/reports/error_analysis_summary_model10000_threshold025.json
outputs/reports/error_analysis_public_summary_threshold025.csv
outputs/reports/error_analysis_false_negative_subtype_summary_threshold025.csv
outputs/figures/error_analysis/
```

Error analysis includes:

- false-positive count;
- false-negative count;
- error-group probability distribution;
- false-negative subtype summary;
- confusion matrix.

Recommended next improvement:

> Regenerate the same error-analysis report specifically for ConvNeXt-Tiny at threshold 0.10.

---

## Qualitative Explainability

Grad-CAM qualitative analysis was generated locally.

Qualitative goals:

| Case Type | Purpose |
|---|---|
| High-confidence true positive | Verify model attends to plausible hemorrhage regions |
| False positive | Understand over-prioritization behavior |
| False negative | Study missed hemorrhage-positive cases |
| Low-score normal | Confirm normal cases receive low activation |

Generated local files:

```text
outputs/figures/gradcam_model10000/
outputs/figures/paper_ready/figure_gradcam_composite_model10000.png
outputs/figures/paper_ready/figure_gradcam_composite_model10000.pdf
outputs/figures/paper_ready/figure_gradcam_cases_table.csv
```

Important:

> These CT-derived Grad-CAM figures are not uploaded publicly because they may be considered patient-derived medical visualizations.

Public repository excludes these files.

---

## Generated Figures

### Backbone Comparison

```text
outputs/figures/model_comparison/backbone_any_auc_ap_comparison.png
outputs/figures/model_comparison/backbone_per_label_auc_comparison.png
outputs/figures/model_comparison/backbone_recall90_workload_comparison.png
outputs/figures/model_comparison/backbone_triage_recall_curve.png
```

### ConvNeXt-Tiny Paper Metrics

```text
outputs/figures/paper_metrics_convnext_tiny/roc_curve_any_convnext_tiny_10000.png
outputs/figures/paper_metrics_convnext_tiny/pr_curve_any_convnext_tiny_10000.png
outputs/figures/paper_metrics_convnext_tiny/calibration_curve_any_convnext_tiny_10000.png
outputs/figures/paper_metrics_convnext_tiny/per_label_auc_ap_convnext_tiny_10000.png
outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_triage_recall90_threshold_010.png
outputs/figures/paper_metrics_convnext_tiny/confusion_matrix_any_best_f1_threshold_065.png
```

### Statistical Comparison

```text
outputs/figures/statistical_comparison/paired_bootstrap_auc_ap_difference.png
```

### Ablation Study

```text
outputs/figures/ablation_study/paired_bootstrap_multiwindow_vs_brainonly.png
```

### Bootstrap CI

```text
outputs/figures/paper_metrics/bootstrap_ci_auc_ap_convnext_tiny_10000.png
```

---

## Generated Reports

```text
outputs/reports/backbone_comparison_holdout_5000.csv
outputs/reports/backbone_comparison_per_label_auc_ap.csv
outputs/reports/backbone_comparison_summary.md
outputs/reports/bootstrap_ci_convnext_tiny_10000.csv
outputs/reports/bootstrap_ci_convnext_tiny_10000.json
outputs/reports/bootstrap_threshold_ci_any_convnext_tiny_10000.csv
outputs/reports/paper_metrics_summary_convnext_tiny_10000.json
outputs/reports/paper_metrics_per_label_auc_ap_convnext_tiny_10000.csv
outputs/reports/calibration_curve_any_convnext_tiny_10000.csv
outputs/reports/paired_bootstrap_backbone_comparison.csv
outputs/reports/paired_bootstrap_backbone_comparison.json
outputs/reports/paired_bootstrap_ablation_multiwindow_vs_brainonly.csv
outputs/reports/paired_bootstrap_ablation_multiwindow_vs_brainonly.json
```

---

## Repository Structure

```text
ai-ct-hemorrhage-triage/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── datasets/
│   │   └── rsna_dataset.py
│   ├── models/
│   │   └── classifier.py
│   ├── preprocessing/
│   │   └── dicom_utils.py
│   ├── evaluation/
│   ├── training/
│   ├── visualization/
│   └── utils/
├── scripts/
│   ├── check_gpu.py
│   ├── check_imports.py
│   ├── test_ct_windowing.py
│   ├── prepare_rsna_labels.py
│   ├── inspect_rsna_zip.py
│   ├── extract_rsna_train_10000.py
│   ├── extract_rsna_holdout_5000.py
│   ├── train_baseline_10000.py
│   ├── train_densenet121_10000.py
│   ├── train_convnext_tiny_10000.py
│   ├── train_convnext_tiny_brain_only_10000.py
│   ├── evaluate_holdout_5000_model10000.py
│   ├── evaluate_holdout_5000_densenet121.py
│   ├── evaluate_holdout_5000_convnext_tiny.py
│   ├── evaluate_holdout_5000_convnext_tiny_brain_only.py
│   ├── generate_backbone_comparison_report.py
│   ├── generate_paper_metrics_figures_convnext_tiny.py
│   ├── bootstrap_confidence_intervals_convnext_tiny.py
│   ├── paired_bootstrap_backbone_comparison.py
│   ├── paired_bootstrap_ablation_multiwindow_vs_brainonly.py
│   ├── error_analysis_model10000.py
│   ├── generate_manuscript_draft.py
│   └── update_manuscript_with_stats_ablation.py
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

## Reproducibility Workflow

### 1. Activate Environment

```powershell
conda activate main
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download RSNA Dataset

```powershell
kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p data\raw\rsna
```

### 4. Prepare Labels

```powershell
python scripts\prepare_rsna_labels.py
```

### 5. Extract Training and Holdout DICOMs

```powershell
python scripts\extract_rsna_train_10000.py
python scripts\extract_rsna_holdout_5000.py
```

### 6. Train Backbones

```powershell
python scripts\train_baseline_10000.py
python scripts\train_densenet121_10000.py
python scripts\train_convnext_tiny_10000.py
python scripts\train_convnext_tiny_brain_only_10000.py
```

### 7. Evaluate Backbones

```powershell
python scripts\evaluate_holdout_5000_model10000.py
python scripts\evaluate_holdout_5000_densenet121.py
python scripts\evaluate_holdout_5000_convnext_tiny.py
python scripts\evaluate_holdout_5000_convnext_tiny_brain_only.py
```

### 8. Generate Reports

```powershell
python scripts\generate_backbone_comparison_report.py
python scripts\generate_paper_metrics_figures_convnext_tiny.py
python scripts\bootstrap_confidence_intervals_convnext_tiny.py
python scripts\paired_bootstrap_backbone_comparison.py
python scripts\paired_bootstrap_ablation_multiwindow_vs_brainonly.py
```

---

## GitHub Data Governance

### Uploaded to GitHub

Allowed:

- source code;
- scripts;
- aggregate metrics;
- aggregate CSV reports;
- aggregate figures;
- manuscript files;
- README;
- requirements;
- `.gitignore`.

### Not Uploaded to GitHub

Excluded:

- `data/`
- raw DICOM files;
- RSNA zip archive;
- trained checkpoints;
- prediction CSV files with image identifiers;
- patient-derived CT visualizations;
- Grad-CAM overlays;
- large medical image artifacts.

This is intentional for ethical and data governance reasons.

---

## Q1-Readiness Assessment

### Already Strong

| Item | Status |
|---|---|
| Reproducible code | Strong |
| Natural-prevalence holdout | Strong |
| Multiple model backbones | Strong |
| Statistical comparison | Strong |
| Bootstrap confidence intervals | Strong |
| Ablation study | Strong |
| Calibration analysis | Present |
| Error analysis | Present |
| Explainability | Present locally |
| Manuscript draft | Present |
| GitHub-safe repository | Present |

### Still Needed

| Item | Current Status | Importance |
|---|---|---|
| External validation | Missing | Very high |
| Patient-level aggregation | Missing | Very high |
| Study-level evaluation | Missing | Very high |
| ConvNeXt-specific error analysis | Partially missing | Medium-high |
| Calibration improvement | Missing | Medium-high |
| Journal formatting | Missing | Medium |
| Formal ethics statement | Needs polishing | Medium |
| Radiologist review | Missing | High |

---

## Remaining Gaps Before Serious Q1 Submission

To improve the chance of Q1-level acceptance, the next steps should be:

1. **External validation**
   - Validate on a different institutional dataset.

2. **Patient-level aggregation**
   - Aggregate slice predictions to study-level probability.

3. **Calibration optimization**
   - Apply temperature scaling or isotonic regression.

4. **ConvNeXt-specific error analysis**
   - Generate false positive / false negative analysis at threshold 0.10.

5. **Expanded ablation**
   - pretrained vs non-pretrained;
   - full dataset vs 10k subset;
   - natural-prevalence training vs balanced training.

6. **Radiologist qualitative review**
   - Expert review of Grad-CAM examples.

7. **Manuscript polishing**
   - Journal target selection;
   - professional English editing;
   - reference formatting;
   - ethics/data availability statement.

---

## Limitations

1. **Slice-level evaluation**  
   The current model evaluates individual images/slices, not complete CT studies.

2. **Internal dataset holdout**  
   The holdout set is sampled from the RSNA dataset, not an external institution.

3. **No prospective validation**  
   Workflow improvement is simulated, not tested prospectively.

4. **Rare subtype instability**  
   Epidural hemorrhage has only 17 positive holdout examples.

5. **Calibration not optimized**  
   Brier score and calibration curve were reported, but calibration was not improved.

6. **Grad-CAM is qualitative**  
   Grad-CAM should not be treated as causal or definitive localization.

7. **Not a clinical product**  
   This model is a research prototype only.

---

## Future Work

Recommended next development sequence:

1. Generate ConvNeXt-specific error analysis.
2. Add patient-level aggregation.
3. Add external validation.
4. Add calibration optimization.
5. Add inference latency benchmarking.
6. Add study-level triage simulation.
7. Compare with transformer-based architectures.
8. Expand training to larger RSNA subsets.
9. Conduct radiologist review of model outputs.
10. Prepare journal-specific manuscript.

---

## Medical and Ethical Disclaimer

This repository is for **research and education only**.

The model:

- is not clinically validated;
- is not regulatory-approved;
- is not a medical device;
- must not be used for diagnosis;
- must not be used for treatment decisions;
- must not replace radiologist interpretation.

Any clinical use requires:

- institutional approval;
- ethical review;
- external validation;
- prospective evaluation;
- clinical safety assessment;
- regulatory review.

---

## Suggested Citation

```bibtex
@misc{rizal2026convnext_ich_triage,
  title={Explainable ConvNeXt-Based Triage of Intracranial Hemorrhage on Head CT: Natural-Prevalence Evaluation, Paired Statistical Backbone Comparison, and CT Window Ablation},
  author={Rizal},
  year={2026},
  note={Research prototype and reproducible GitHub implementation}
}
```

---

## References

1. Radiological Society of North America. **RSNA Intracranial Hemorrhage Detection Challenge**, 2019.  
   https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019

2. Kaggle. **RSNA Intracranial Hemorrhage Detection**.  
   https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

3. Tan M, Le QV. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. ICML, 2019.  
   https://arxiv.org/abs/1905.11946

4. Huang G, Liu Z, Van Der Maaten L, Weinberger KQ. **Densely Connected Convolutional Networks**. CVPR, 2017.  
   https://arxiv.org/abs/1608.06993

5. Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. **A ConvNet for the 2020s**. CVPR, 2022.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

6. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization**. ICCV, 2017.  
   https://arxiv.org/abs/1610.02391

7. Wang X et al. **A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans**. 2021.  
   https://www.sciencedirect.com/science/article/pii/S2213158221002291

8. Salehinejad H et al. **Intracranial Hemorrhage Detection on Head CT**. 2021.  
   https://arxiv.org/pdf/2102.04869

9. Cortés-Ferre L et al. **Deep Learning Applied to Intracranial Hemorrhage Detection**. 2023.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9963867/

10. Lee S et al. **Emergency triage of brain computed tomography via anomaly detection with a deep generative model**. Nature Communications, 2022.  
    https://www.nature.com/articles/s41467-022-31808-0

11. Choi SY et al. **Impact of a deep learning-based brain CT interpretation algorithm in a simulated clinical environment**. Scientific Reports, 2024.  
    https://www.nature.com/articles/s41598-024-73589-0

12. Karamian A et al. **Diagnostic Accuracy of Deep Learning for Intracranial Hemorrhage Detection on Non-Contrast CT: A Meta-Analysis**. 2025.  
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11989428/

---

## Final Research Statement

This repository is a complete research package for AI-assisted intracranial hemorrhage triage on head CT images. The current best main backbone, ConvNeXt-Tiny, achieved strong natural-prevalence holdout performance with statistically supported superiority over EfficientNet-B0 and DenseNet121.

The CT window ablation showed that brain-window-only input performs comparably to multi-window input for the slice-level any-hemorrhage endpoint, which is an important honest finding for manuscript discussion.

The project is not yet a clinically deployable system, but it is a strong Q1-oriented research foundation. The next critical steps are external validation, patient-level aggregation, calibration improvement, and expert clinical review.
