# AI CT Hemorrhage Triage

Research prototype for deep learning-based intracranial hemorrhage triage on head CT images using the RSNA Intracranial Hemorrhage Detection dataset.

## Objective

This project explores whether a lightweight convolutional neural network can prioritize head CT slices with possible intracranial hemorrhage for radiology triage.

The system is designed as a research prototype, not as a clinical diagnostic product.

## Core Idea

Instead of replacing radiologists, the model ranks CT cases by predicted hemorrhage risk so that potentially urgent scans can be reviewed earlier.

## Current Best Result

Best model:

- Architecture: EfficientNet-B0
- Training sample: 10,000 DICOM images
- Training distribution: 5,000 normal / 5,000 hemorrhage-positive
- Natural holdout: 5,000 DICOM images
- Natural holdout hemorrhage prevalence: 0.1402

Performance on natural holdout:

| Metric | Value |
|---|---:|
| Any hemorrhage AUC | 0.916446 |
| Any hemorrhage Average Precision | 0.709731 |
| Best F1 threshold | 0.90 |
| Best F1 | 0.644090 |
| Recall >= 0.90 threshold | 0.25 |
| Recall at that threshold | 0.908702 |
| Precision at that threshold | 0.359076 |
| Prioritized workload at that threshold | 0.354800 |

## Per-Label Performance

| label | positive_count | prevalence | AUC | Average Precision |
|---|---:|---:|---:|---:|
| any | 701 | 0.1402 | 0.916446 | 0.709731 |
| epidural | 17 | 0.0034 | 0.749584 | 0.074396 |
| intraparenchymal | 254 | 0.0508 | 0.911260 | 0.631626 |
| intraventricular | 160 | 0.0320 | 0.943908 | 0.649052 |
| subarachnoid | 234 | 0.0468 | 0.839241 | 0.282576 |
| subdural | 301 | 0.0602 | 0.874984 | 0.450763 |

## Main Pipeline

1. Download RSNA Intracranial Hemorrhage Detection dataset.
2. Extract and prepare label CSV.
3. Convert labels from long format to wide multi-label format.
4. Read DICOM images.
5. Convert pixel values to Hounsfield Units.
6. Apply multi-window CT preprocessing:
   - brain window
   - subdural window
   - bone window
7. Train EfficientNet-B0 multi-label classifier.
8. Evaluate on natural-prevalence holdout.
9. Perform threshold sweep.
10. Run triage simulation.
11. Generate Grad-CAM examples for local analysis.

## Key Outputs

Reports:

- `outputs/reports/model_comparison_holdout_5000.csv`
- `outputs/reports/subtype_auc_ap_model10000.csv`
- `outputs/reports/experiment_summary.md`

Safe aggregate figures:

- `outputs/figures/experiment_summary/model_auc_ap_comparison.png`
- `outputs/figures/experiment_summary/triage_recall_curve.png`
- `outputs/figures/experiment_summary/triage_precision_curve.png`
- `outputs/figures/experiment_summary/threshold_tradeoff_model10000.png`
- `outputs/figures/experiment_summary/subtype_auc_ap_model10000.png`

## Project Structure

```text
ai-ct-hemorrhage-triage/
├── README.md
├── requirements.txt
├── src/
│   ├── datasets/
│   ├── models/
│   └── preprocessing/
├── scripts/
├── outputs/
│   ├── figures/
│   ├── logs/
│   ├── models/
│   └── reports/
└── paper/
```

## Important Disclaimer

This repository is for research and education only.

The model is not clinically validated, not regulatory-approved, and must not be used for patient diagnosis or medical decision-making.

Do not commit or redistribute DICOM files, patient-derived images, trained checkpoints, or Grad-CAM images unless dataset governance rules explicitly allow it.
