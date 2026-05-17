# Experiment Summary: RSNA Intracranial Hemorrhage Triage

## Model Comparison on Natural Holdout 5,000

| model                       |   train_size |   holdout_size |   holdout_any_prevalence |   any_auc |   any_average_precision |   best_f1_threshold |   best_f1 |   best_recall90_threshold |   best_recall90_precision |   best_recall90_recall |   best_recall90_predicted_positive_rate |
|:----------------------------|-------------:|---------------:|-------------------------:|----------:|------------------------:|--------------------:|----------:|--------------------------:|--------------------------:|-----------------------:|----------------------------------------:|
| EfficientNet-B0 train_2000  |         2000 |           5000 |                   0.1402 |  0.871442 |                0.567904 |                0.95 |  0.546362 |                      0.2  |                  0.278603 |               0.910128 |                                  0.458  |
| EfficientNet-B0 train_10000 |        10000 |           5000 |                   0.1402 |  0.916446 |                0.709731 |                0.9  |  0.64409  |                      0.25 |                  0.359076 |               0.908702 |                                  0.3548 |

## Model 10,000 Per-Label Metrics

| label            |   positive_count |   prevalence |      auc |   average_precision |
|:-----------------|-----------------:|-------------:|---------:|--------------------:|
| any              |              701 |       0.1402 | 0.916446 |           0.709731  |
| epidural         |               17 |       0.0034 | 0.749584 |           0.0743958 |
| intraparenchymal |              254 |       0.0508 | 0.91126  |           0.631626  |
| intraventricular |              160 |       0.032  | 0.943908 |           0.649052  |
| subarachnoid     |              234 |       0.0468 | 0.839241 |           0.282576  |
| subdural         |              301 |       0.0602 | 0.874984 |           0.450763  |

## Key Interpretation

- Increasing training data from 2,000 to 10,000 DICOM improved any-hemorrhage AUC on natural holdout.
- The 10,000-sample model achieved stronger triage ranking performance.
- At threshold 0.25, the 10,000 model reached recall above 0.90 with substantially lower priority workload than the 2,000 model.
- The model is promising for research-grade triage simulation, but it is not clinically validated.

## Generated Figures

- outputs\figures\experiment_summary\model_auc_ap_comparison.png
- outputs\figures\experiment_summary\subtype_auc_ap_model10000.png
- outputs\figures\experiment_summary\threshold_tradeoff_model10000.png
- outputs\figures\experiment_summary\triage_precision_curve.png
- outputs\figures\experiment_summary\triage_recall_curve.png