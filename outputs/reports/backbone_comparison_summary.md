# Backbone Comparison Summary

## Natural Holdout Performance

| model           |   train_size |   holdout_size |   holdout_prevalence_any |    parameters |   training_time_seconds |   validation_best_any_auc |   holdout_any_auc |   holdout_any_average_precision |   best_f1_threshold |   best_f1 |   recall90_threshold |   recall90_precision |   recall90_recall |   recall90_false_negative_rate |   recall90_predicted_positive_rate |
|:----------------|-------------:|---------------:|-------------------------:|--------------:|------------------------:|--------------------------:|------------------:|--------------------------------:|--------------------:|----------:|---------------------:|---------------------:|------------------:|-------------------------------:|-----------------------------------:|
| EfficientNet-B0 |        10000 |           5000 |                   0.1402 | nan           |                 1144.32 |                  0.904689 |          0.916446 |                        0.709731 |                0.9  |  0.64409  |                 0.25 |             0.359076 |          0.908702 |                      0.0912981 |                             0.3548 |
| DenseNet121     |        10000 |           5000 |                   0.1402 |   6.96001e+06 |                 1163.08 |                  0.925171 |          0.936476 |                        0.781555 |                0.85 |  0.708696 |                 0.35 |             0.373974 |          0.910128 |                      0.0898716 |                             0.3412 |

## Per-Label AUC/AP

| model           | label            |   positive_count |   prevalence |      auc |   average_precision |
|:----------------|:-----------------|-----------------:|-------------:|---------:|--------------------:|
| EfficientNet-B0 | any              |              701 |       0.1402 | 0.916446 |           0.709731  |
| EfficientNet-B0 | epidural         |               17 |       0.0034 | 0.749584 |           0.0743958 |
| EfficientNet-B0 | intraparenchymal |              254 |       0.0508 | 0.91126  |           0.631626  |
| EfficientNet-B0 | intraventricular |              160 |       0.032  | 0.943908 |           0.649052  |
| EfficientNet-B0 | subarachnoid     |              234 |       0.0468 | 0.839241 |           0.282576  |
| EfficientNet-B0 | subdural         |              301 |       0.0602 | 0.874984 |           0.450763  |
| DenseNet121     | any              |              701 |       0.1402 | 0.936476 |           0.781555  |
| DenseNet121     | epidural         |               17 |       0.0034 | 0.884265 |           0.0894539 |
| DenseNet121     | intraparenchymal |              254 |       0.0508 | 0.965289 |           0.774511  |
| DenseNet121     | intraventricular |              160 |       0.032  | 0.983364 |           0.773168  |
| DenseNet121     | subarachnoid     |              234 |       0.0468 | 0.927029 |           0.498294  |
| DenseNet121     | subdural         |              301 |       0.0602 | 0.895069 |           0.479882  |

## Interpretation

- DenseNet121 outperformed EfficientNet-B0 on the natural-prevalence holdout set.
- DenseNet121 improved any-hemorrhage AUC and average precision.
- DenseNet121 required lower prioritized workload to achieve approximately 90% recall.
- This supports including DenseNet121 as the current best-performing backbone in the manuscript.

## Generated Figures

- outputs\figures\model_comparison\backbone_any_auc_ap_comparison.png
- outputs\figures\model_comparison\backbone_per_label_auc_comparison.png
- outputs\figures\model_comparison\backbone_recall90_workload_comparison.png
- outputs\figures\model_comparison\backbone_triage_recall_curve.png