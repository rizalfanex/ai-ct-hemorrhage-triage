# Statistical and Ablation Summary

## Paired Backbone Comparison

| comparison                      | model_a       | model_b         | metric            |   model_a_value |   model_b_value |   difference |   difference_ci_low |   difference_ci_high |   bootstrap_p_value |   n_bootstrap |
|:--------------------------------|:--------------|:----------------|:------------------|----------------:|----------------:|-------------:|--------------------:|---------------------:|--------------------:|--------------:|
| ConvNeXt-Tiny - DenseNet121     | ConvNeXt-Tiny | DenseNet121     | auc               |        0.945789 |        0.936476 |   0.00931295 |          0.00183995 |            0.016482  |               0.009 |          2000 |
| ConvNeXt-Tiny - DenseNet121     | ConvNeXt-Tiny | DenseNet121     | average_precision |        0.816036 |        0.781555 |   0.0344816  |          0.0170937  |            0.0512718 |               0     |          2000 |
| ConvNeXt-Tiny - EfficientNet-B0 | ConvNeXt-Tiny | EfficientNet-B0 | auc               |        0.945789 |        0.916446 |   0.0293433  |          0.0198511  |            0.0390573 |               0     |          2000 |
| ConvNeXt-Tiny - EfficientNet-B0 | ConvNeXt-Tiny | EfficientNet-B0 | average_precision |        0.816036 |        0.709731 |   0.106306   |          0.0812763  |            0.132075  |               0     |          2000 |
| DenseNet121 - EfficientNet-B0   | DenseNet121   | EfficientNet-B0 | auc               |        0.936476 |        0.916446 |   0.0200304  |          0.0110756  |            0.0294479 |               0     |          2000 |
| DenseNet121 - EfficientNet-B0   | DenseNet121   | EfficientNet-B0 | average_precision |        0.781555 |        0.709731 |   0.0718239  |          0.0462379  |            0.0978753 |               0     |          2000 |

## CT Window Ablation

| comparison                    | metric            |   multi_window_value |   brain_only_value |   difference |   difference_ci_low |   difference_ci_high |   bootstrap_p_value |   n_bootstrap |
|:------------------------------|:------------------|---------------------:|-------------------:|-------------:|--------------------:|---------------------:|--------------------:|--------------:|
| multi_window_minus_brain_only | auc               |             0.945789 |           0.947194 | -0.00140496  |          -0.0074152 |           0.00506251 |               0.646 |          2000 |
| multi_window_minus_brain_only | average_precision |             0.816036 |           0.815377 |  0.000659217 |          -0.0131154 |           0.0146974  |               0.929 |          2000 |

## Main Interpretation

ConvNeXt-Tiny significantly outperformed DenseNet121 and EfficientNet-B0 under paired bootstrap comparison on the same natural-prevalence holdout set.

The CT window ablation did not show a statistically meaningful difference between multi-window and brain-window-only input. Brain-only was slightly higher in AUC, while multi-window was slightly higher in AP, but both differences were small and not statistically robust.

## Manuscript Implication

The paper should claim that ConvNeXt-Tiny is the strongest evaluated backbone. However, it should not claim that multi-window preprocessing clearly outperforms brain-only preprocessing for the any-hemorrhage endpoint.
