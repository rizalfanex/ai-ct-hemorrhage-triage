# Figure Plan

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
