# Q1 Submission Gap Checklist

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
