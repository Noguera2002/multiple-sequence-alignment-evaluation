# Results & Plotting

This document explains how to generate result summaries and plots for trained models.

## Single-model report
Generates:
- Calibration curve
- Probability histogram
- ROC / PR curves
- Precision/coverage sweeps
- `summary.md` with headline numbers

### For logistic regression model
python3 results/scripts/make_model_report.py \
  --model src/model/model_SPall_sigC3_cv5.train.joblib \
  --pairs src/data/processed/train_pairs_SPall.test.csv \
  --outdir results/LR_model_results/

### For random forest model
python3 results/scripts/make_model_report.py \
  --model src/model/model_SPall_rf_iso_cv5.train.joblib \
  --pairs src/data/processed/train_pairs_SPall.test.csv \
  --outdir results/RF_model_results/

## Comparing Logistic vs Random Forest
python3 results/scripts/compare_models.py \
  --lr-model src/model/model_SPall_sigC3_cv5.train.joblib \
  --rf-model src/model/model_SPall_rf_iso_cv5.train.joblib \
  --pairs src/data/processed/train_pairs_SPall.test.csv \
  --outdir results/

## Produces overlay plots for:
- Precision vs coverage
- Calibration curves

## Output folders
- results/LR_model_results/ — Logistic regression plots + summary
- results/RF_model_results/ — Random forest plots + summary
- results/ — Comparison plots when running compare_models.py
