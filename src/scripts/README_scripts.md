## README_scripts.md

# Scripts for MSA Evaluation and Model Pipeline

This document describes the **core scripts** for feature generation, model training, evaluation, and prediction.  
Data preparation scripts are covered in `README_data.md`.

---

## Core scripts

| Script | Purpose | Example usage |
|---|---|---|
| **src/scripts/split_pairs_by_family.py** | Split pairs **by family** into train/test (e.g., 80/20). | `python3 src/scripts/split_pairs_by_family.py --pairs src/data/processed/train_pairs_SPall.csv --out-train src/data/processed/train_pairs_SPall.train.csv --out-test src/data/processed/train_pairs_SPall.test.csv --seed 42 --test-frac 0.2` |
| **src/scripts/train_and_calibrate.py** | Train **logistic regression** (L2, no intercept) and **sigmoid-calibrate** with 5-fold CV. | `python3 src/scripts/train_and_calibrate.py --pairs src/data/processed/train_pairs_SPall.train.csv --model-out src/model/model_SPall_sigC3_cv5.train.joblib --penalty l2 --C 3.0 --calib sigmoid --cv 5` |
| **src/scripts/train_random_forest.py** | Train **Random Forest** and **isotonic-calibrate** with 5-fold CV. | `python3 src/scripts/train_random_forest.py --pairs src/data/processed/train_pairs_SPall.train.csv --model-out src/model/model_SPall_rf_iso_cv5.train.joblib --calib isotonic --cv 5 --n-estimators 600 --max-depth 8 --min-samples-leaf 20 --augment-antisym` |
| **src/scripts/evaluate.py** | Precision/coverage at a **confidence margin** (`--conf-thresh`). | `python3 src/scripts/evaluate.py --model src/model/model_SPall_sigC3_cv5.train.joblib --test-pairs src/data/processed/train_pairs_SPall.test.csv --conf-thresh 0.2` |
| **src/scripts/predict.py** | Compare **two MSAs**; output calibrated probability **P(MSA1 > MSA2)**. | `python3 src/scripts/predict.py path/to/MSA1.fasta path/to/MSA2.fasta --model src/model/model_SPall_sigC3_cv5.train.joblib` |


## Notes
- **Logistic regression** (`model_SPall_sigC3_cv5.train.joblib`) is the main reference model.
- **Random forest** (`model_SPall_rf_iso_cv5.train.joblib`) is included for comparison.
- For end users, only `predict.py` + a trained model is required.

