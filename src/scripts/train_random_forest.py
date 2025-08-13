#!/usr/bin/env python3
"""
train_random_forest.py

Train a Random Forest on antisymmetric delta features and calibrate probabilities.
- Keeps the same I/O contract as your logistic script.
- Saves feature order on the model (.feature_names_) for downstream plotting code.
- Optional antisymmetry augmentation (adds (-x, 1-y) for each (x, y) in train only).

Usage
-----
python src/scripts/train_random_forest.py \
  --pairs src/data/processed/train_pairs_SPall.train.csv \
  --model-out src/model/model_SPall_rf_isoC5.train.joblib \
  --calib isotonic --cv 5 \
  --n-estimators 600 --max-depth 8 --min-samples-leaf 20 \
  --augment-antisym
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_X(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("delta_")]
    return df[cols].values, cols

def maybe_antisym_augment(X: np.ndarray, y: np.ndarray, enable: bool):
    if not enable:
        return X, y
    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([y, 1 - y])
    return X_aug, y_aug

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Pairwise CSV from prepare_pairwise_data.py (train split)")
    ap.add_argument("--model-out", required=True, help="Path to write calibrated model (.joblib)")
    ap.add_argument("--calib", default="isotonic", choices=["isotonic", "sigmoid"], help="Calibration method (default isotonic)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for calibration (default 5)")

    # RF hyperparams (sane defaults for your dataset size)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-samples-leaf", type=int, default=20)
    ap.add_argument("--max-features", default="sqrt", help="e.g., 'sqrt', 'log2', or int/float")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)

    # Extras
    ap.add_argument("--augment-antisym", action="store_true",
                    help="Augment training with (-x, 1-y) to encourage antisymmetry")
    ap.add_argument("--scale", action="store_true",
                    help="Include StandardScaler before RF (harmless; off by default)")

    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    X, feature_names = build_X(df)
    y = df["label"].astype(int).values

    # Optional antisymmetry augmentation (train only)
    X_train, y_train = maybe_antisym_augment(X, y, args.augment_antisym)

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    # Pipeline (scaler optional; trees don't need it, but it doesn't hurt)
    steps = []
    if args.scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("clf", rf))

    base = Pipeline(steps)

    # Cross-validated probability calibration (like your logistic script)
    clf = CalibratedClassifierCV(base, method=args.calib, cv=args.cv)
    clf.fit(X_train, y_train)

    # Save feature order for plotting scripts
    clf.feature_names_ = feature_names

    outdir = os.path.dirname(args.model_out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    joblib.dump(clf, args.model_out)
    print(f"✔ Saved calibrated Random Forest to {args.model_out}")
    print(f"   (n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"min_samples_leaf={args.min_samples_leaf}, max_features={args.max_features}, "
          f"calibration={args.calib}, cv={args.cv}, antisym_aug={args.augment_antisym})")

if __name__ == "__main__":
    main()
