#!/usr/bin/env python3
"""
train_and_calibrate.py (antisymmetric, no interaction terms)

Train a logistic regression on antisymmetric delta features.
- Uses ONLY linear delta_* features (no squares/products)
- Enforces antisymmetry by setting fit_intercept=False
- Calibrates probabilities with cross-validated isotonic (default) or sigmoid

Outputs: calibrated model (.joblib)
"""
import argparse
import os  # NEW
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_X(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("delta_")]
    return df[cols].values, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Pairwise CSV from prepare_pairwise_data.py")
    ap.add_argument("--model-out", required=True, help="Path to write calibrated model (.joblib)")
    ap.add_argument("--penalty", default="l2", choices=["l2","none"], help="Logistic penalty (default l2)")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default 1.0)")
    ap.add_argument("--calib", default="isotonic", choices=["isotonic","sigmoid"], help="Calibration method")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for calibration")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    X, feature_names = build_X(df)   # CHANGED: keep feature_names
    y = df["label"].astype(int).values

    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty=None if args.penalty=="none" else args.penalty,
            C=args.C,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            fit_intercept=False
        ))
    ])

    clf = CalibratedClassifierCV(base, method=args.calib, cv=args.cv)
    clf.fit(X, y)

    # NEW: save feature order and ensure output dir exists
    clf.feature_names_ = feature_names
    outdir = os.path.dirname(args.model_out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    joblib.dump(clf, args.model_out)
    print(f"✔ Saved calibrated model to {args.model_out}")

if __name__ == "__main__":
    main()
