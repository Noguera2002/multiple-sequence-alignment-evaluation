#!/usr/bin/env python3
"""
Train elastic-net logistic on pairwise delta features with interaction terms,
then perform cross-validated calibration to get well-calibrated probabilities.

Outputs a calibrated classifier (.joblib).
"""
import pandas as pd
import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_features(df, add_interactions=True):
    # assume columns: tool1, tool2, label, and delta_<metric> features
    feature_cols = [c for c in df.columns if c.startswith('delta_')]
    X = df[feature_cols].copy()
    if add_interactions:
        # pairwise products (including squares) of features
        for i in range(len(feature_cols)):
            for j in range(i, len(feature_cols)):
                a = feature_cols[i]
                b = feature_cols[j]
                X[f'{a}_x_{b}'] = X[a] * X[b]
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-pairs', required=True, help='Train pairwise CSV (used for both fitting and calibration)')
    parser.add_argument('--model-out', required=True, help='Output calibrated model (.joblib)')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV folds for inner grid and calibration')
    parser.add_argument('--method', choices=['isotonic','sigmoid'], default='isotonic', help='Calibration method')
    parser.add_argument('--no-interactions', action='store_true', help='Do not add interaction terms / squares')
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_pairs)
    y = df_train['label']
    X_raw = build_features(df_train, add_interactions=not args.no_interactions)

    # Standardize features to help elastic-net
    scaler = StandardScaler()

    # Base logistic with elastic-net (saga supports elasticnet)
    base = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=5000,
        l1_ratio=0.5,  # placeholder; overridden by grid
        C=1.0,
        tol=1e-4,
        random_state=42,
    )

    pipe = Pipeline([
        ('scale', scaler),
        ('logreg', base),
    ])

    # Grid search over C and l1_ratio
    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__l1_ratio': [0.1, 0.5, 0.9],
    }
    cv_inner = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv_inner,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_raw, y)
    best_pipe = grid.best_estimator_
    print(f"Best CV params: {grid.best_params_}, avg_precision={grid.best_score_:.4f}")

    # Calibrate with cross-validated calibration on same data
    calibrated = CalibratedClassifierCV(best_pipe, cv=args.cv_folds, method=args.method)

    calibrated.fit(X_raw, y)

    joblib.dump(calibrated, args.model_out)
    print(f"Calibrated model saved to {args.model_out}")

if __name__ == '__main__':
    main()
