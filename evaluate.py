#!/usr/bin/env python3
"""
Evaluate confident pairwise predictions at a given confidence threshold.
Builds the same expanded features (delta_ and their pairwise products)
that the model was trained on, then reports coverage and precision for
calls where |p - 0.5| >= conf_thresh (i.e., high-confidence).
"""
import argparse
import pandas as pd
import joblib
import numpy as np

def build_features(df, add_interactions=True):
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].copy()
    if add_interactions:
        for i in range(len(feature_cols)):
            for j in range(i, len(feature_cols)):
                a = feature_cols[i]
                b = feature_cols[j]
                X[f"{a}_x_{b}"] = X[a] * X[b]
    return X

def confident_stats(y, probs, thresh=0.3):
    mask = np.abs(probs - 0.5) >= thresh
    pred = (probs > 0.5).astype(int)
    correct = ((pred == 1) & (y == 1)) | ((pred == 0) & (y == 0))
    coverage = mask.sum() / len(y) if len(y) > 0 else 0.0
    precision = (correct & mask).sum() / (mask.sum() if mask.sum() > 0 else 1)
    return coverage, precision, mask.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Calibrated classifier (.joblib)')
    parser.add_argument('--test-pairs', required=True, help='CSV of test pairs with columns tool1,tool2,label and delta_ features')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Confidence threshold: |p - 0.5| >= conf-thresh implies >=(0.5+conf-thresh) confidence')
    parser.add_argument('--no-interactions', action='store_true',
                        help='If the model was trained without interaction terms, disable interaction expansion here')
    args = parser.parse_args()

    # Load data and model
    df = pd.read_csv(args.test_pairs)
    clf = joblib.load(args.model)

    # Build features consistent with training
    X = build_features(df, add_interactions=not args.no_interactions)
    y = df['label']

    # Predict probability P(tool1 > tool2)
    probs = clf.predict_proba(X)[:, 1]
    df['p_tool1_gt_tool2'] = probs
    df['confident'] = (np.abs(probs - 0.5) >= args.conf_thresh)

    # Compute stats
    total_pairs = len(df)
    coverage, precision, n_confident = confident_stats(y, probs, thresh=args.conf_thresh)

    print(f"Total pairs         : {total_pairs}")
    print(f"Confident pairs     : {n_confident} ({coverage:.1%} coverage)")
    target_conf = 0.5 + args.conf_thresh
    print(f"Precision@{int(target_conf*100)}%+ confidence: {precision:.3f}")

    # Optionally save confident calls
    # df[df['confident']].to_csv('confident_calls.csv', index=False)

if __name__ == '__main__':
    main()
