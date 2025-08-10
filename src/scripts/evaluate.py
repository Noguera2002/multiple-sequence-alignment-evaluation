#!/usr/bin/env python3
"""
Evaluate confident pairwise predictions at a given confidence threshold.

Builds ONLY the features the model expects (from model.feature_names_):
- base delta_* features
- interaction features a_x_b only if they appear in feature_names_

Reports coverage and precision where |p - 0.5| >= conf_thresh.
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from itertools import combinations_with_replacement

def confident_stats(y, probs, thresh=0.3):
    mask = np.abs(probs - 0.5) >= thresh
    pred = (probs > 0.5).astype(int)
    correct = (pred == y)
    coverage = mask.mean() if len(y) else 0.0
    precision = (correct & mask).sum() / max(mask.sum(), 1)
    return coverage, precision, int(mask.sum())

def build_X_matching_model(df_pairs, feature_names):
    """Return X as numpy array with columns exactly = feature_names."""
    # base deltas present in pairs file
    base_cols = [c for c in df_pairs.columns if c.startswith("delta_")]
    base = df_pairs[base_cols].copy()

    # precompute a dict for fast lookup
    base_dict = {c: base[c].to_numpy() for c in base_cols}

    # if interactions are expected, compute only those that appear in feature_names_
    # interaction naming: "<a>_x_<b>" where a,b are base delta col names
    X_cols = []
    for name in feature_names:
        if name in base_dict:
            X_cols.append(base_dict[name])
        elif "_x_" in name:
            a, b = name.split("_x_")
            xa = base_dict.get(a)
            xb = base_dict.get(b)
            if xa is None or xb is None:
                # missing base -> fill zeros
                X_cols.append(np.zeros(len(df_pairs), dtype=float))
            else:
                X_cols.append(xa * xb)
        else:
            # unknown feature -> zeros
            X_cols.append(np.zeros(len(df_pairs), dtype=float))

    return np.column_stack(X_cols)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Calibrated classifier (.joblib)')
    p.add_argument('--test-pairs', required=True, help='CSV with label and delta_* features')
    p.add_argument('--conf-thresh', type=float, default=0.3,
                   help='Confidence threshold: |p-0.5| >= conf-thresh')
    args = p.parse_args()

    df = pd.read_csv(args.test_pairs)
    y = df['label'].astype(int).to_numpy()

    clf = joblib.load(args.model)
    feature_names = getattr(clf, "feature_names_", None)
    if feature_names is None:
        # Fallback: use whatever delta_* columns exist (sorted) — not ideal, but works
        feature_names = sorted([c for c in df.columns if c.startswith("delta_")])

    X = build_X_matching_model(df, feature_names)  # numpy array, ordered

    probs = clf.predict_proba(X)[:, 1]
    total_pairs = len(df)
    coverage, precision, n_conf = confident_stats(y, probs, thresh=args.conf_thresh)

    print(f"Total pairs         : {total_pairs}")
    print(f"Confident pairs     : {n_conf} ({coverage:.1%} coverage)")
    target_conf = 0.5 + args.conf_thresh
    print(f"Precision@{int(target_conf*100)}%+ confidence: {precision:.3f}")

if __name__ == '__main__':
    main()
