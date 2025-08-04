#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

def build_features(df, add_interactions=True):
    feature_cols = [c for c in df.columns if c.startswith('delta_')]
    X = df[feature_cols].copy()
    if add_interactions:
        for i in range(len(feature_cols)):
            for j in range(i, len(feature_cols)):
                a = feature_cols[i]
                b = feature_cols[j]
                X[f'{a}_x_{b}'] = X[a] * X[b]
    return X

def precision_coverage(probs, y, thresholds):
    precisions = []
    coverages = []
    for t in thresholds:
        sel = probs >= t
        if sel.sum() == 0:
            precisions.append(np.nan)
            coverages.append(0.0)
        else:
            precisions.append((y[sel] == 1).sum() / sel.sum())
            coverages.append(sel.sum() / len(y))
    return np.array(precisions), np.array(coverages)

def plot_all(model_path, test_pairs, out_prefix, no_interactions=False):
    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(test_pairs)
    if 'label' not in df.columns:
        raise ValueError("Expected 'label' column in test pairs CSV")
    y = df['label'].to_numpy()

    # Build features the same way as in training
    X_feat = build_features(df, add_interactions=not no_interactions)

    # Predict probabilities
    probs = model.predict_proba(X_feat)[:, 1]

    # 1. Calibration curve
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy='uniform')
    plt.figure()
    plt.plot(mean_pred, frac_pos, 'o-', label='Empirical')
    plt.plot([0,1], [0,1], '--', label='Perfect') 
    plt.xlabel('Predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration / Reliability Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname1 = f"{out_prefix}_calibration_curve.png"
    plt.savefig(fname1)
    print(f"Saved {fname1}")
    plt.close()

    # 2. Precision vs coverage
    thresholds = np.linspace(0,1,101)
    precisions, coverages = precision_coverage(probs, y, thresholds)
    plt.figure()
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, coverages, label='Coverage')
    plt.xlabel('Confidence threshold')
    plt.ylabel('Value')
    plt.title('Precision and Coverage vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname2 = f"{out_prefix}_precision_coverage.png"
    plt.savefig(fname2)
    print(f"Saved {fname2}")
    plt.close()

    # 3. Histogram of predicted confidences
    plt.figure()
    plt.hist(probs, bins=30, edgecolor='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Confidences')
    plt.tight_layout()
    fname3 = f"{out_prefix}_confidence_histogram.png"
    plt.savefig(fname3)
    print(f"Saved {fname3}")
    plt.close()

    # 4. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, probs)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title('Precision-Recall Curve')
    fig.tight_layout()
    fname4 = f"{out_prefix}_pr_curve.png"
    fig.savefig(fname4)
    print(f"Saved {fname4}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Calibrated model joblib file')
    parser.add_argument('--test-pairs', required=True, help='CSV of test pairwise data with "label" column')
    parser.add_argument('--out-prefix', default='results/eval', help='Prefix for output PNG files')
    parser.add_argument('--no-interactions', action='store_true', help='If set, do not include interaction terms when building features')
    args = parser.parse_args()

    plot_all(args.model, args.test_pairs, args.out_prefix, no_interactions=args.no_interactions)

if __name__ == '__main__':
    main()
