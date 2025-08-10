#!/usr/bin/env python3
"""
predict.py — antisymmetric, no interactions, no sklearn warnings.

Builds ONLY the linear delta_* features the model expects, in the exact
order saved as model.feature_names_. Passes a NumPy array to avoid
'X has feature names' warnings from scikit-learn.
"""

import argparse
import joblib
import numpy as np
from pathlib import Path

# reuse your metrics helpers
from metrics import compute_metrics, read_alignment

def compute_metrics_for_file(msa_file, tool_label="MSA", family_label="Query"):
    seqs = read_alignment(msa_file)
    return compute_metrics(seqs)  # returns a dict of numeric metrics

def build_X_from_two_msas(m1: dict, m2: dict, feature_names):
    """
    Build a single-row NumPy array with columns exactly feature_names.
    feature_names are like ['delta_Gap', 'delta_Match', ...]
    """
    # raw deltas for all numeric metrics present
    deltas = {}
    for k, v in m1.items():
        if isinstance(v, (int, float)) and k in m2:
            try:
                deltas[f"delta_{k}"] = float(m1[k]) - float(m2[k])
            except Exception:
                pass

    # assemble in trained order; fill missing with 0.0
    row = [deltas.get(name, 0.0) for name in feature_names]
    return np.array([row], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msa1", help="Path to first MSA (FASTA)")
    ap.add_argument("msa2", help="Path to second MSA (FASTA)")
    ap.add_argument("--model", default="src/model/model_SPall_sigC3.joblib",
                    help="Path to trained calibrated model (.joblib)")
    args = ap.parse_args()

    # Load model & feature order
    model = joblib.load(args.model)
    feature_names = getattr(model, "feature_names_", None)
    if not feature_names:
        raise SystemExit("Model missing feature_names_. Retrain with updated train_and_calibrate.py")

    # Compute metrics for both MSAs
    m1 = compute_metrics_for_file(args.msa1, tool_label="MSA1")
    m2 = compute_metrics_for_file(args.msa2, tool_label="MSA2")

    # Build NumPy X in the exact expected order
    X = build_X_from_two_msas(m1, m2, feature_names)

    # Predict P(MSA1 > MSA2)
    prob = float(model.predict_proba(X)[:, 1][0])

    print(f"P(MSA1 > MSA2) = {prob:.3f}")
    print("Interpretation:")
    print("- ~1.0: strong support for MSA1")
    print("- ~0.5: no clear preference")
    print("- ~0.0: strong support for MSA2")

if __name__ == "__main__":
    main()
