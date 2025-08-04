#!/usr/bin/env python3
"""
predict.py

Compare two MSAs with the trained calibrated model and output the probability
that the first MSA is better than the second.

Interpretation for users:
- Values near 1 indicate strong support for the first MSA.
- Values near 0 indicate strong support for the second MSA.
- Values near 0.5 indicate no clear preference.

Usage:
    python src/scripts/predict.py msa1.fasta msa2.fasta
"""

import sys
import pandas as pd
import joblib
from pathlib import Path
from tempfile import TemporaryDirectory

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "calibrated_model.joblib"

# ---- Helpers to reuse your existing code ----
from metrics import compute_metrics, read_alignment
from prepare_pairwise_data import TOP_METRICS

def compute_metrics_for_file(msa_file, tool_label="ToolX", family_label="Query"):
    """Compute the same metrics as in metrics.py for a single MSA file."""
    seqs = read_alignment(msa_file)
    mets = compute_metrics(seqs)
    rec = {
        **mets,
        "Family": family_label,
        "Source": tool_label,
        "Filename": Path(msa_file).name
    }
    return rec

def build_delta_features(df_metrics):
    """
    Build delta features exactly like prepare_pairwise_data.py
    but without SP scores (label is omitted for prediction).
    """
    assert len(df_metrics) == 2, "Need exactly two MSAs"
    tool1, tool2 = df_metrics.iloc[0], df_metrics.iloc[1]
    rec = {"tool1": tool1["Source"], "tool2": tool2["Source"]}
    for m in TOP_METRICS:
        rec[f"delta_{m}"] = tool1[m] - tool2[m]
    return pd.DataFrame([rec])

def add_interactions(df):
    """Add interaction features as done in training."""
    feature_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[feature_cols].copy()
    for i in range(len(feature_cols)):
        for j in range(i, len(feature_cols)):
            a = feature_cols[i]
            b = feature_cols[j]
            X[f"{a}_x_{b}"] = X[a] * X[b]
    return X

def main(msa1, msa2):
    # Compute metrics for the two new MSAs
    df_metrics = pd.DataFrame([
        compute_metrics_for_file(msa1, tool_label="MSA1"),
        compute_metrics_for_file(msa2, tool_label="MSA2"),
    ])

    # Build pairwise delta features
    df_delta = build_delta_features(df_metrics)
    X = add_interactions(df_delta)

    # Load trained, calibrated model
    model = joblib.load(MODEL_PATH)

    # Predict probability that MSA1 > MSA2
    prob = model.predict_proba(X)[:, 1][0]

    # Print result
    print(f"P(MSA1 > MSA2) = {prob:.3f}")
    print("Interpretation for the user:")
    print("- Values near 1 indicate strong support for the first MSA.")
    print("- Values near 0 indicate strong support for the second MSA.")
    print("- Values near 0.5 indicate no clear preference.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <msa1.fasta> <msa2.fasta>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
