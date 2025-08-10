#!/usr/bin/env python3
"""
predict.py (antisymmetric, no interactions)

Compare two MSAs with the trained calibrated model and output
P(MSA1 > MSA2). Builds ONLY the linear delta_* features, in the
exact order saved in the model (feature_names_).
"""

import sys
import argparse
import pandas as pd
import joblib
from pathlib import Path

# Reuse metric code
from metrics import compute_metrics, read_alignment

ID_COLS = {"Family","family","Source","tool","Filename","filename"}

def compute_metrics_for_file(msa_file, tool_label="MSA", family_label="Query"):
    seqs = read_alignment(msa_file)
    mets = compute_metrics(seqs)
    return {
        **mets,
        "Family": family_label,
        "Source": tool_label,
        "Filename": Path(msa_file).name
    }

def infer_metric_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def build_delta_frame(m1: dict, m2: dict, feature_names):
    """
    Build a single-row DataFrame with columns exactly = feature_names.
    feature_names are like ['delta_Gap','delta_Match',...]
    """
    # raw deltas for all numeric metrics present
    df_tmp = pd.DataFrame([m1, m2])
    metric_cols = infer_metric_cols(df_tmp)
    deltas = {f"delta_{k}": float(m1[k]) - float(m2[k]) for k in metric_cols}

    # create X row with the trained feature order; fill missing with 0.0
    row = {name: deltas.get(name, 0.0) for name in feature_names}
    return pd.DataFrame([row])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msa1", help="Path to first MSA (FASTA)")
    ap.add_argument("msa2", help="Path to second MSA (FASTA)")
    ap.add_argument("--model", default="models/logreg_calibrated.joblib",
                    help="Path to trained calibrated model (.joblib)")
    args = ap.parse_args()

    # Load model and its feature order
    model = joblib.load(args.model)
    feature_names = getattr(model, "feature_names_", None)
    if not feature_names:
        sys.exit("Model missing feature_names_. Retrain with updated train_and_calibrate.py")

    # Compute metrics for both MSAs
    m1 = compute_metrics_for_file(args.msa1, tool_label="MSA1")
    m2 = compute_metrics_for_file(args.msa2, tool_label="MSA2")

    # Build delta features in the exact trained order
    X = build_delta_frame(m1, m2, feature_names)

    # Predict P(MSA1 > MSA2)
    prob = float(model.predict_proba(X)[:,1][0])

    print(f"P(MSA1 > MSA2) = {prob:.3f}")
    print("Interpretation:")
    print("- ~1.0: strong support for MSA1")
    print("- ~0.5: no clear preference")
    print("- ~0.0: strong support for MSA2")

if __name__ == "__main__":
    main()
