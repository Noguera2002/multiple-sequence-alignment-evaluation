#!/usr/bin/env python3
"""
Generate pairwise feature datasets from family-level metrics and SP scores.
Uses the top 5 metrics: IC, FlipScore, MeanPairwiseID, Mismatch, Match.
Outputs CSV suitable for training/calibration/testing.
"""
import pandas as pd
import argparse
import itertools

# Top predictive metrics
TOP_METRICS = ["IC", "FlipScore", "MeanPairwiseID", "Mismatch", "Match"]


def make_pairs(df_metrics, df_sp):
    # Merge metrics and SP
    df = pd.merge(df_metrics, df_sp, on=["family", "tool"], validate="one_to_one")
    # Ensure metrics exist
    metric_cols = [m for m in TOP_METRICS if m in df.columns]

    records = []
    for fam, group in df.groupby("family"):
        tools = group["tool"].unique()
        for a, b in itertools.permutations(tools, 2):
            rowA = group[group["tool"] == a].iloc[0]
            rowB = group[group["tool"] == b].iloc[0]
            rec = {"tool1": a, "tool2": b}
            # Delta features
            for m in metric_cols:
                rec[f"delta_{m}"] = rowA[m] - rowB[m]
            # Label: 1 if SP(tool1)>SP(tool2), else 0
            rec["label"] = 1 if rowA["sp"] > rowB["sp"] else 0
            records.append(rec)

    return pd.DataFrame(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, help='Metrics CSV (family,tool,IC,...)')
    parser.add_argument('--sp', required=True, help='SP CSV (family,tool,sp)')
    parser.add_argument('--output', required=True, help='Output pairwise CSV')
    args = parser.parse_args()

    df_met = pd.read_csv(args.metrics)
    df_sp = pd.read_csv(args.sp)
    df_pairs = make_pairs(df_met, df_sp)
    df_pairs.to_csv(args.output, index=False)
    print(f"Generated {len(df_pairs)} pairwise records → {args.output}")
