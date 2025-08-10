#!/usr/bin/env python3
"""
prepare_pairwise_data.py (robust merge, antisymmetric deltas)

Build pairwise training data from:
  - metrics.csv  (family, tool, ... numeric metrics ...)
  - sp.csv       (family, tool, <label column>)

Key features:
  * Canonicalize family (RFxxxxx) and tool names on BOTH tables
  * Collapse duplicates by averaging per (family, tool)
  * Only linear antisymmetric deltas (A - B); no interactions
  * Drop near ties via --tie-margin
"""

import argparse
import itertools
import re
import pandas as pd
import numpy as np

ID_COLS = {"Family","family","Source","tool","Filename","filename"}

RF_RE = re.compile(r"(RF\d{5})", re.IGNORECASE)

def canonical_family(x: str) -> str:
    if pd.isna(x): return x
    m = RF_RE.search(str(x))
    return m.group(1).upper() if m else str(x).strip()

def canonical_tool(x: str) -> str:
    if pd.isna(x): return x
    s = str(x).lower().strip()
    # remove separators
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    # normalize common variants
    s = s.replace("tcoffee", "tcoffee")
    s = s.replace("tcoffe", "tcoffee")
    s = s.replace("clustalo", "clustal")
    s = s.replace("mafft", "mafft")
    s = s.replace("muscle", "muscle")
    s = s.replace("rfam", "rfam")
    return s

def infer_numeric_cols(df: pd.DataFrame):
    cols=[]
    for c in df.columns:
        if c in ID_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def tidy_metrics(dfm: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    dfm = dfm.rename(columns={"Family":"family","Source":"tool"}).copy()
    dfm["family"] = dfm["family"].map(canonical_family)
    dfm["tool"]   = dfm["tool"].map(canonical_tool)
    metric_cols = infer_numeric_cols(dfm)
    # average duplicates per (family, tool)
    dfm = (dfm[["family","tool"] + metric_cols]
           .groupby(["family","tool"], as_index=False)
           .mean(numeric_only=True))
    return dfm, metric_cols

def tidy_labels(dfs: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in dfs.columns:
        raise SystemExit(f"--label-col '{label_col}' not found in SP file. "
                         f"Available: {', '.join([c for c in dfs.columns if c not in ('family','tool')])}")
    dfs = dfs.rename(columns={"Family":"family","Source":"tool"}).copy()
    dfs["family"] = dfs["family"].map(canonical_family)
    dfs["tool"]   = dfs["tool"].map(canonical_tool)
    dfs = dfs[["family","tool",label_col]].rename(columns={label_col:"sp"})
    dfs = dfs.dropna(subset=["sp"])
    # average duplicates per (family, tool)
    dfs = dfs.groupby(["family","tool"], as_index=False).mean(numeric_only=True)
    return dfs

def make_pairs(df_metrics: pd.DataFrame, df_sp: pd.DataFrame, label_col: str, tie_margin: float) -> pd.DataFrame:
    dfm, metric_cols = tidy_metrics(df_metrics)
    dfsp = tidy_labels(df_sp, label_col)

    # inner join; now guaranteed unique on both sides
    df = pd.merge(dfm, dfsp, on=["family","tool"], how="inner", validate="one_to_one")

    rows=[]
    for fam, g in df.groupby("family", sort=False):
        # deterministic order by tool to avoid duplicate (A,B)/(B,A)
        recs = g.sort_values("tool").to_dict("records")
        for a, b in itertools.combinations(recs, 2):
            d_sp = a["sp"] - b["sp"]
            if np.isnan(d_sp) or abs(d_sp) < tie_margin:
                continue
            label = 1 if d_sp > 0 else 0
            row = {"family": fam, "tool1": a["tool"], "tool2": b["tool"], "label": label}
            for m in metric_cols:
                # antisymmetric deltas: ALWAYS A - B (do not flip by label)
                row[f"delta_{m}"] = float(a[m]) - float(b[m])
            rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Create pairwise training data with antisymmetric delta features.")
    ap.add_argument("--metrics", required=True, help="metrics.csv (family,tool,...metrics...)")
    ap.add_argument("--sp", required=True, help="sp_for_train_*.csv (family,tool,sp)")
    ap.add_argument("--output", required=True, help="Output pairwise CSV")
    ap.add_argument("--label-col", default="sp", help="Column name in --sp to use as label (default: sp)")
    ap.add_argument("--tie-margin", type=float, default=0.0, help="Drop pairs with |Δlabel| < margin")
    args = ap.parse_args()

    df_met = pd.read_csv(args.metrics)
    df_sp  = pd.read_csv(args.sp)

    df_pairs = make_pairs(df_met, df_sp, args.label_col, args.tie_margin)
    if df_pairs.empty:
        raise SystemExit("No pairs produced. Check that families overlap and tie-margin isn't too large.")
    df_pairs.to_csv(args.output, index=False)
    print(f"✔ Generated {len(df_pairs)} pairs; features: "
          + ", ".join([c for c in df_pairs.columns if c.startswith('delta_')]))

if __name__ == "__main__":
    main()
