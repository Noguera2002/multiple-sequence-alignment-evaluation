#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import pandas as pd
import joblib

RF_RE = re.compile(r"(RF\d{5})", re.IGNORECASE)

def canon_family(x):
    if pd.isna(x): return x
    m = RF_RE.search(str(x))
    return m.group(1).upper() if m else str(x).strip()

def canon_tool(x):
    if pd.isna(x): return x
    s = str(x).lower().strip().replace(" ","").replace("-","").replace("_","")
    if s.startswith("tcof"): s = "tcoffee"
    if s.startswith("clustal"): s = "clustal"
    if s.startswith("mafft"): s = "mafft"
    if s.startswith("muscle"): s = "muscle"
    if s.startswith("rfam"): s = "rfam"
    return s

def load_pairs(pairs_csv):
    df = pd.read_csv(pairs_csv)
    df["family"] = df["family"].map(canon_family)
    df["tool1"]  = df["tool1"].map(canon_tool)
    df["tool2"]  = df["tool2"].map(canon_tool)
    return df

def load_quality(sp_csv):
    df = pd.read_csv(sp_csv).rename(columns={"Family":"family","Source":"tool"})
    df["family"] = df["family"].map(canon_family)
    df["tool"]   = df["tool"].map(canon_tool)
    keep = [c for c in ["family","tool","SP","SP_stem","SP_loop"] if c in df.columns]
    df = df[keep]
    for c in ["SP","SP_stem","SP_loop"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.groupby(["family","tool"], as_index=False).mean(numeric_only=True)
    return df

def build_X(df_pairs, model):
    feat = getattr(model, "feature_names_", None)
    if feat is None:
        feat = sorted([c for c in df_pairs.columns if c.startswith("delta_")])
    Xdf = pd.DataFrame({c: df_pairs.get(c, 0.0) for c in feat})
    return Xdf.to_numpy(), feat

def main():
    ap = argparse.ArgumentParser(description="Cross-metric regret & winner-accuracy (SP, SP_stem, SP_loop)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--sp", required=True)
    ap.add_argument("--out", default="src/data/processed/eval_crossmetric_SPall.csv")
    args = ap.parse_args()

    pairs   = load_pairs(args.pairs)
    quality = load_quality(args.sp)
    model   = joblib.load(args.model)

    X, _ = build_X(pairs, model)
    p = model.predict_proba(X)[:,1]
    win1 = (p >= 0.5)

    results = []
    for metric in ["SP","SP_stem","SP_loop"]:
        if metric not in quality.columns: 
            continue
        q = quality[["family","tool",metric]].rename(columns={metric:"q"})

        q1 = pairs[["family","tool1"]].merge(q, left_on=["family","tool1"], right_on=["family","tool"], how="left")["q"].to_numpy()
        q2 = pairs[["family","tool2"]].merge(q, left_on=["family","tool2"], right_on=["family","tool"], how="left")["q"].to_numpy()

        chosen = np.where(win1, q1, q2)
        other  = np.where(win1, q2, q1)

        valid = np.isfinite(chosen) & np.isfinite(other)
        if valid.sum()==0: 
            continue

        ch = chosen[valid]; ot = other[valid]
        best = np.maximum(q1[valid], q2[valid])
        regret = best - ch
        winner_acc_strict  = float((ch > ot).mean())
        winner_acc_tiehalf = float(((ch > ot) + 0.5*(np.isclose(ch, ot))).mean())
        avg_regret = float(np.mean(regret))

        results.append({
            "metric": metric,
            "avg_regret": avg_regret,
            "winner_acc_strict": winner_acc_strict,
            "winner_acc_tiehalf": winner_acc_tiehalf,
            "n": int(valid.sum())
        })

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    if not out.empty:
        print(out.to_string(index=False))
    print(f"\n✔ Wrote {args.out}")

if __name__ == "__main__":
    main()
