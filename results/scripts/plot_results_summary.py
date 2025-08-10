#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt

def build_X_matching_model(df_pairs, model):
    feat = getattr(model, "feature_names_", None) or sorted([c for c in df_pairs.columns if c.startswith("delta_")])
    return pd.DataFrame({c: df_pairs.get(c, 0.0) for c in feat}).to_numpy()

def load_quality(sp_csv):
    q = pd.read_csv(sp_csv).rename(columns={"Family":"family","Source":"tool"})
    keep = [c for c in ["family","tool","SP","SP_stem","SP_loop"] if c in q.columns]
    q = q[keep]
    for c in ["SP","SP_stem","SP_loop"]:
        if c in q.columns: q[c] = pd.to_numeric(q[c], errors="coerce")
    return q.groupby(["family","tool"], as_index=False).mean(numeric_only=True)

def regret_for_metric(pairs, quality, metric):
    q = quality[["family","tool",metric]].rename(columns={metric:"q"})
    q1 = pairs[["family","tool1"]].merge(q, left_on=["family","tool1"], right_on=["family","tool"], how="left")["q"].to_numpy()
    q2 = pairs[["family","tool2"]].merge(q, left_on=["family","tool2"], right_on=["family","tool"], how="left")["q"].to_numpy()
    return q1, q2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="src/model/model_SPall_sigC3.joblib")
    ap.add_argument("--pairs", default="src/data/processed/train_pairs_SPall.csv")
    ap.add_argument("--sp",    default="src/data/processed/sp_struct.csv")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(here, "..", "plots_results")))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pairs = pd.read_csv(args.pairs)
    y     = pairs["label"].astype(int).to_numpy()
    model = joblib.load(args.model)
    X     = build_X_matching_model(pairs, model)
    p     = model.predict_proba(X)[:,1]
    pred  = (p >= 0.5).astype(int)
    correct = (pred == y)

    # probs (correct vs incorrect)
    plt.figure(); plt.hist(p[correct], bins=30)
    plt.xlabel("Predicted probability (correct)"); plt.ylabel("Count"); plt.title("Probability distribution (correct)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "prob_hist_correct.png")); plt.close()

    plt.figure(); plt.hist(p[~correct], bins=30)
    plt.xlabel("Predicted probability (incorrect)"); plt.ylabel("Count"); plt.title("Probability distribution (incorrect)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "prob_hist_incorrect.png")); plt.close()

    # per-family accuracy
    fam_acc = pairs.assign(correct=correct).groupby("family")["correct"].mean().values
    plt.figure(); plt.hist(fam_acc, bins=20)
    plt.xlabel("Per-family accuracy"); plt.ylabel("Families"); plt.title("Per-family accuracy distribution")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "per_family_accuracy_hist.png")); plt.close()

    # regret & winrate by metric
    quality = load_quality(args.sp)
    metrics = [m for m in ["SP","SP_stem","SP_loop"] if m in quality.columns]
    regrets=[]; winrates=[]
    for m in metrics:
        q1, q2 = regret_for_metric(pairs, quality, m)
        chosen = np.where(pred==1, q1, q2)
        other  = np.where(pred==1, q2, q1)
        valid  = np.isfinite(chosen) & np.isfinite(other)
        if valid.sum()==0: regrets.append(np.nan); winrates.append(np.nan); continue
        ch, ot = chosen[valid], other[valid]
        best = np.maximum(q1[valid], q2[valid])
        regrets.append(float((best - ch).mean()))
        winrates.append(float((ch > ot).mean()))

    # bars
    plt.figure(); plt.bar(range(len(metrics)), regrets)
    plt.xticks(range(len(metrics)), metrics)
    plt.ylabel("Average regret"); plt.title("Average regret by metric")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "avg_regret_bar.png")); plt.close()

    plt.figure(); plt.bar(range(len(metrics)), winrates)
    plt.xticks(range(len(metrics)), metrics)
    plt.ylabel("Strict winner accuracy"); plt.title("Winner accuracy by metric")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "winner_accuracy_bar.png")); plt.close()

    print(f"Saved result plots to {args.outdir}")

if __name__ == "__main__":
    main()
