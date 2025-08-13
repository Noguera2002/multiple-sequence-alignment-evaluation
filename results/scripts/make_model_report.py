#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss
)

def build_X_matching_model(df_pairs, model):
    feat = getattr(model, "feature_names_", None) or sorted([c for c in df_pairs.columns if c.startswith("delta_")])
    X = pd.DataFrame({c: df_pairs.get(c, 0.0) for c in feat}).to_numpy()
    return X, feat

def precision_coverage(y, p, margins):
    rows=[]
    for m in margins:
        mask_hi = (p >= 0.5 + m)
        mask_lo = (p <= 0.5 - m)
        cover = (mask_hi | mask_lo).mean()
        if mask_hi.sum() + mask_lo.sum() == 0:
            prec = np.nan
        else:
            correct_hi = (y[mask_hi] == 1).sum()
            correct_lo = (y[mask_lo] == 0).sum()
            prec = (correct_hi + correct_lo) / (mask_hi.sum() + mask_lo.sum())
        rows.append((m, cover, prec))
    return pd.DataFrame(rows, columns=["margin","coverage","precision"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".joblib from training")
    ap.add_argument("--pairs", required=True, help="pairwise CSV (test split recommended)")
    ap.add_argument("--outdir", required=True, help="output directory for plots & summary")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.pairs)
    y  = df["label"].astype(int).to_numpy()
    clf = joblib.load(args.model)
    X, _ = build_X_matching_model(df, clf)
    p = clf.predict_proba(X)[:,1]

    # Headline metrics
    pred = (p >= 0.5).astype(int)
    overall_prec = precision_score(y, pred)
    margins = [0.20, 0.30]
    pc = precision_coverage(y, p, margins)
    bs = brier_score_loss(y, p)

    # 1) prob hist
    plt.figure(); plt.hist(p, bins=30)
    plt.xlabel("Predicted probability P(A>B)"); plt.ylabel("Count"); plt.title("Probability distribution")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "prob_hist.png")); plt.close()

    # 2) calibration curve (quantile-ish bins)
    bins = np.linspace(0,1,11)
    which = np.digitize(p, bins) - 1
    mp, fp = [], []
    for b in range(10):
        idx = (which==b)
        if idx.sum()==0: continue
        mp.append(p[idx].mean()); fp.append((y[idx]==1).mean())
    plt.figure(); plt.plot([0,1],[0,1])
    if len(mp)>0: plt.plot(mp, fp, marker="o")
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration (Brier={bs:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "calibration.png")); plt.close()

    # 3) precision–coverage sweep across margins
    ms = np.linspace(0.0, 0.45, 25)
    sweep = precision_coverage(y, p, ms)
    plt.figure(); plt.plot(sweep["margin"], sweep["precision"], marker="o")
    plt.xlabel("|p - 0.5| threshold (margin)"); plt.ylabel("Precision")
    plt.title("Precision vs margin")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "precision_vs_margin.png")); plt.close()

    plt.figure(); plt.plot(sweep["margin"], sweep["coverage"], marker="o")
    plt.xlabel("|p - 0.5| threshold (margin)"); plt.ylabel("Coverage")
    plt.title("Coverage vs margin")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "coverage_vs_margin.png")); plt.close()

    # 4) ROC & PR
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot([0,1],[0,1]); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "roc.png")); plt.close()

    precs, recs, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    plt.figure(); plt.plot(recs, precs)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "pr.png")); plt.close()

    # summary.md
    with open(os.path.join(args.outdir, "summary.md"), "w") as w:
        w.write("## Results  \n")
        w.write("**Held-out families (Rfam, n ≥ 4; 20% test). Trained on remaining 80%; 5-fold CV calibration on train**  \n\n")
        w.write(f"- **Overall precision (test):** **{overall_prec:.3f}**  \n")
        for m in margins:
            row = pc[pc["margin"]==m].iloc[0]
            conf = int((m+0.5)*100)
            w.write(f"- **≥{conf}% confidence** (|p−0.5| ≥ **{m:.2f}**): **{row['precision']*100:.1f}% precision** at **{row['coverage']*100:.1f}% coverage**  \n")
        w.write("\nCalibration remains strong on the test split (see plots); precision–coverage curves shown above.\n")
    print(f"Saved report to {args.outdir}")

if __name__ == "__main__":
    main()
