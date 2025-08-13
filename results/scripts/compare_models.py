#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt

def build_X_matching_model(df_pairs, model):
    feat = getattr(model, "feature_names_", None) or sorted([c for c in df_pairs.columns if c.startswith("delta_")])
    X = pd.DataFrame({c: df_pairs.get(c, 0.0) for c in feat}).to_numpy()
    return X

def prec_cov(y, p, margins):
    out=[]
    for m in margins:
        mask_hi = (p >= 0.5 + m)
        mask_lo = (p <= 0.5 - m)
        cover = (mask_hi | mask_lo).mean()
        if mask_hi.sum() + mask_lo.sum() == 0:
            prec = np.nan
        else:
            correct = (y[mask_hi]==1).sum() + (y[mask_lo]==0).sum()
            prec = correct / (mask_hi.sum() + mask_lo.sum())
        out.append((m, cover, prec))
    return np.array(out)

def calib_points(y, p, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    which = np.digitize(p, bins) - 1
    mp, fp = [], []
    for b in range(n_bins):
        idx = (which==b)
        if idx.sum()==0: continue
        mp.append(p[idx].mean())
        fp.append((y[idx]==1).mean())
    return np.array(mp), np.array(fp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-model", required=True)
    ap.add_argument("--rf-model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.pairs)
    y  = df["label"].astype(int).to_numpy()

    lr = joblib.load(args.lr_model); X_lr = build_X_matching_model(df, lr); p_lr = lr.predict_proba(X_lr)[:,1]
    rf = joblib.load(args.rf_model); X_rf = build_X_matching_model(df, rf); p_rf = rf.predict_proba(X_rf)[:,1]

    margins = np.linspace(0.0, 0.45, 25)
    lr_pc = prec_cov(y, p_lr, margins)
    rf_pc = prec_cov(y, p_rf, margins)

    # Precision–coverage
    plt.figure()
    plt.plot(margins, lr_pc[:,2], label="Logistic", marker="o")
    plt.plot(margins, rf_pc[:,2], label="Random Forest", marker="o")
    plt.xlabel("|p - 0.5| threshold (margin)"); plt.ylabel("Precision")
    plt.title("Precision vs margin"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "precision_vs_margin_lr_vs_rf.png")); plt.close()

    plt.figure()
    plt.plot(margins, lr_pc[:,1], label="Logistic", marker="o")
    plt.plot(margins, rf_pc[:,1], label="Random Forest", marker="o")
    plt.xlabel("|p - 0.5| threshold (margin)"); plt.ylabel("Coverage")
    plt.title("Coverage vs margin"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "coverage_vs_margin_lr_vs_rf.png")); plt.close()

    # Calibration overlay
    mp_lr, fp_lr = calib_points(y, p_lr, n_bins=10)
    mp_rf, fp_rf = calib_points(y, p_rf, n_bins=10)
    plt.figure(); plt.plot([0,1],[0,1], color="black")
    if len(mp_lr)>0: plt.plot(mp_lr, fp_lr, marker="o", label="Logistic")
    if len(mp_rf)>0: plt.plot(mp_rf, fp_rf, marker="o", label="Random Forest")
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy"); plt.title("Calibration (reliability)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "calibration_lr_vs_rf.png")); plt.close()

    # Print canonical margins to stdout
    for m in [0.20, 0.30]:
        def row(pc, m):
            i = np.argmin(np.abs(pc[:,0]-m)); return pc[i,1], pc[i,2]
        cov_lr, pre_lr = row(lr_pc, m)
        cov_rf, pre_rf = row(rf_pc, m)
        print(f"|p-0.5|>={m:.2f}  LR: {pre_lr*100:.1f}% @ {cov_lr*100:.1f}%   RF: {pre_rf*100:.1f}% @ {cov_rf*100:.1f}%")

    print(f"Saved LR vs RF comparison to {args.outdir}")

if __name__ == "__main__":
    main()
