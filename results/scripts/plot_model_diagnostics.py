#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_curve, auc, precision_recall_curve, average_precision_score

def build_X_matching_model(df_pairs, model):
    feat = getattr(model, "feature_names_", None) or sorted([c for c in df_pairs.columns if c.startswith("delta_")])
    X = pd.DataFrame({c: df_pairs.get(c, 0.0) for c in feat}).to_numpy()
    return X, feat

def threshold_sweep(y, p, thresholds):
    rows=[]
    for thr in thresholds:
        conf = thr - 0.5
        mask = (np.abs(p-0.5) >= conf)
        if mask.sum()==0: rows.append((thr, 0.0, np.nan, 0)); continue
        preds = (p[mask] >= 0.5).astype(int)
        rows.append((thr, mask.mean(), (preds == y[mask]).mean(), int(mask.sum())))
    return map(np.array, zip(*rows))

def calibration_curve_points(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    which = np.digitize(p, bins) - 1
    prob_true=[]; prob_pred=[]
    for b in range(n_bins):
        idx = (which==b)
        if idx.sum()==0: continue
        prob_true.append((y[idx]==1).mean())
        prob_pred.append(p[idx].mean())
    return np.array(prob_pred), np.array(prob_true)

def avg_coefficients(model):
    coefs=[]
    for cc in getattr(model, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est and hasattr(est, "named_steps") and "clf" in est.named_steps:
            clf = est.named_steps["clf"]
            if hasattr(clf, "coef_"): coefs.append(clf.coef_.ravel())
    if not coefs: return None
    return np.mean(np.vstack(coefs), axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="src/model/model_SPall_sigC3.joblib")
    ap.add_argument("--pairs",  default="src/data/processed/train_pairs_SPall.csv")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(here, "..", "plots_model")))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.pairs)
    y  = df["label"].astype(int).to_numpy()
    model = joblib.load(args.model)
    X, feature_names = build_X_matching_model(df, model)
    p = model.predict_proba(X)[:,1]

    # probability hist
    plt.figure(); plt.hist(p, bins=30)
    plt.xlabel("Predicted P(tool1 > tool2)"); plt.ylabel("Count"); plt.title("Probability distribution")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "prob_hist.png")); plt.close()

    # margin hist
    plt.figure(); plt.hist(np.abs(p-0.5), bins=30)
    plt.xlabel("|p - 0.5|"); plt.ylabel("Count"); plt.title("Confidence (margin) distribution")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "margin_hist.png")); plt.close()

    # calibration
    pred, true = calibration_curve_points(y, p, n_bins=10)
    bs = brier_score_loss(y, p)
    plt.figure(); plt.plot([0,1],[0,1])
    if len(pred)>0: plt.plot(pred, true, marker="o")
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy"); plt.title(f"Reliability (Brier={bs:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "calibration.png")); plt.close()

    # threshold sweeps
    thresholds = np.linspace(0.5, 0.95, 10)
    thr, cov, prec, n = threshold_sweep(y, p, thresholds); thr, cov, prec, n = map(np.array, (thr,cov,prec,n))
    plt.figure(); plt.plot(thr, prec, marker="o"); plt.xlabel("Decision threshold"); plt.ylabel("Precision"); plt.title("Precision vs threshold")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "precision_vs_threshold.png")); plt.close()
    plt.figure(); plt.plot(thr, cov, marker="o"); plt.xlabel("Decision threshold"); plt.ylabel("Coverage"); plt.title("Coverage vs threshold")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "coverage_vs_threshold.png")); plt.close()
    pd.DataFrame({"threshold":thr,"coverage":cov,"precision":prec,"n":n}).to_csv(os.path.join(args.outdir, "threshold_sweep.csv"), index=False)

    # ROC
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot([0,1],[0,1]); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "roc.png")); plt.close()

    # PR
    precs, recalls, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    plt.figure(); plt.plot(recalls, precs); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "pr.png")); plt.close()

    # top-20 coeffs
    coefs = avg_coefficients(model)
    if coefs is not None and feature_names is not None:
        order = np.argsort(np.abs(coefs))[::-1][:20]
        names = [feature_names[i] for i in order]; vals = np.abs(coefs[order])
        plt.figure(); plt.bar(range(len(names)), vals); plt.xticks(range(len(names)), names, rotation=90)
        plt.ylabel("|coefficient| (std. scaled)"); plt.title("Top feature effects (logistic)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "coef_magnitude_top20.png")); plt.close()

    print(f"Saved model diagnostics to {args.outdir}")

if __name__ == "__main__":
    main()
