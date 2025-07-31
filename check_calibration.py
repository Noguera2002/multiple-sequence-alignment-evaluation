import pandas as pd, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def build_features(df):
    feature_cols = [c for c in df.columns if c.startswith('delta_')]
    X = df[feature_cols].copy()
    for i in range(len(feature_cols)):
        for j in range(i, len(feature_cols)):
            a = feature_cols[i]
            b = feature_cols[j]
            X[f'{a}_x_{b}'] = X[a] * X[b]
    return X

# Load
df_cal = pd.read_csv("calib_pairs.csv")
df_test = pd.read_csv("test_pairs.csv")
clf = joblib.load("calibrated_model.joblib")

X_cal = build_features(df_cal); y_cal = df_cal["label"]
X_test = build_features(df_test); y_test = df_test["label"]

probs_cal = clf.predict_proba(X_cal)[:,1]
probs_test = clf.predict_proba(X_test)[:,1]

# Calibration curve
frac_pos_cal, mean_pred_cal = calibration_curve(y_cal, probs_cal, n_bins=10, strategy="uniform")
frac_pos_test, mean_pred_test = calibration_curve(y_test, probs_test, n_bins=10, strategy="uniform")

# Brier and ECE
def ece(y, probs, bins=10):
    edges = np.linspace(0,1,bins+1)
    e = 0.0
    N = len(y)
    for i in range(bins):
        if i < bins-1:
            mask = (probs >= edges[i]) & (probs < edges[i+1])
        else:
            mask = (probs >= edges[i]) & (probs <= edges[i+1])
        if mask.sum() == 0: continue
        acc = y[mask].mean()
        conf = probs[mask].mean()
        e += (mask.sum()/N) * abs(acc - conf)
    return e

brier_cal = brier_score_loss(y_cal, probs_cal)
brier_test = brier_score_loss(y_test, probs_test)
ece_cal = ece(y_cal, probs_cal)
ece_test = ece(y_test, probs_test)

# Plot reliability
plt.figure()
plt.plot(mean_pred_cal, frac_pos_cal, "s-", label="calib")
plt.plot([0,1],[0,1],"k:", label="ideal")
plt.title("Reliability Curve - Calibration")
plt.xlabel("Predicted prob")
plt.ylabel("Observed freq")
plt.legend()
plt.savefig("reliability_calibration.png")

plt.figure()
plt.plot(mean_pred_test, frac_pos_test, "s-", label="test")
plt.plot([0,1],[0,1],"k:", label="ideal")
plt.title("Reliability Curve - Test")
plt.xlabel("Predicted prob")
plt.ylabel("Observed freq")
plt.legend()
plt.savefig("reliability_test.png")

# Confident region stats (|p-0.5|>=0.3 -> >=80% confidence)
def confident_stats(y, probs, thresh=0.3):
    mask = np.abs(probs - 0.5) >= thresh
    pred = (probs > 0.5).astype(int)
    correct = ((pred == 1) & (y == 1)) | ((pred == 0) & (y == 0))
    coverage = mask.sum() / len(y)
    precision = (correct & mask).sum() / (mask.sum() if mask.sum()>0 else 1)
    return coverage, precision, mask.sum()

cov_cal, prec_cal, n_cal = confident_stats(y_cal, probs_cal)
cov_test, prec_test, n_test = confident_stats(y_test, probs_test)

# Report
print("CALIBRATION SET")
print(f"  Brier: {brier_cal:.4f}, ECE: {ece_cal:.4f}")
print(f"  Confident (>=80%) pairs: {n_cal} ({cov_cal:.1%} coverage), Precision: {prec_cal:.3f}")
print("TEST SET")
print(f"  Brier: {brier_test:.4f}, ECE: {ece_test:.4f}")
print(f"  Confident (>=80%) pairs: {n_test} ({cov_test:.1%} coverage), Precision: {prec_test:.3f}")
