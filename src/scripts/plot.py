import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

# Parameters
model_path = "src/model/calibrated_model.joblib"
test_pairs = "src/data/processed/test_pairs.csv"
thresholds = [0.3, 0.5, 0.75]  # choose the cutoffs you want to highlight
outfile = "src/results/simple_precision_coverage.png"

# Load
model = joblib.load(model_path)
df = pd.read_csv(test_pairs)
y = df['label'].to_numpy()

# Reconstruct features same as training (assuming interactions were used)
def build_features(df, add_interactions=True):
    feature_cols = [c for c in df.columns if c.startswith('delta_')]
    X = df[feature_cols].copy()
    if add_interactions:
        for i in range(len(feature_cols)):
            for j in range(i, len(feature_cols)):
                a = feature_cols[i]
                b = feature_cols[j]
                X[f'{a}_x_{b}'] = X[a] * X[b]
    return X

X = build_features(df, add_interactions=True)
probs = model.predict_proba(X)[:,1]

# Compute precision and coverage
precisions = []
coverages = []
for t in thresholds:
    sel = probs >= t
    if sel.sum() == 0:
        precisions.append(0.0)
        coverages.append(0.0)
    else:
        precisions.append((y[sel] == 1).sum() / sel.sum())
        coverages.append(sel.sum() / len(y))

# Plot grouped bars
x = np.arange(len(thresholds))
width = 0.35

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(x - width/2, precisions, width, label='Precision', edgecolor='black')
ax.bar(x + width/2, coverages, width, label='Coverage', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
ax.set_ylim(0,1)
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Fraction")
ax.set_title("Precision and Coverage at Selected Thresholds")
ax.legend()
for i in range(len(thresholds)):
    ax.text(x[i]-width/2, precisions[i]+0.02, f"{precisions[i]:.2f}", ha='center', va='bottom', fontsize=8)
    ax.text(x[i]+width/2, coverages[i]+0.02, f"{coverages[i]:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(outfile, dpi=300)
print(f"Saved simple plot to {outfile}")
