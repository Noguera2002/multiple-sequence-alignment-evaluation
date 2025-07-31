#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split

# load the merged table
df = pd.read_csv("merged.csv")
fams = df['family'].unique()

# 70% train, 30% temp
train_f, temp_f = train_test_split(fams, test_size=0.30, random_state=42)
# split temp into 50/50 → 15% calib, 15% test
calib_f, test_f = train_test_split(temp_f, test_size=0.50, random_state=42)

sets = {
    'train': df[df.family.isin(train_f)],
    'calib': df[df.family.isin(calib_f)],
    'test' : df[df.family.isin(test_f)],
}

METRICS = ["Match","Mismatch","Gap","IC","GapAC","LFPR","FlipScore","MeanPairwiseID","AlignmentLength"]

for name, subset in sets.items():
    # full table
    subset.to_csv(f"{name}_full.csv", index=False)
    # separate metrics vs sp
    subset[['family','tool']+METRICS].to_csv(f"{name}_metrics.csv", index=False)
    subset[['family','tool','sp']].to_csv(f"{name}_sp.csv",      index=False)
    print(f"Wrote {name}_metrics.csv ({len(subset)} rows) and {name}_sp.csv")

