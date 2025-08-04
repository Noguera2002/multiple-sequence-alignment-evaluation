#!/usr/bin/env python3
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

METRICS = ["Match","Mismatch","Gap","IC","GapAC","LFPR","FlipScore","MeanPairwiseID","AlignmentLength"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', required=True, help='Merged CSV from metrics+SP')
    parser.add_argument('--out-dir', required=True, help='Directory to write split files')
    args = parser.parse_args()

    df = pd.read_csv(args.merged)
    fams = df['family'].unique()

    train_f, temp_f = train_test_split(fams, test_size=0.30, random_state=42)
    calib_f, test_f = train_test_split(temp_f, test_size=0.50, random_state=42)

    sets = {
        'train': df[df.family.isin(train_f)],
        'calib': df[df.family.isin(calib_f)],
        'test' : df[df.family.isin(test_f)],
    }

    for name, subset in sets.items():
        subset.to_csv(f"{args.out_dir}/{name}_full.csv", index=False)
        subset[['family','tool']+METRICS].to_csv(f"{args.out_dir}/{name}_metrics.csv", index=False)
        subset[['family','tool','sp']].to_csv(f"{args.out_dir}/{name}_sp.csv", index=False)
        print(f"Wrote {name}_metrics.csv ({len(subset)} rows) and {name}_sp.csv in {args.out_dir}")

if __name__ == '__main__':
    main()
