#!/usr/bin/env python3
"""
Split pairwise data by FAMILY into train/test (reproducible).
- Ensures all pairs from a family go to the same split.
- Writes train/test CSVs + family lists for transparency.

Usage:
  python3 src/scripts/split_pairs_by_family.py \
    --pairs src/data/processed/train_pairs_SPall.csv \
    --out-train src/data/processed/train_pairs_SPall.train.csv \
    --out-test  src/data/processed/train_pairs_SPall.test.csv \
    --seed 42 --test-frac 0.2
"""
import argparse, os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    fams = np.array(sorted(df["family"].unique()))
    rng = np.random.RandomState(args.seed)
    n_test = max(1, int(len(fams) * args.test_frac))
    test_fams = set(rng.choice(fams, size=n_test, replace=False))
    train_fams = [f for f in fams if f not in test_fams]

    df_train = df[df["family"].isin(train_fams)].reset_index(drop=True)
    df_test  = df[df["family"].isin(test_fams)].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test), exist_ok=True)
    df_train.to_csv(args.out_train, index=False)
    df_test.to_csv(args.out_test, index=False)

    # Also write family lists for reproducibility
    base_dir = os.path.dirname(args.out_train) or "."
    with open(os.path.join(base_dir, "train_families.txt"), "w") as f:
        for fam in train_fams: f.write(f"{fam}\n")
    with open(os.path.join(base_dir, "test_families.txt"), "w") as f:
        for fam in sorted(test_fams): f.write(f"{fam}\n")

    print(f"Train families: {len(train_fams)} | pairs: {len(df_train)} -> {args.out_train}")
    print(f"Test  families: {len(test_fams)}  | pairs: {len(df_test)}  -> {args.out_test}")
    print(f"Wrote family lists to {base_dir}/train_families.txt and {base_dir}/test_families.txt")

if __name__ == "__main__":
    main()
