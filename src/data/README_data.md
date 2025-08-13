# Data Preparation

This document explains how to obtain and preprocess the data used to train and evaluate the MSA selector.

---

## Overview
1. **Get Rfam.seed** and split into per-family **Stockholm** references.
2. **Collect / generate tool MSAs** (MAFFT, Clustal Omega, T-Coffee, MUSCLE, plus the Rfam benchmark MSA).
3. **Produce processed CSVs** used by the model pipeline.

---

## Step 1 – Download `Rfam.seed`
Place the Stockholm file at: src/data/raw/Rfam.seed

## Step 2 – Split into per-family `.sto` references
python3 src/data/scripts/make_stockholm_from_seed.py \
  --seed   src/data/raw/Rfam.seed \
  --outdir src/data/stockholm

## Step 3 – Prepare tool MSA directories
Ensure these exist and contain the tool outputs:
src/data/msa/Rfam
src/data/msa/MAFFT
src/data/msa/Clustal
src/data/msa/T-coffee
src/data/msa/Muscle

## Step 4 – Filter families (n ≥ 4)
python3 src/scripts/filter_families.py \
  --rfam-dir src/data/msa/Rfam \
  --min-seqs 4 \
  --out src/data/processed/kept_families.txt

## Step 5 – Compute descriptive metrics
python3 src/scripts/metrics.py \
  src/data/msa/Rfam src/data/msa/MAFFT src/data/msa/Clustal \
  src/data/msa/T-coffee src/data/msa/Muscle \
  -o src/data/processed/metrics.csv \
  --families src/data/processed/kept_families.txt

## Step 6 – Compute SP scores
python3 src/scripts/compute_sp_alltools.py \
  --ref-stockholm src/data/stockholm \
  --clustal   src/data/msa/Clustal \
  --mafft     src/data/msa/MAFFT \
  --tcoffee   src/data/msa/T-coffee \
  --muscle    src/data/msa/Muscle \
  --families  src/data/processed/kept_families.txt \
  --out       src/data/processed/sp_for_train_SPall.csv

## Step 7 – Generate pairwise training data
python3 src/scripts/prepare_pairwise_data.py \
  --metrics   src/data/processed/metrics.csv \
  --sp        src/data/processed/sp_for_train_SPall.csv \
  --label-col sp \
  --tie-margin 0.02 \
  --output    src/data/processed/train_pairs_SPall.csv

## Notes
- Family filter: all reported results use kept_families.txt (n ≥ 4).
- Tie margin: 0.02 is recommended to remove near-ties in SP.
 Keep tool versions in src/data/msa/version.txt for reproducibility.




