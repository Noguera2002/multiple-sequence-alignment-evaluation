# Data Directory

This folder explains how to obtain and preprocess the data used to train and evaluate the MSA selector.

## What this covers
1) **Get Rfam seed** and split into per-family **Stockholm** references  
2) **Collect / generate tool MSAs** (MAFFT, Clustal Omega, T-Coffee, MUSCLE, plus the Rfam benchmark MSA)  
3) **Produce processed CSVs** used by the model pipeline

## Preparation steps

### 1) Download `Rfam.seed`
Place the Stockholm file at: 
src/data/raw/Rfam.seed

### 2) Split into per-family Stockholm files
python3 src/data/scripts/make_stockholm_from_seed.py \
  --seed   src/data/raw/Rfam.seed \
  --outdir src/data/stockholm

### 3) Ensure tool MSAs directories exist
src/data/msa/Rfam

src/data/msa/MAFFT

src/data/msa/Clustal

src/data/msa/T-coffee

src/data/msa/Muscle

### 4) Filter families (n ≥ 4) -> kept_families.txt
Create the list used by downstream steps:

## Produce processed CSVs (for the model)

### A) Descriptive metrics
python3 src/scripts/metrics.py \
  src/data/msa/Rfam src/data/msa/MAFFT src/data/msa/Clustal \
  src/data/msa/T-coffee src/data/msa/Muscle \
  -o src/data/processed/metrics.csv \
  --families src/data/processed/kept_families.txt

### B) SP scores against the Stockholm reference (SP, SP_stem, SP_loop)
python3 src/scripts/compute_sp_alltools.py \
  --ref-stockholm src/data/stockholm \
  --clustal   src/data/msa/Clustal \
  --mafft     src/data/msa/MAFFT \
  --tcoffee   src/data/msa/T-coffee \
  --muscle    src/data/msa/Muscle \
  --families  src/data/processed/kept_families.txt \
  --out       src/data/processed/sp_struct.csv

### C) Composite training label (mean of SP, SP_stem, SP_loop)
python3 src/scripts/prepare_pairwise_data.py \
  --metrics   src/data/processed/metrics.csv \
  --sp        src/data/processed/sp_for_train_SPall.csv \
  --label-col sp \
  --tie-margin 0.01 \
  --output    src/data/processed/train_pairs_SPall.csv

### D) Pairwise training data (antisymmetric deltas; drop near ties)
python3 src/scripts/prepare_pairwise_data.py \
  --metrics   src/data/processed/metrics.csv \
  --sp        src/data/processed/sp_for_train_SPall.csv \
  --label-col sp \
  --tie-margin 0.01 \
  --output    src/data/processed/train_pairs_SPall.csv

## Notes
- Family filter: all reported results use the filtered set in src/data/processed/kept_families.txt (n ≥ 4).
- Reproducibility: keep tool versions in src/data/msa/version.txt if you re-run aligners.











