# Data Directory

This directory describes how to obtain and process the data used for the model development.

The contents support:

**1. Extraction of seed families from Rfam** 

**2. Generation of multiple sequence alignments (MSAs) with standard tools** 

**3. Filtering and preparation of processed datasets for the predictive model** 

## Data Preparation Pipeline

### 1. Download Rfam.seed from Rfam website**

### 2. Extract raw ungapped sequences from Rfam.seed**

python src/data/scripts/extract_seed_raws.py

### 3. Generate reference per-family FASTAs from Rfam.seed**

python src/data/scripts/export_rfam_to_fasta.py \
    --stockholm raw/Rfam.seed \
    --outdir msa/Rfam

### 4. Generate MSAs with all tools

bash src/data/scripts/run_all_msa.sh

## Notes

- Families with fewer than 4 sequences are removed later in the pipeline by filter_families.py
- Tool versions are logged in msa/versions.txt for reproducibility
- Only this folder and the Python scripts in scripts/ are needed to recreate the raw and MSA data
