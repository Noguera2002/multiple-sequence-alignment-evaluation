# Data Directory

This directory contains all raw and processed data used for MSA evaluation and model training.

The contents support:
1. **Extraction of seed families from Rfam**
2. **Generation of multiple sequence alignments (MSAs) with standard tools**
3. **Filtering and preparation of processed datasets for the predictive model**

## Directory Structure

├── raw/ # Source files from Rfam
│ ├── Rfam.seed # Stockholm alignment (downloaded from Rfam)
│ └── RawSeqs/ # Ungapped FASTA sequences extracted from Rfam.seed
├── msa/ # Tool-generated MSAs + Rfam reference alignments
│ ├── Clustal/
│ ├── MAFFT/
│ ├── Muscle/
│ ├── T-coffee/
│ └── Rfam/
│
├── processed/ # Filtered families, computed metrics, SP scores, train/test splits
│
├── scripts/ # Data preparation scripts
│ ├── export_rfam_to_fasta.py # Convert Rfam.seed to per-family gapped FASTA
│ ├── extract_seed_raws.py # Extract ungapped sequences to RawSeqs/
│ └── run_all_msa.sh # Generate MSAs for all tools from RawSeqs/
│
├── requirements_data.txt # Environment for data preparation
└── README_data.md # This file

## Data Preparation Pipeline

### 1. Extract raw ungapped sequences from Rfam.seed

python src/data/scripts/extract_seed_raws.py

### 2. Generate reference per-family FASTAs from Rfam.seed

python src/data/scripts/export_rfam_to_fasta.py \
    --stockholm raw/Rfam.seed \
    --outdir msa/Rfam

## 3. Generate MSAs with all tools

bash src/data/scripts/run_all_msa.sh

## Notes

- Families with fewer than 4 sequences are removed later in the pipeline by filter_families.py
- Tool versions are logged in msa/versions.txt for reproducibility
- Only this folder and the Python scripts in scripts/ are needed to recreate the raw and MSA data