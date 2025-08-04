# Scripts for MSA Evaluation and Model Pipeline

This directory contains all the Python scripts for:

1. **Computing MSA metrics**
2. **Preparing pairwise feature datasets**
3. **Training and calibrating the logistic regression model**
4. **Evaluating and predicting with the trained model**

---

## Script Overview

| Script | Purpose | Example Usage |
|--------|---------|----------------|
| **metrics.py** | Compute descriptive metrics for all MSAs (Rfam + tools) | `python3 metrics.py ../data/msa/Rfam ../data/msa/MAFFT ../data/msa/Clustal ../data/msa/T-coffee ../data/msa/Muscle -o ../data/processed/all_metrics.csv` |
| **compute_sp_alltools.py** | Compute Sum-of-Pairs (SP) scores of tool MSAs vs Rfam | `python3 compute_sp_alltools.py --rfam ../data/msa/Rfam --mafft ../data/msa/MAFFT --clustal ../data/msa/Clustal --tcoffee ../data/msa/T-coffee --muscle ../data/msa/Muscle --out ../data/processed/all_sp_scores.csv` |
| **assemble_data.py** | Merge metrics and SP scores into a single CSV for pairwise generation | `python3 assemble_data.py --metrics ../data/processed/all_metrics.csv --sp ../data/processed/all_sp_scores.csv --out ../data/processed/merged_data.csv` |
| **filter_families.py** | Keep only families with ≥N sequences | `python3 filter_families.py --rawdir ../data/raw/RawSeqs --min-seqs 4 --out ../data/processed/keep_families.txt` |
| **prepare_pairwise_data.py** | Generate pairwise feature datasets with labels for training | `python3 prepare_pairwise_data.py --metrics ../data/processed/merged_data.csv --sp ../data/processed/all_sp_scores.csv --output ../data/processed/pairwise_data.csv` |
| **train_and_calibrate.py** | Train elastic-net logistic regression and calibrate probabilities | `python3 train_and_calibrate.py --train-pairs ../data/processed/train_pairs.csv --model-out ../model/calibrated_model.joblib --cv-folds 5 --method isotonic` |
| **evaluate.py** | Evaluate the model on a labeled test set with a confidence threshold | `python3 evaluate.py --model ../model/calibrated_model.joblib --test-pairs ../data/processed/test_pairs.csv --conf-thresh 0.3` |
| **predict.py** | Compare two new MSAs and output calibrated probability that MSA1 is better | `python predict.py msa1.fasta msa2.fasta` |

---

## Notes

- predict.py is the only script needed for end users who just want to compare two new MSAs.
- All other scripts are for data preparation, model training, and evaluation.
- Ensure the Python dependencies in requirements_scripts.txt are installed.