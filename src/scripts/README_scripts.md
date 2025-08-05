# Scripts for MSA Evaluation and Model Pipeline

This directory contains all the Python scripts for:

1. **Computing MSA metrics**
2. **Preparing pairwise feature datasets**
3. **Training and calibrating the logistic regression model**
4. **Evaluating and predicting with the trained model**

---


## Script Overview

| Script                                 | Purpose                                                                                | Example Usage                                                                                                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **metrics.py**                         | Compute descriptive metrics for all MSAs (Rfam + tools)                                | `python3 metrics.py ../data/msa/Rfam ../data/msa/MAFFT ../data/msa/Clustal ../data/msa/T-coffee ../data/msa/Muscle -o ../data/processed/all_metrics.csv`                                                              |
| **compute\_sp\_alltools.py**           | Compute Sum-of-Pairs (SP) scores of tool MSAs vs Rfam                                  | `python3 compute_sp_alltools.py --rfam ../data/msa/Rfam --mafft ../data/msa/MAFFT --clustal ../data/msa/Clustal --tcoffee ../data/msa/T-coffee --muscle ../data/msa/Muscle --out ../data/processed/all_sp_scores.csv` |
| **assemble\_data.py**                  | Merge metrics and SP scores into a single CSV for pairwise generation                  | `python3 assemble_data.py --metrics ../data/processed/all_metrics.csv --sp ../data/processed/all_sp_scores.csv --out ../data/processed/merged_data.csv`                                                               |
| **filter\_families.py**                | Keep only families with ≥N sequences                                                   | `python3 filter_families.py --rawdir ../data/raw/RawSeqs --min-seqs 4 --out ../data/processed/keep_families.txt`                                                                                                      |
| **select\_metrics.py**                 | Select necessary delta-metrics via LRT-based backward elimination or RFECV             | `python3 select_metrics.py ../data/processed/train_pairs.csv --method lrt --alpha 0.05 --output ../data/processed/selected_metrics.txt`                                                                               |
| **prepare\_pairwise\_data.py (train)** | Generate pairwise feature dataset for the training split using all nine MSA metrics    | `python3 prepare_pairwise_data.py --metrics ../data/processed/splits/train_metrics.csv --sp ../data/processed/splits/train_sp.csv --output ../data/processed/train_pairs_all9.csv`                                    |
| **prepare\_pairwise\_data.py (calib)** | Generate pairwise feature dataset for the calibration split using all nine MSA metrics | `python3 prepare_pairwise_data.py --metrics ../data/processed/splits/calib_metrics.csv --sp ../data/processed/splits/calib_sp.csv --output ../data/processed/calib_pairs_all9.csv`                                    |
| **prepare\_pairwise\_data.py (test)**  | Generate pairwise feature dataset for the test split using all nine MSA metrics        | `python3 prepare_pairwise_data.py --metrics ../data/processed/splits/test_metrics.csv --sp ../data/processed/splits/test_sp.csv --output ../data/processed/test_pairs_all9.csv`                                       |
| **train\_and\_calibrate.py**           | Train elastic-net logistic regression and calibrate probabilities                      | `python3 train_and_calibrate.py --train-pairs ../data/processed/train_pairs_all9.csv --model-out ../model/calibrated_model.joblib --cv-folds 5 --method isotonic`                                                          |
| **evaluate.py**                        | Evaluate the model on a labeled test set with a confidence threshold                   | `python3 evaluate.py --model ../model/calibrated_model.joblib --test-pairs ../data/processed/test_pairs_all9.csv --conf-thresh 0.3`                                                                                        |
| **predict.py**                         | Compare two new MSAs and output calibrated probability that MSA1 is better             | `python predict.py msa1.fasta msa2.fasta`                                                                                                                                                                             |


## Notes

- predict.py is the only script needed for end users who just want to compare two new MSAs.
- All other scripts are for data preparation, model training, and evaluation.
- Ensure the Python dependencies in requirements_scripts.txt are installed.
