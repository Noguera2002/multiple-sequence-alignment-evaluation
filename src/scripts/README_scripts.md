# Scripts for MSA Evaluation and Model Pipeline

This directory contains all scripts to compute metrics, build pairwise features, train & calibrate the model, evaluate, predict, and plot results.

All examples assume you run them **from the repo root**.

## Core scripts (with examples)

| Script | Purpose | Example usage |
|---|---|---|
| **src/data/scripts/make_stockholm_from_seed.py** | Split `Rfam.seed` into per-family **Stockholm** files. | `python3 src/data/scripts/make_stockholm_from_seed.py --seed src/data/raw/Rfam.seed --outdir src/data/stockholm` |
| **src/scripts/filter_families.py** | Create whitelist of families with **n ≥ 4** sequences. | `python3 src/scripts/filter_families.py --rfam-dir src/data/msa/Rfam --min-seqs 4 --out src/data/processed/kept_families.txt` |
| **src/scripts/metrics.py** | Compute **descriptive metrics** for all MSAs (Rfam + tools). | `python3 src/scripts/metrics.py src/data/msa/Rfam src/data/msa/MAFFT src/data/msa/Clustal src/data/msa/T-coffee src/data/msa/Muscle -o src/data/processed/metrics.csv --families src/data/processed/kept_families.txt` |
| **src/scripts/compute_sp_alltools.py** | Compute **SP**, **SP_stem**, **SP_loop** for each tool MSA **vs the per-family Stockholm reference**. | `python3 src/scripts/compute_sp_alltools.py --ref-stockholm src/data/stockholm --clustal src/data/msa/Clustal --mafft src/data/msa/MAFFT --tcoffee src/data/msa/T-coffee --muscle src/data/msa/Muscle --families src/data/processed/kept_families.txt --out src/data/processed/sp_struct.csv` |
| **src/scripts/build_labels.py** | Build **composite label** `sp = mean(SP, SP_stem, SP_loop)`. | `python3 src/scripts/build_labels.py --sp-input src/data/processed/sp_struct.csv --out src/data/processed/sp_for_train_SPall.csv` |
| **src/scripts/prepare_pairwise_data.py** | Generate **pairwise features**: antisymmetric deltas (A−B); drop near ties. | `python3 src/scripts/prepare_pairwise_data.py --metrics src/data/processed/metrics.csv --sp src/data/processed/sp_for_train_SPall.csv --label-col sp --tie-margin 0.02 --output src/data/processed/train_pairs_SPall.csv` |
| **src/scripts/split_pairs_by_family.py** | Split pairs **by family** into train/test (e.g., 80/20). | `python3 src/scripts/split_pairs_by_family.py --pairs src/data/processed/train_pairs_SPall.csv --out-train src/data/processed/train_pairs_SPall.train.csv --out-test src/data/processed/train_pairs_SPall.test.csv --seed 42 --test-frac 0.2` |
| **src/scripts/train_and_calibrate.py** | Train logistic regression (**L2**, **no intercept**) and **sigmoid-calibrate** with 5-fold CV. | `python3 src/scripts/train_and_calibrate.py --pairs src/data/processed/train_pairs_SPall.train.csv --model-out src/model/model_SPall_sigC3.train.joblib --penalty l2 --C 3.0 --calib sigmoid --cv 5` |
| **src/scripts/evaluate.py** | Precision/coverage at a **confidence margin** (`--conf-thresh`). | `python3 src/scripts/evaluate.py --model src/model/model_SPall_sigC3.train.joblib --test-pairs src/data/processed/train_pairs_SPall.test.csv --conf-thresh 0.0`<br>`python3 src/scripts/evaluate.py --model src/model/model_SPall_sigC3.train.joblib --test-pairs src/data/processed/train_pairs_SPall.test.csv --conf-thresh 0.2`<br>`python3 src/scripts/evaluate.py --model src/model/model_SPall_sigC3.train.joblib --test-pairs src/data/processed/train_pairs_SPall.test.csv --conf-thresh 0.3` |
| **src/scripts/predict.py** | Compare **two MSAs**; output calibrated probability **P(MSA1 > MSA2)**. | `python3 src/scripts/predict.py path/to/MSA1.fasta path/to/MSA2.fasta --model src/model/model_SPall_sigC3.train.joblib` |
| **src/scripts/evaluate_crossmetric.py** *(optional)* | Regret & winner-accuracy versus **SP / SP_stem / SP_loop**. | `python3 src/scripts/evaluate_crossmetric.py --model src/model/model_SPall_sigC3.train.joblib --pairs src/data/processed/train_pairs_SPall.test.csv --sp src/data/processed/sp_struct.csv --out src/data/processed/eval_crossmetric_SPall.csv` |



## Plotting (results/)

| Script | Purpose | Output |
|---|---|---|
| **results/scripts/plot_model_diagnostics.py** | Calibration, probability & margin histograms, ROC/PR, precision/coverage sweeps, top coefficients. | `results/plots_model/*.png` |
| **results/scripts/plot_results_summary.py** | Per-family accuracy, average regret & winner-accuracy by metric, probability histograms (correct/incorrect). | `results/plots_results/*.png` |
| **results/scripts/plots.sh** | Run both plotting scripts with sensible defaults. | Both plot folders are populated. |



## Notes

- predict.py is the only script needed (together with the model_SPall_sigC3.joblib) for end users who just want to compare two new MSAs.
- All other scripts are for data preparation, model training, and evaluation.
- Ensure the Python dependencies in requirements_scripts.txt are installed.
