# Jaime Noguera Thesis: MSA Evaluation

**What this does:** given two MSAs, predict a **calibrated probability** that the first is better than the second.

- ~1.0 → strong support for **MSA1**
- ~0.5 → no clear preference
- ~0.0 → strong support for **MSA2**

**How it was trained:**
- **Labels:** for each family, we score every tool-produced MSA **against the per-family Rfam Stockholm reference** using **SP**, **SP_stem** (paired columns), and **SP_loop** (unpaired). The training target is their mean.
- **Features:** **antisymmetric deltas** (A−B) of simple, fast alignment metrics.
- **Model:** **logistic regression**, **no intercept**, **sigmoid calibration**, **C=3**.

## Applicability

- **Inference on any RNA:** compare **any two RNA MSAs** (FASTA). No Rfam data needed at prediction time.
- **Training domain:** trained and CV-calibrated on **Rfam families (n ≥ 4)**; calibration/precision may shift on very different datasets.
- **Input expectations:** nucleotide alphabet (A/C/G/U; T treated as U), aligned FASTA, ≥4 sequences per MSA works best.
- **Future work:** small labeled sets in new domains can be used to **re-calibrate** with the provided training script.

## Overview

- **Extract and filter families from Rfam** (n ≥ 4), then split `Rfam.seed` into per-family **Stockholm** files.
- **Generate MSAs** with tools (**MAFFT**, **Clustal Omega**, **T-Coffee**, **MUSCLE**) and include the **Rfam** benchmark MSA.
- **Compute descriptive metrics** and **SP-based scores** against the Stockholm reference: **SP**, **SP_stem**, **SP_loop**.
- **Build pairwise features** as **antisymmetric deltas** (A−B) of the metrics; training label = **mean(SP, SP_stem, SP_loop)**.
- **Train** logistic regression (L2, **no intercept**, **no interactions**); **sigmoid** calibration with 5-fold CV.
- **Evaluate** with **precision/coverage** at confidence thresholds; include model/result plots.

## Prerequisites

- Python 3.10+
- Recommended: virtual environment

pip install -r requirements.txt


## Compare two MSAs

python src/scripts/predict.py path/to/MSA1.fasta path/to/MSA2.fasta

**Example output**

P(MSA1 > MSA2) = 0.823
Interpretation for the user:
- Values near 1 indicate strong support for the first MSA.
- Values near 0 indicate strong support for the second MSA.
- Values near 0.5 indicate no clear preference.

## Results

**Internal validation on Rfam; calibrated with 5-fold CV**

- Overall precision: 0.720 (all pairs)
- ≥70% confidence (|p−0.5| ≥ 0.20): 83.5% precision at 43.9% coverage
- ≥80% confidence (|p−0.5| ≥ 0.30): 88.6% precision at 24.1% coverage

Calibration is strong (reliability curve near diagonal; Brier ≈ **0.19**). Cross-metric and per-family analyses indicate low regret and balanced performance (see results/plots_model/ and results/plots_results/).


## Contact

Author: Jaime Noguera
Affiliation: Technical University of Denmark (DTU)
Email: s233773@dtu.dk

