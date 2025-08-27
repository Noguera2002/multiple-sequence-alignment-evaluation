# Jaime Noguera Thesis: MSA Evaluation

**What this does:** given two MSAs, predict a **probability** that the first is better than the second.

- ~1.0 → strong support for **MSA1**
- ~0.5 → no clear preference
- ~0.0 → strong support for **MSA2**

**How it was trained:**
- **Labels:** for each family, we score every tool-produced MSA **against the per-family Rfam reference** using **SP**. We use **SP** difference to label comparisons. If tool1 is better than tool2 the label is 1. If its the contrary, the label is 0.
- **Features:** **antisymmetric deltas** (A−B) of simple, fast alignment metrics.
- **Model:** **logistic regression**, **no intercept**, **sigmoid calibration**, **C=3**.

## Applicability

- **Inference on any RNA:** compare **any two RNA MSAs** (FASTA). No Rfam data needed at prediction time.
- **Training data:** trained and CV-calibrated on **Rfam families (n ≥ 4)**.
- **Input expectations:** nucleotide alphabet (A/C/G/U; T treated as U), aligned FASTA, ≥4 sequences per MSA recommended.
- **Future work: expand** training data and tools.

## Overview

- **Extract and filter families from Rfam (n ≥ 4)**, then split Rfam.seed into per-family Stockholm files.
- **Generate MSAs with tools** (**MAFFT, Clustal Omega, T-Coffee, MUSCLE**) and include the **Rfam reference** MSA.
- **Compute descriptive metrics** for each alignment **and SP score against the reference**.
- **Build pairwise features** as antisymmetric deltas **(metric(tool1) − metric(tool2))**.
- **Label each pair** by the **SP difference**: label=1 if tool1 has higher SP than tool2, otherwise 0.
- **Train logistic regression** (**L2 regularization, no intercept, no interactions**); **sigmoid calibration** with **5-fold CV**.
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
**Held-out families (Rfam, n ≥ 4; 20% test). Trained on remaining 80%; 5-fold CV calibration on train; tie-margin = 0.02**  

- **Overall precision (test):** **0.732**  
- **≥70% confidence** (|p−0.5| ≥ **0.20**): **82.7% precision** at **57.1% coverage**  
- **≥80% confidence** (|p−0.5| ≥ **0.30**): **88.7% precision** at **35.5% coverage**  

Calibration remains strong on the test split (see **results/plots_model/**); cross-metric and per-family analyses indicate low regret and balanced performance (see **results/plots_results/**).



## Contact

Author: Jaime Noguera, Email: s233773@dtu.dk

