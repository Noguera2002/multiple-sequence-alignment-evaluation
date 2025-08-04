# Jaime Noguera Thesis: MSA Evaluation

**Interpretable and simple comparison of multiple sequence alignments.**

## Model

This model compares two MSAs and outputs the probability that the first alignment is better than the second alignment.
Confidence is calibrated, so a probability of 0.8 means that about 8 out of 10 similar predictions are expected to be correct.

- Values near 1 indicate strong support for the first MSA.
- Values near 0 indicate strong support for the second MSA.
- Values near 0.5 indicate no clear preference.

## Overview

The repository explains the development of this model.
The pipeline of works to develop the model is the following:
- Extract and filter seed families from Rfam.
- MSA generation with tools (MAFFT, Clustal Omega, T-coffee, Muscle) + MSA from Rfam
- Compute descriptive alignment metrics and SP (sum of pairs) scores.  
- Build pairwise feature differences and train a logistic regression model with elastic-net-style interactions.  
- Calibrate its output probabilities (isotonic) to ensure reliability.  
- Evaluate high-confidence predictions through precision/coverage thresholding.

## Highlights

- Calibrated probabilistic output: confidence scores reflect empirical accuracy.  
- Thresholding enables trading coverage for precision—users can choose conservatively reliable comparisons.  
- Simple model (logistic regression + interactions) keeps interpretability.  
- Family-size filtering improves usable confident coverage.

## Repository Layout

├── README.md # Project description + Overview
├── LICENSE # License (Apache)
├── requirements.txt
├── src/
│ ├── data/ # Raw, MSA, and processed data (see src/data/README.md)
│ ├── scripts/ # Processing / training / evaluation scripts (see src/scripts/README.md)
│ ├── model/ # Saved calibrated model(s)
│ └── results/ # Evaluation summaries and figures

## Prerequisites

- Python 3.10 or newer
- Recommended: create a virtual environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## How to run program: Comparing Two MSAs

python src/scripts/predict.py path/to/MSA1.fasta path/to/MSA2.fasta

**Example output**

P(MSA1 > MSA2) = 0.823
Interpretation for the user:
- Values near 1 indicate strong support for the first MSA.
- Values near 0 indicate strong support for the second MSA.
- Values near 0.5 indicate no clear preference.


## Results

On the held-out test set:
- Precision at 0.8 confidence: 84.6%
- Coverage at 0.8 confidence: 12.8%
- Calibration and precision/coverage plots are in src/results/.
- High-confidence predictions allow conservative, reliable comparisons of alignments.

## Contact

Author: Jaime Noguera
Affiliation: Technical University of Denmark (DTU)
Email: s233773@dtu.dk