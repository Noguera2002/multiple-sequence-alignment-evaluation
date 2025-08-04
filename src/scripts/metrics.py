#!/usr/bin/env python3
"""
model.py

Compute core MSA metrics for Rfam, MAFFT, Clustal, T-coffee, and Muscle directories.

Usage:
    python3 model.py <rfam_dir> <mafft_dir> <clustal_dir> <tcoffee_dir> <muscle_dir> \
                     -o metrics_raw.csv [--families keep_families.txt]

Output CSV columns:
    Family, Source, Filename,
    Match, Mismatch, Gap, IC, GapAC, LFPR, FlipScore,
    MeanPairwiseID, AlignmentLength
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.fft import rfft

def read_alignment(path):
    # Read FASTA alignment, return padded sequences
    seqs = []
    curr = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                curr = []
                seqs.append(curr)
            else:
                curr.extend(list(line))
    if not seqs:
        return []
    # pad all to same length
    L = max(len(s) for s in seqs)
    return [''.join(s).ljust(L, '-') for s in seqs]

def build_state_mask(seqs):
    # 0 = mismatch, 1 = gap, 2 = match
    N, L = len(seqs), len(seqs[0])
    # consensus per column
    cons = []
    for j in range(L):
        col = [s[j] for s in seqs if s[j] != '-']
        cons.append(Counter(col).most_common(1)[0][0] if col else '-')
    mask = np.zeros((N, L), int)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            if c == '-':
                mask[i, j] = 1
            elif c == cons[j]:
                mask[i, j] = 2
    return mask

def mean_column_entropy(seqs):
    L = len(seqs[0])
    ent = []
    for j in range(L):
        col = [s[j] for s in seqs if s[j] != '-']
        if not col:
            ent.append(0.0)
        else:
            freqs = np.array(list(Counter(col).values()), float)
            freqs /= freqs.sum()
            ent.append(-np.sum(freqs * np.log2(freqs)))
    return float(np.mean(ent))

def gap_adj_corr(seqs):
    N, L = len(seqs), len(seqs[0])
    gap = np.array([[c=='-' for c in s] for s in seqs], float)
    corrs = []
    for j in range(L-1):
        x, y = gap[:, j], gap[:, j+1]
        if x.std()>0 and y.std()>0:
            corrs.append(np.corrcoef(x, y)[0,1])
    return float(np.mean(corrs)) if corrs else 0.0

def LFPR(mask_match):
    N, L = mask_match.shape
    if L < 2:
        return 0.0
    agg = (mask_match.sum(axis=0) > (N//2)).astype(float)
    spec = np.abs(rfft(agg))**2
    total = spec.sum()
    if total <= 0.0:
        return 0.0
    cutoff = max(1, int(len(spec)*0.1))
    return float(spec[:cutoff].sum() / total)

def flip_score(mask_match, mask_gap):
    state = (mask_match*2 + mask_gap).astype(int)
    N, L = state.shape
    if L < 2:
        return 1.0
    flips = sum(((row[:-1] != row[1:]).sum()) for row in state)
    return 1.0 - flips / (N * (L-1))

def mean_pairwise_id(seqs):
    """
    Fast, exact average pairwise identity:
    For each column j, sum n_a*(n_a-1) over residues,
    normalize by N*(N-1)*L.
    """
    N = len(seqs)
    if N < 2:
        return 1.0
    L = len(seqs[0])
    total = 0
    for j in range(L):
        counts = Counter(s[j] for s in seqs if s[j] != '-')
        total += sum(n*(n-1) for n in counts.values())
    return float(total) / (N*(N-1)*L)

def compute_metrics(seqs):
    state     = build_state_mask(seqs)
    mask_match = (state == 2).astype(float)
    mask_gap   = (state == 1).astype(float)
    mets = {
        'Match':       mask_match.mean(),
        'Mismatch':    (state == 0).astype(float).mean(),
        'Gap':         mask_gap.mean(),
        'IC':          1.0 - mean_column_entropy(seqs),
        'GapAC':       gap_adj_corr(seqs),
        'LFPR':        LFPR(mask_match),
        'FlipScore':   flip_score(mask_match, mask_gap),
        'MeanPairwiseID': mean_pairwise_id(seqs),
        'AlignmentLength': len(seqs[0])
    }
    return mets

def process_dir(root, label, keep_families=None):
    recs = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.fasta'):
            continue
        fam_match = re.match(r'^(RF\d{5})', fn)
        if not fam_match:
            continue
        family = fam_match.group(1)
        if keep_families is not None and family not in keep_families:
            continue  # skip filtered-out family
        path = os.path.join(root, fn)
        seqs = read_alignment(path)
        # trim columns that are all gaps
        arr = np.array([list(s) for s in seqs])
        keep = ~(arr == '-').all(axis=0)
        seqs = [''.join(r[keep]) for r in arr]
        m = compute_metrics(seqs)
        rec = {
            **m,
            'Family':   family,
            'Source':   label,
            'Filename': fn
        }
        recs.append(rec)
    return recs

def main():
    p = argparse.ArgumentParser(
        description='Compute MSA metrics for five tool outputs'
    )
    p.add_argument('rfam_dir',    help='Ground-truth Rfam seed alignments')
    p.add_argument('mafft_dir',   help='MSA outputs from MAFFT')
    p.add_argument('clustal_dir', help='MSA outputs from Clustal')
    p.add_argument('tcoffee_dir', help='MSA outputs from T-Coffee')
    p.add_argument('muscle_dir',  help='MSA outputs from MUSCLE')
    p.add_argument('-o', '--output', required=True,
                   help='Path to write combined metrics CSV')
    p.add_argument('--families', help='Optional whitelist file of family accessions to include (one per line)')
    args = p.parse_args()

    keep = None
    if args.families:
        with open(args.families) as f:
            keep = set(line.strip() for line in f if line.strip())
        print(f"Filtering to {len(keep)} families from {args.families}")

    all_recs = []
    all_recs += process_dir(args.rfam_dir,    'Rfam', keep_families=keep)
    all_recs += process_dir(args.mafft_dir,   'MAFFT', keep_families=keep)
    all_recs += process_dir(args.clustal_dir, 'Clustal', keep_families=keep)
    all_recs += process_dir(args.tcoffee_dir, 'T-Coffee', keep_families=keep)
    all_recs += process_dir(args.muscle_dir,  'Muscle', keep_families=keep)

    df = pd.DataFrame.from_records(all_recs)
    cols = [
        'Family','Source','Filename',
        'Match','Mismatch','Gap','IC','GapAC','LFPR','FlipScore',
        'MeanPairwiseID','AlignmentLength'
    ]
    df.to_csv(args.output, index=False, columns=cols, float_format='%.6f')
    print(f"✔ Wrote {len(df)} rows to {args.output}")

if __name__ == '__main__':
    main()
