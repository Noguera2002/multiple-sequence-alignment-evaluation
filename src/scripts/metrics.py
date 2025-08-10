#!/usr/bin/env python3
"""
metrics.py (pruned + enriched)

Outputs per-MSA metrics:
  Match, Gap, GapAC, GapAC_lag2, LFPR, MeanPairwiseID,
  AlignmentLength, GapOpenRate, GapRunMean
plus bookkeeping columns:
  Family, Source, Filename

Notes:
- We keep Gap and Match (drop Mismatch: = 1 - Gap - Match).
- We drop IC and FlipScore to reduce redundancy.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.fft import rfft

# ---------------- IO ----------------

def read_alignment(path):
    """Read FASTA alignment and return sequences padded to same length with '-'."""
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
    L = max(len(s) for s in seqs)
    return [''.join(s).ljust(L, '-') for s in seqs]

# ---------------- Core helpers ----------------

def build_state_mask(seqs):
    """Return integer mask with: 0=mismatch, 1=gap, 2=match-to-consensus."""
    N, L = len(seqs), len(seqs[0])
    # per-column consensus ignoring gaps
    cons = []
    for j in range(L):
        col = [s[j] for s in seqs if s[j] != '-']
        cons.append(Counter(col).most_common(1)[0][0] if col else '-')
    mask = np.zeros((N, L), dtype=int)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            if c == '-':
                mask[i, j] = 1
            elif c == cons[j]:
                mask[i, j] = 2
            # else: 0 = mismatch
    return mask

def gap_adj_corr_lag(seqs, lag=1):
    """Average correlation of gap indicators between columns separated by `lag`."""
    N, L = len(seqs), len(seqs[0])
    if L <= lag:
        return 0.0
    gap = np.array([[c == '-' for c in s] for s in seqs], float)
    corrs = []
    for j in range(L - lag):
        x, y = gap[:, j], gap[:, j + lag]
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            corrs.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0

def LFPR(mask_match):
    """
    Low-Frequency Power Ratio of majority-match indicator across columns.
    Emphasizes blocky/smooth patterns rather than choppy ones.
    """
    N, L = mask_match.shape
    if L < 2:
        return 0.0
    agg = (mask_match.sum(axis=0) > (N // 2)).astype(float)  # majority match per column
    spec = np.abs(rfft(agg)) ** 2
    total = spec.sum()
    if total <= 0.0:
        return 0.0
    cutoff = max(1, int(len(spec) * 0.1))  # lowest 10% of frequencies
    return float(spec[:cutoff].sum() / total)

def mean_pairwise_id(seqs):
    """
    Exact average pairwise identity over residues (gaps ignored).
    For each column j, sum n_a*(n_a-1) over residue counts and
    normalize by N*(N-1)*L.
    """
    N = len(seqs)
    if N < 2:
        return 1.0
    L = len(seqs[0])
    total = 0
    for j in range(L):
        counts = Counter(s[j] for s in seqs if s[j] != '-')
        total += sum(n * (n - 1) for n in counts.values())
    return float(total) / (N * (N - 1) * L)

# ---------------- New, cheap indel-shape metrics ----------------

def gap_open_rate(seqs):
    """
    Fraction of positions that begin a gap (transition 0->1 in the gap mask).
    Computed across all sequences and positions (excluding first column).
    """
    if not seqs:
        return 0.0
    gap = np.array([[c == '-' for c in s] for s in seqs], dtype=int)
    if gap.shape[1] < 2:
        return 0.0
    opens = ((gap[:, 1:] == 1) & (gap[:, :-1] == 0)).sum()
    total = gap.shape[0] * (gap.shape[1] - 1)
    return float(opens) / float(total)

def gap_run_mean(seqs):
    """Mean run length of '-' across all sequences (0 if no gaps)."""
    if not seqs:
        return 0.0
    runs = []
    for s in seqs:
        r = 0
        for c in s:
            if c == '-':
                r += 1
            else:
                if r > 0:
                    runs.append(r)
                    r = 0
        if r > 0:
            runs.append(r)
    return float(np.mean(runs)) if runs else 0.0

# ---------------- Metric aggregator ----------------

def compute_metrics(seqs):
    """
    Return pruned + enriched metrics:
      Match, Gap, GapAC, GapAC_lag2, LFPR, MeanPairwiseID,
      AlignmentLength, GapOpenRate, GapRunMean
    """
    if not seqs:
        return {
            'Match': 0.0, 'Gap': 0.0, 'GapAC': 0.0, 'GapAC_lag2': 0.0,
            'LFPR': 0.0, 'MeanPairwiseID': 0.0, 'AlignmentLength': 0,
            'GapOpenRate': 0.0, 'GapRunMean': 0.0
        }

    state = build_state_mask(seqs)
    mask_match = (state == 2).astype(float)
    mask_gap   = (state == 1).astype(float)

    mets = {
        'Match':            float(mask_match.mean()),
        'Gap':              float(mask_gap.mean()),
        'GapAC':            gap_adj_corr_lag(seqs, lag=1),
        'GapAC_lag2':       gap_adj_corr_lag(seqs, lag=2),
        'LFPR':             LFPR(mask_match),
        'MeanPairwiseID':   mean_pairwise_id(seqs),
        'AlignmentLength':  len(seqs[0]),
        'GapOpenRate':      gap_open_rate(seqs),
        'GapRunMean':       gap_run_mean(seqs),
    }
    return mets

# ---------------- Directory traversal & CLI ----------------

def process_dir(root, label, keep_families=None):
    recs = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.fasta'):
            continue
        m = re.match(r'^(RF\d{5})', fn)
        if not m:
            continue
        family = m.group(1)
        if keep_families is not None and family not in keep_families:
            continue
        path = os.path.join(root, fn)
        seqs = read_alignment(path)

        # Trim all-gap columns for robustness
        if seqs:
            arr = np.array([list(s) for s in seqs])
            keep = ~(arr == '-').all(axis=0)
            seqs = [''.join(r[keep]) for r in arr]

        rec = compute_metrics(seqs)
        rec.update({'Family': family, 'Source': label, 'Filename': fn})
        recs.append(rec)
    return recs

def main():
    p = argparse.ArgumentParser(description='Compute compact, informative MSA metrics')
    p.add_argument('rfam_dir',    help='Rfam seed alignments')
    p.add_argument('mafft_dir',   help='MAFFT outputs')
    p.add_argument('clustal_dir', help='Clustal outputs')
    p.add_argument('tcoffee_dir', help='T-Coffee outputs')
    p.add_argument('muscle_dir',  help='MUSCLE outputs')
    p.add_argument('-o', '--output', required=True, help='Path to write metrics CSV')
    p.add_argument('--families', help='Optional whitelist file of family accessions (one per line)')
    args = p.parse_args()

    keep = None
    if args.families:
        with open(args.families) as f:
            keep = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(keep)} families from {args.families}")

    all_recs = []
    all_recs += process_dir(args.rfam_dir,    'Rfam',   keep)
    all_recs += process_dir(args.mafft_dir,   'MAFFT',  keep)
    all_recs += process_dir(args.clustal_dir, 'Clustal',keep)
    all_recs += process_dir(args.tcoffee_dir, 'T-Coffee', keep)
    all_recs += process_dir(args.muscle_dir,  'Muscle', keep)

    df = pd.DataFrame.from_records(all_recs)
    cols = [
        'Family','Source','Filename',
        'Match','Gap','GapAC','GapAC_lag2','LFPR',
        'MeanPairwiseID','AlignmentLength','GapOpenRate','GapRunMean'
    ]
    df.to_csv(args.output, index=False, columns=cols, float_format='%.6f')
    print(f"✔ Wrote {len(df)} rows to {args.output}")

if __name__ == '__main__':
    main()
