#!/usr/bin/env python3
"""
compute_sp_alltools.py

Compute Sum-of-Pairs (SP) scores of each aligner MSA against the Rfam reference,
for all families present in the Rfam directory, optionally filtered.

Usage:
  python3 compute_sp_alltools.py \
    --rfam    Rfam/ \
    --clustal Clustal/ \
    --mafft   MAFFT/ \
    --tcoffee T-coffee/ \
    --muscle  Muscle/ \
    --out     all_sp_scores.csv \
    [--families keep_families.txt]
"""
import os
import sys
import glob
import argparse
import pandas as pd
from Bio import AlignIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_residue_index_map(aln):
    """Map each alignment column to residue index (or None for gaps)."""
    mapping = []
    for rec in aln:
        seq = str(rec.seq)
        resi = 0
        cols = []
        for c in seq:
            if c == '-':
                cols.append(None)
            else:
                cols.append(resi)
                resi += 1
        mapping.append(cols)
    return mapping

def get_test_residx_to_col(aln):
    """Map each residue index to its column position in the test alignment."""
    mapping = []
    for rec in aln:
        seq = str(rec.seq)
        idx2col = {}
        residx = 0
        for pos, c in enumerate(seq):
            if c != '-':
                idx2col[residx] = pos
                residx += 1
        mapping.append(idx2col)
    return mapping

def sum_of_pairs_score(ref, test):
    """
    SP = (# of residue pairs co-aligned in both ref & test)
         / (# of residue pairs in ref)
    """
    if len(ref) != len(test):
        raise ValueError("Ref/test have different numbers of sequences")
    n = len(ref)
    ref_map  = get_residue_index_map(ref)
    test_map = get_test_residx_to_col(test)
    L        = ref.get_alignment_length()
    N_ref    = 0
    N_corr   = 0

    # count pairs in reference and in test
    for c in range(L):
        seq_idxs = [i for i in range(n) if ref_map[i][c] is not None]
        m = len(seq_idxs)
        if m < 2:
            continue
        # all pairs in this column of the reference
        N_ref += m * (m - 1) // 2

        # for those residues, see how many still co-occur in same test column
        test_cols = []
        for i in seq_idxs:
            resi = ref_map[i][c]
            col_i = test_map[i].get(resi)
            if col_i is not None:
                test_cols.append(col_i)
        if test_cols:
            counts = Counter(test_cols)
            for cnt in counts.values():
                if cnt > 1:
                    N_corr += cnt * (cnt - 1) // 2

    return (N_corr / N_ref) if N_ref else 0.0

def score_one(task):
    fam, tool, ref_path, test_path = task
    try:
        ref  = AlignIO.read(ref_path,  "fasta")
        test = AlignIO.read(test_path, "fasta")
        sp   = sum_of_pairs_score(ref, test)
    except Exception as e:
        print(f"[ERROR] {fam} / {tool}: {e}", file=sys.stderr)
        sp = None
    return {"family": fam, "tool": tool, "SP": sp}

def find_test_file(dirpath, fam, tool):
    """
    Look for files named <fam>_<tool>*.fasta or .fa
    """
    patterns = [
        os.path.join(dirpath, f"{fam}_{tool}*.fasta"),
        os.path.join(dirpath, f"{fam}_{tool}*.fa")
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rfam',    required=True, help='Rfam reference MSA directory')
    parser.add_argument('--clustal', required=True, help='Clustal MSA directory')
    parser.add_argument('--mafft',   required=True, help='MAFFT MSA directory')
    parser.add_argument('--tcoffee', required=True, help='T-coffee MSA directory')
    parser.add_argument('--muscle',  required=True, help='Muscle MSA directory')
    parser.add_argument('--out',     required=True, help='Output CSV path')
    parser.add_argument('--families', help='Optional whitelist file of family accessions to include (one per line)')
    args = parser.parse_args()

    keep = None
    if args.families:
        with open(args.families) as f:
            keep = set(line.strip() for line in f if line.strip())
        print(f"Filtering to {len(keep)} families from {args.families}")

    # gather all reference files
    ref_files = [
        f for f in os.listdir(args.rfam)
        if f.endswith(('.fasta', '.fa'))
    ]
    tasks = []
    tool_dirs = {
        'clustal': args.clustal,
        'mafft':   args.mafft,
        'tcoffee': args.tcoffee,   # matches RFxxxxx_tcoffee.fasta
        'muscle':  args.muscle
    }

    for fname in sorted(ref_files):
        fam = os.path.splitext(fname)[0].split('_')[0]
        if keep is not None and fam not in keep:
            continue  # skip filtered family
        ref_path = os.path.join(args.rfam, fname)
        for tool, d in tool_dirs.items():
            test_path = find_test_file(d, fam, tool)
            if test_path:
                tasks.append((fam, tool, ref_path, test_path))
            else:
                print(f"[WARN] Missing {tool} for family {fam}", file=sys.stderr)

    # parallel SP computation
    results = []
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(score_one, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())

    # save
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == '__main__':
    main()
