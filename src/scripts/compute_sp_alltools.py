#!/usr/bin/env python3
"""
compute_sp_alltools.py (Stockholm-as-reference, BP_compat removed)

- Uses per-family Rfam Stockholm (.sto/.stockholm/.stk) as the reference MSA.
- Computes:
    SP        : classic sum-of-pairs vs reference
    SP_stem   : SP restricted to base-paired columns (from SS_cons)
    SP_loop   : SP restricted to unpaired columns

Outputs CSV with columns:
  family, tool, SP, SP_stem, SP_loop

Requirements: biopython, pandas
"""

import os, glob, argparse, warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

FA_EXTS = (".fa", ".fasta", ".aln", ".afa")

# ---------------- I/O helpers ----------------

def _sanitize_id(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    s = s.split()[0]
    if s.endswith("/1") or s.endswith("/2"):
        s = s[:-2]
    return s

def _find_first_file(d, fam):
    d = Path(d)
    for ext in FA_EXTS:
        for p in sorted(d.glob(f"{fam}*{ext}")):
            return str(p)
    for p in sorted(d.glob(f"*{fam}*")):
        if p.suffix.lower() in FA_EXTS:
            return str(p)
    return None

def _normalize_gaps(aln):
    """Return a copy with '.' converted to '-' so gaps are consistent."""
    recs = []
    for rec in aln:
        s = str(rec.seq).replace('.', '-')
        recs.append(SeqRecord(Seq(s), id=rec.id, description=rec.description))
    return MultipleSeqAlignment(recs)

def _align_intersection_by_id(ref_aln, test_aln):
    """Return (ref_sub, test_sub) aligned on the intersection of IDs, ref order."""
    ref_ids = [_sanitize_id(r.id) for r in ref_aln]
    tmap = { _sanitize_id(r.id): r for r in test_aln }
    shared = [rid for rid in ref_ids if rid in tmap]
    if len(shared) < 2:
        return None, None
    ref_recs, test_recs = [], []
    for rid in shared:
        rrec = next(r for r in ref_aln if _sanitize_id(r.id) == rid)
        trec = tmap[rid]
        ref_recs.append(rrec)
        test_recs.append(trec)
    return MultipleSeqAlignment(ref_recs), MultipleSeqAlignment(test_recs)

# ---------------- SP core ----------------

def _ref_col_to_res_idx(aln):
    """Map each ref column to residue index (or None for gaps), per sequence."""
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

def _test_res_idx_to_col(aln):
    """Map each residue index to column position in the test alignment, per sequence."""
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

def sum_of_pairs_score(ref, test, col_mask=None):
    """
    SP = (# residue pairs co-aligned in both ref & test) / (# residue pairs in ref subset)
    Assumes ref and test have the same sequences in the same order.
    """
    if len(ref) != len(test):
        raise ValueError("Ref/test have different numbers of sequences")
    n = len(ref)
    L = ref.get_alignment_length()
    ref_map  = _ref_col_to_res_idx(ref)
    test_map = _test_res_idx_to_col(test)
    if col_mask is None:
        col_mask = [True]*L
    N_ref = 0
    N_corr = 0
    for c in range(L):
        if not col_mask[c]:
            continue
        seq_idxs = [i for i in range(n) if ref_map[i][c] is not None]
        m = len(seq_idxs)
        if m < 2:
            continue
        N_ref += m * (m - 1) // 2
        tcols = []
        for i in seq_idxs:
            resi = ref_map[i][c]
            col_i = test_map[i].get(resi)
            if col_i is not None:
                tcols.append(col_i)
        if tcols:
            counts = Counter(tcols)
            for cnt in counts.values():
                if cnt > 1:
                    N_corr += cnt * (cnt - 1) // 2
    return (N_corr / N_ref) if N_ref else 0.0

# ---------------- SS_cons parsing ----------------

BR_OPENS = "([{<"
BR_CLOSE = ")]}>"
OPEN_TO_CLOSE = dict(zip(BR_OPENS, BR_CLOSE))
CLOSE_TO_OPEN = dict(zip(BR_CLOSE, BR_OPENS))

def read_ss_cons(stockholm_path):
    """Extract concatenated SS_cons string from a per-family Stockholm file (binary-safe)."""
    ss_lines = []
    try:
        with open(stockholm_path, "rb") as f:
            for raw in f:
                if raw.startswith(b"#=GC SS_cons"):
                    parts = raw.rstrip().split(None, 2)
                    if len(parts) >= 3:
                        ss_lines.append(parts[2].decode("ascii", errors="ignore"))
    except Exception as e:
        warnings.warn(f"Failed to read SS_cons from {stockholm_path}: {e}")
        return None
    return "".join(ss_lines) if ss_lines else None

def paired_columns_from_ss(ss):
    """Return (stem_mask, pairs) from SS_cons; ignores pseudoknot letters."""
    if not ss:
        return None, []
    L = len(ss)
    stem_mask = [False]*L
    pairs = []
    stacks = {op:[] for op in BR_OPENS}
    for i,ch in enumerate(ss):
        if ch in OPEN_TO_CLOSE:
            stacks[ch].append(i)
        elif ch in CLOSE_TO_OPEN:
            op = CLOSE_TO_OPEN[ch]
            if stacks[op]:
                j = stacks[op].pop()
                a, b = j, i
                pairs.append((a,b))
                stem_mask[a] = stem_mask[b] = True
    return stem_mask, pairs

# ---------------- worker ----------------

def score_one(task):
    fam, sto_path, tool, test_path = task
    try:
        ref = AlignIO.read(sto_path, "stockholm")
        ref = _normalize_gaps(ref)
    except Exception as e:
        return {"family": fam, "tool": tool, "SP": None, "SP_stem": None, "SP_loop": None, "error": f"read_ref_fail:{e}"}

    try:
        test = AlignIO.read(test_path, "fasta")
        test = _normalize_gaps(test)
    except Exception as e:
        return {"family": fam, "tool": tool, "SP": None, "SP_stem": None, "SP_loop": None, "error": f"read_test_fail:{e}"}

    ref2, test2 = _align_intersection_by_id(ref, test)
    if ref2 is None:
        return {"family": fam, "tool": tool, "SP": None, "SP_stem": None, "SP_loop": None, "error": "no_shared_ids"}

    ss = read_ss_cons(sto_path)
    ss_mask, _ss_pairs = paired_columns_from_ss(ss) if ss else (None, [])

    try:
        sp_all = sum_of_pairs_score(ref2, test2)
        if ss_mask and len(ss_mask) == ref2.get_alignment_length():
            sp_stem = sum_of_pairs_score(ref2, test2, col_mask=ss_mask)
            loop_mask = [not x for x in ss_mask]
            sp_loop = sum_of_pairs_score(ref2, test2, col_mask=loop_mask)
        else:
            sp_stem = None
            sp_loop = None

        return {"family": fam, "tool": tool, "SP": sp_all, "SP_stem": sp_stem, "SP_loop": sp_loop}
    except Exception as e:
        return {"family": fam, "tool": tool, "SP": None, "SP_stem": None, "SP_loop": None, "error": f"sp_fail:{e}"}

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Compute SP (and stem/loop variants) using Rfam Stockholm as reference.")
    ap.add_argument("--ref-stockholm", required=True, help="Dir with per-family RFxxxxx.sto/.stockholm/.stk files")
    ap.add_argument("--clustal", required=True, help="Directory of Clustal alignments")
    ap.add_argument("--mafft", required=True, help="Directory of MAFFT alignments")
    ap.add_argument("--tcoffee", required=True, help="Directory of T-Coffee alignments")
    ap.add_argument("--muscle", required=True, help="Directory of MUSCLE alignments")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--families", help="Optional file with RFxxxxx accessions to include (one per line)")
    args = ap.parse_args()

    keep = None
    if args.families:
        with open(args.families) as f:
            keep = {ln.strip().split()[0] for ln in f if ln.strip()}

    # Discover families from the Stockholm dir
    sto_files = []
    for p in sorted(Path(args.ref_stockholm).glob("RF*.*")):
        if p.suffix.lower() in (".sto", ".stockholm", ".stk"):
            fam = p.stem.split("_")[0]
            sto_files.append((fam, str(p)))
    if not sto_files:
        raise SystemExit(f"No RFxxxxx .sto/.stockholm/.stk files found in {args.ref_stockholm}")

    tool_dirs = {"clustal": args.clustal, "mafft": args.mafft, "tcoffee": args.tcoffee, "muscle": args.muscle}

    tasks = []
    for fam, sto_path in sto_files:
        if keep and fam not in keep:
            continue
        for tool, d in tool_dirs.items():
            test_path = _find_first_file(d, fam)
            if not test_path:
                warnings.warn(f"Missing {tool} for family {fam}")
                continue
            tasks.append((fam, sto_path, tool, test_path))

    if not tasks:
        raise SystemExit("No tasks found. Check tool directories and RFxxxxx file naming.")

    results = []
    with ProcessPoolExecutor() as exe:
        futs = [exe.submit(score_one, t) for t in tasks]
        for fu in as_completed(futs):
            results.append(fu.result())

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"✔ Wrote {len(results)} rows to {args.out}")

if __name__ == "__main__":
    main()
