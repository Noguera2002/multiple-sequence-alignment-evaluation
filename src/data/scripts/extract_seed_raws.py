#!/usr/bin/env python3
"""
extract_seed_raws.py

Parse Rfam.seed (Stockholm) in the current directory, strip gaps,
and write one raw (ungapped) FASTA per family into ./RawSeqs/.

Handles non-UTF8 bytes by using latin-1 decoding.
"""

import os
import re

def extract_raw_from_stockholm(stockholm_path, outdir="RawSeqs"):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    seq_line_re = re.compile(r"^(\S+)\s+([AUGCaugc\.-]+)$")
    current_block = []

    # Open with latin-1 so every byte maps to a character
    with open(stockholm_path, encoding="latin-1") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "//":
                _process_block(current_block, outdir, seq_line_re)
                current_block = []
            else:
                current_block.append(line)

        # process any trailing block
        if current_block:
            _process_block(current_block, outdir, seq_line_re)


def _process_block(lines, outdir, seq_re):
    # 1) Find accession
    acc = None
    for L in lines:
        if L.startswith("#=GF AC"):
            parts = L.split()
            if len(parts) >= 3:
                acc = parts[2]
            break
    if not acc:
        return

    # 2) Collect seqID + gapped sequence
    seqs = []
    for L in lines:
        if L.startswith("#"):
            continue
        m = seq_re.match(L)
        if m:
            seq_id, gapped = m.groups()
            # strip gaps, convert to uppercase RNA
            raw = gapped.replace("-", "").replace(".", "").upper().replace("T", "U")
            seqs.append((seq_id, raw))

    if not seqs:
        return

    # 3) Write to FASTA
    outpath = os.path.join(outdir, f"{acc}.fa")
    with open(outpath, "w") as out:
        for seq_id, raw in seqs:
            out.write(f">{seq_id}\n{raw}\n")
    print(f"Wrote {len(seqs)} sequences to {outpath}")


if __name__ == "__main__":
    extract_raw_from_stockholm("Rfam.seed", outdir="RawSeqs")
