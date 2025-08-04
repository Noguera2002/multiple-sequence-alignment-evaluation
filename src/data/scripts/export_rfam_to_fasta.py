#!/usr/bin/env python3
"""
export_rfam_to_fasta.py

Parse Rfam.seed (Stockholm) into individual-family FASTA MSAs.
Handles non-UTF8 bytes by using latin-1 decoding.

Usage:
    python3 export_rfam_to_fasta.py --stockholm Rfam.seed --outdir Rfam/
"""

import os
import re
import argparse

def parse_rfam_seed(stockholm_path, max_families=None):
    """
    Parse Rfam.seed (Stockholm) into a dict:
      { accession: [ (seqID, aligned_seq), … ] }
    Stops after max_families if provided.
    """
    fams = {}
    current_block = []
    families_seen = 0

    # Use latin-1 to avoid decode errors on non-UTF8 bytes
    with open(stockholm_path, 'r', encoding='latin-1') as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.strip() == "//":
                if current_block:
                    _process_block(current_block, fams)
                    families_seen += 1
                    current_block = []
                    if max_families is not None and families_seen >= max_families:
                        break
            else:
                current_block.append(line)

        # Process any trailing block (if no final "//")
        if current_block and (max_families is None or families_seen < max_families):
            _process_block(current_block, fams)

    return fams

def _process_block(block_lines, families_dict):
    """
    Given lines of one Stockholm block (no "//"), extract:
      - accession (from "#=GF AC   RFxxxxx")
      - all sequence rows (seqID + gapped sequence)
    Store in families_dict[accession].
    """
    accession = None
    seq_list = []

    # 1) Find accession line
    for line in block_lines:
        if line.startswith("#=GF AC"):
            parts = line.split()
            if len(parts) >= 3:
                accession = parts[2]
            break
    if accession is None:
        return

    # 2) Collect sequence lines (skip any line starting with "#")
    seq_pattern = re.compile(r"^(\S+)\s+([AUGCaugc\.\-\~]+)$")
    for line in block_lines:
        if line.startswith("#"):
            continue
        m = seq_pattern.match(line)
        if m:
            seq_id = m.group(1)
            aligned_seq = m.group(2)
            seq_list.append((seq_id, aligned_seq))

    if seq_list:
        families_dict[accession] = seq_list

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Parse Rfam.seed → per-family FASTA MSAs"
    )
    p.add_argument(
        "--stockholm", "-s",
        required=True,
        help="Path to Rfam.seed (Stockholm format)"
    )
    p.add_argument(
        "--outdir", "-o",
        default="Rfam",
        help="Directory to write per-family FASTA files"
    )
    p.add_argument(
        "--max-families", "-n",
        type=int, default=None,
        help="Stop after this many families (for testing)"
    )
    args = p.parse_args()

    # Parse
    print(f"Parsing {args.stockholm} …")
    fams = parse_rfam_seed(args.stockholm, max_families=args.max_families)
    print(f"  Parsed {len(fams)} families.")

    # Prepare output dir
    os.makedirs(args.outdir, exist_ok=True)

    # Write FASTA per family
    for acc, seq_list in fams.items():
        outpath = os.path.join(args.outdir, f"{acc}_bench.fasta")
        with open(outpath, "w") as outfh:
            for seq_id, aligned_seq in seq_list:
                outfh.write(f">{seq_id}\n")
                # wrap at 80 chars
                for i in range(0, len(aligned_seq), 80):
                    outfh.write(aligned_seq[i:i+80] + "\n")
    print(f"Wrote {len(fams)} FASTA MSAs into {args.outdir}/")
