#!/usr/bin/env python3
import os
import argparse

def count_seqs_in_fasta(path):
    count = 0
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Select families with at least N sequences from RawSeqs"
    )
    parser.add_argument('--rawdir', required=True,
                        help='Directory containing raw per-family FASTA files, e.g., src/data/raw/RawSeqs')
    parser.add_argument('--min-seqs', type=int, default=5,
                        help='Minimum number of sequences required to keep a family (default 5)')
    parser.add_argument('--out', required=True,
                        help='Output whitelist file (one family accession per line)')
    args = parser.parse_args()

    kept = []
    skipped = []
    for fn in sorted(os.listdir(args.rawdir)):
        if not fn.startswith('RF') or not (fn.endswith('.fa') or fn.endswith('.fasta')):
            continue
        path = os.path.join(args.rawdir, fn)
        num = count_seqs_in_fasta(path)
        fam = os.path.splitext(fn)[0]  # RFxxxxx
        if num >= args.min_seqs:
            kept.append(fam)
        else:
            skipped.append((fam, num))
            print(f"Skipping {fn}: only {num} sequences (<{args.min_seqs})")

    with open(args.out, 'w') as out:
        for fam in kept:
            out.write(f"{fam}\n")
    print()
    print(f"Summary:")
    print(f"  Total families examined: {len(kept) + len(skipped)}")
    print(f"  Kept (≥ {args.min_seqs} seqs): {len(kept)}")
    print(f"  Skipped (< {args.min_seqs} seqs): {len(skipped)}")
    if skipped:
        print("  Skipped families (family, sequence count):")
        for fam, num in skipped:
            print(f"    {fam}: {num}")
    print(f"Whitelist written to {args.out}")

if __name__ == '__main__':
    main()
