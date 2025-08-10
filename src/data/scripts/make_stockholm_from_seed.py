#!/usr/bin/env python3
"""
Split Rfam.seed (multi-family Stockholm) into per-family .sto files.

Reads/writes in *binary* to avoid Unicode issues.

Usage:
  python src/data/scripts/make_stockholm_from_seed.py \
    --seed src/data/raw/Rfam.seed \
    --outdir src/data/stockholm
"""
import argparse, os, re
from pathlib import Path

ACC_RE = re.compile(br"RF\d{5}")

def split_rfam_seed(seed_path: Path, outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)

    acc = None           # current accession (bytes, e.g. b'RF00001')
    buf = []             # list of lines (bytes) for current record
    count = 0

    def flush():
        nonlocal acc, buf, count
        if acc and buf:
            # Write the record, ensuring it ends with '//' line
            # (buf excludes the '//' trigger; we add it)
            name = f"{acc.decode('ascii')}.sto"
            k = 1
            while (outdir / name).exists():
                k += 1
                name = f"{acc.decode('ascii')}_{k}.sto"
            with open(outdir / name, "wb") as w:
                w.writelines(buf)
                w.write(b"//\n")
            count += 1
        acc = None
        buf = []

    with open(seed_path, "rb") as f:
        for line in f:
            # End of one Stockholm record
            if line.strip() == b"//":
                flush()
                continue

            # Capture accession when seen
            if line.startswith(b"#=GF AC"):
                # e.g. b"#=GF AC   RF00001"
                m = ACC_RE.search(line)
                if m:
                    acc = m.group(0)
            buf.append(line)

    # Handle file not ending with "//"
    if buf:
        flush()

    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="src/data/raw/Rfam.seed",
                    help="Path to Rfam.seed (multi-family Stockholm)")
    ap.add_argument("--outdir", default="src/data/stockholm",
                    help="Directory to write per-family .sto files")
    args = ap.parse_args()

    seed = Path(args.seed)
    outdir = Path(args.outdir)
    if not seed.exists():
        raise SystemExit(f"Input not found: {seed}")

    n = split_rfam_seed(seed, outdir)
    print(f"✔ Wrote {n} Stockholm files to {outdir}")

if __name__ == "__main__":
    main()
