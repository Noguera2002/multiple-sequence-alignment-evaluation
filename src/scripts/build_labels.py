#!/usr/bin/env python3
import argparse, pandas as pd, os

def main():
    ap = argparse.ArgumentParser(description="Build composite SP label from sp_struct.csv")
    ap.add_argument("--sp-input", default="src/data/processed/sp_struct.csv")
    ap.add_argument("--out", default="src/data/processed/sp_for_train_SPall.csv")
    ap.add_argument("--use", default="SP,SP_stem,SP_loop",
                    help="Comma-separated scores to average (default: SP,SP_stem,SP_loop)")
    args = ap.parse_args()

    use_cols = [c.strip() for c in args.use.split(",") if c.strip()]
    df = pd.read_csv(args.sp_input)

    avail = [c for c in use_cols if c in df.columns]
    if not avail:
        raise SystemExit(f"No requested columns found. Requested={use_cols}, available={list(df.columns)}")

    df["sp"] = df[avail].mean(axis=1, skipna=True)
    out = df[["family","tool","sp"]].dropna(subset=["sp"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✔ Wrote {len(out)} rows to {args.out} using: {avail}")

if __name__ == "__main__":
    main()
