import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Build SP label (SP only) from sp_struct.csv")
    ap.add_argument("--sp-input", default="src/data/processed/sp_struct.csv")
    ap.add_argument("--out", default="src/data/processed/sp_for_train_SPall.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.sp_input)

    # Be lenient about column case: accept 'SP' or 'sp'
    cols_lower = {c.lower(): c for c in df.columns}
    if "sp" not in cols_lower:
        raise SystemExit(f"Required column 'SP' not found. Available columns: {list(df.columns)}")
    sp_col = cols_lower["sp"]

    # Coerce to numeric in case the CSV has stray strings
    df["sp"] = pd.to_numeric(df[sp_col], errors="coerce")

    # Ensure required id columns exist
    for needed in ("family", "tool"):
        if needed not in df.columns:
            raise SystemExit(f"Required column '{needed}' not found. Available columns: {list(df.columns)}")

    out = df[["family", "tool", "sp"]].dropna(subset=["sp"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✔ Wrote {len(out)} rows to {args.out} using only column: {sp_col}")

if __name__ == "__main__":
    main()
