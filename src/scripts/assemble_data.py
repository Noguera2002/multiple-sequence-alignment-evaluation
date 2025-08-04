#!/usr/bin/env python3
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, help='CSV of family-level metrics (e.g., all_metrics.csv)')
    parser.add_argument('--sp', required=True, help='CSV of SP scores (e.g., all_sp_scores.csv)')
    parser.add_argument('--out', required=True, help='Output merged CSV path')
    args = parser.parse_args()

    df_met = pd.read_csv(args.metrics)         # has Family, Source, 9 metrics…
    df_sp  = pd.read_csv(args.sp)              # has family, tool, SP

    # normalize names
    df_met = df_met.rename(columns={"Family":"family","Source":"tool"})
    df_sp  = df_sp .rename(columns={"SP":"sp"})
    df_met['tool'] = df_met['tool'].str.lower().str.replace('-', '').str.replace('_','')
    df_sp ['tool'] = df_sp ['tool'].str.lower()

    # drop any “rfam” if present
    df_met = df_met[df_met.tool != 'rfam']
    df_sp  = df_sp [df_sp .tool != 'rfam']

    # merge
    df = pd.merge(df_met, df_sp, on=['family','tool'], how='inner')
    print(f"→ {args.out}: {len(df)} rows")
    df.to_csv(args.out, index=False)

if __name__=="__main__":
    main()
