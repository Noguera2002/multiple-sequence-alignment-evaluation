#!/usr/bin/env python3
import pandas as pd

def main():
    # load
    df_met = pd.read_csv("all_metrics.csv")         # has Family, Source, 9 metrics…
    df_sp  = pd.read_csv("all_sp_scores.csv")       # has family, tool, SP

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
    print(f"→ merged.csv: {len(df)} rows")
    df.to_csv("merged.csv", index=False)

if __name__=="__main__":
    main()
