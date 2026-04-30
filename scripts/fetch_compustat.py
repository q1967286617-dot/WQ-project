"""
Fetch Compustat quarterly fundamentals from WRDS and merge into existing panel.

Usage:
    python scripts/fetch_compustat.py

Output: data/processed/{train,val,test}_with_fundamentals.parquet
"""

import wrds
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
SPLITS   = ["train", "val", "test"]
FEATURE_COLS = ["roe", "payout_ratio", "div_coverage", "dvpsxq", "leverage", "profit_margin"]

# ── 1. Connect ────────────────────────────────────────────────────────────────
print("Connecting to WRDS...")
db = wrds.Connection()

# ── 2. Link table ─────────────────────────────────────────────────────────────
print("Downloading link table...")
link = db.raw_sql("""
    SELECT gvkey,
           lpermno  AS permno,
           linktype, linkprim,
           linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
      AND lpermno IS NOT NULL
""")
link["linkdt"]    = pd.to_datetime(link["linkdt"])
link["linkenddt"] = pd.to_datetime(link["linkenddt"].fillna("2030-12-31"))
link["permno"]    = link["permno"].astype("Int64")   # nullable int
print(f"  Link table: {len(link):,} rows")

# ── 3. Compustat quarterly ────────────────────────────────────────────────────
print("Downloading Compustat fundq (2009-2024)...")
compq = db.raw_sql("""
    SELECT gvkey, datadate, rdq,
           niq, ceqq, dvpsxq, atq, dlttq, cshoq, saleq
    FROM comp.fundq
    WHERE indfmt  = 'INDL'
      AND datafmt = 'STD'
      AND popsrc  = 'D'
      AND consol  = 'C'
      AND datadate >= '2009-01-01'
      AND datadate <= '2024-12-31'
      AND niq  IS NOT NULL
      AND ceqq IS NOT NULL
""", date_cols=["datadate", "rdq"])
db.close()
print(f"  Compustat rows: {len(compq):,}")

# ── 4. Build features ─────────────────────────────────────────────────────────
print("Computing features...")
df = compq.copy()

# force numeric (some cols may come as object)
num_cols = ["niq", "ceqq", "dvpsxq", "atq", "dlttq", "cshoq", "saleq"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ROE
df["roe"] = (df["niq"] / df["ceqq"].where(df["ceqq"].abs() > 1e-6)).clip(-2, 2)

# total dividends paid ($ millions) = dvps * shares_outstanding
df["_div_total"] = df["dvpsxq"].fillna(0) * df["cshoq"].fillna(0)

# payout ratio (only when earnings positive)
df["payout_ratio"] = (
    df["_div_total"] / df["niq"].where(df["niq"] > 1e-6)
).clip(0, 5)

# dividend coverage (net income / dividends; only when dividends > 0)
df["div_coverage"] = (
    df["niq"] / df["_div_total"].where(df["_div_total"] > 1e-6)
).clip(-10, 50)

# dividends per share (direct)
df["dvpsxq"] = df["dvpsxq"].fillna(0)

# leverage
df["leverage"] = (
    df["dlttq"] / df["atq"].where(df["atq"].abs() > 1e-6)
).clip(0, 5)

# profit margin
df["profit_margin"] = (
    df["niq"] / df["saleq"].where(df["saleq"] > 1e-6)
).clip(-2, 2)

# availability date (actual report date + 5 days buffer; fallback: quarter-end + 45 days)
df["avail_date"] = df["rdq"] + pd.Timedelta(days=5)
missing_rdq = df["rdq"].isna()
df.loc[missing_rdq, "avail_date"] = df.loc[missing_rdq, "datadate"] + pd.Timedelta(days=45)

df = df[["gvkey", "avail_date"] + FEATURE_COLS].dropna(subset=["avail_date"])
df = df.sort_values(["gvkey", "avail_date"]).reset_index(drop=True)
print(f"  Feature rows: {len(df):,}")

# ── 5. Link gvkey → permno ────────────────────────────────────────────────────
print("Linking to PERMNO...")
df_linked = df.merge(link, on="gvkey", how="left")

# keep only rows where avail_date is within the valid link window
df_linked = df_linked[
    df_linked["permno"].notna() &
    (df_linked["avail_date"] >= df_linked["linkdt"]) &
    (df_linked["avail_date"] <= df_linked["linkenddt"])
].copy()

# one row per (permno, avail_date)
df_linked = (
    df_linked
    .sort_values(["permno", "avail_date"])
    .drop_duplicates(subset=["permno", "avail_date"], keep="last")
    .reset_index(drop=True)
)

# ensure permno is plain int64 (no nullable) for merge compatibility
df_linked["permno"] = df_linked["permno"].astype("int64")
print(f"  Unique PERMNOs linked: {df_linked['permno'].nunique():,}")

merge_cols = df_linked[["permno", "avail_date"] + FEATURE_COLS].copy()

# ── 6. Merge into each panel split ───────────────────────────────────────────
print("Merging into panel splits...")

for split in SPLITS:
    in_path  = DATA_DIR / f"{split}.parquet"
    out_path = DATA_DIR / f"{split}_with_fundamentals.parquet"

    panel = pd.read_parquet(in_path)
    panel["DlyCalDt"] = pd.to_datetime(panel["DlyCalDt"])
    panel["PERMNO"]   = panel["PERMNO"].astype("int64")   # match dtype
    # merge_asof requires the on-key (DlyCalDt) to be globally sorted
    panel = panel.sort_values("DlyCalDt").reset_index(drop=True)

    # rename merge keys to match panel column names
    right = merge_cols.rename(columns={"permno": "PERMNO", "avail_date": "DlyCalDt"})
    right = right.sort_values("DlyCalDt").reset_index(drop=True)

    panel = pd.merge_asof(
        panel, right,
        on="DlyCalDt",
        by="PERMNO",
        direction="backward"   # only use already-released fundamentals
    )

    n_covered = panel["roe"].notna().sum()
    pct = 100 * n_covered / len(panel)
    print(f"  {split:5s}: {len(panel):>8,} rows | fundamentals coverage {pct:.1f}%")

    panel.to_parquet(out_path, index=False)
    print(f"         saved → {out_path}")

print("\nAll done.")
