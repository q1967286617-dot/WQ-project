"""
Fetch market-level macro features and merge into processed splits.

Data sources:
  - crsp.dsi  (WRDS, existing subscription): market returns → momentum & realized vol
  - FRED API  (free, no account needed):      VIX, 10Y Treasury yield

Features added to each split parquet:
  mkt_ret_5d   : value-weighted market 5-day cumulative return
  mkt_ret_21d  : value-weighted market 21-day cumulative return
  mkt_ret_63d  : value-weighted market 63-day cumulative return
  mkt_vol_21d  : 21-day realized volatility of market (annualized)
  mkt_vol_63d  : 63-day realized volatility of market (annualized)
  vix          : CBOE VIX closing level (from FRED VIXCLS)
  dgs10        : 10-year Treasury constant maturity yield % (from FRED DGS10)

Usage:
    python scripts/fetch_market_macro.py
    python scripts/fetch_market_macro.py --no_fred   # skip VIX / DGS10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path("data/processed")
SPLITS   = ["train", "val", "test"]

# Buffer before earliest split date so rolling windows are fully warmed up
FETCH_START   = "2009-01-01"
FETCH_END     = "2025-01-01"
MACRO_CACHE   = Path("data/processed/macro_cache.parquet")  # local cache avoids re-fetching

MACRO_COLS = [
    "mkt_ret_5d", "mkt_ret_21d", "mkt_ret_63d",
    "mkt_vol_21d", "mkt_vol_63d",
    "vix", "dgs10",
]


# ── CRSP dsi ────────────────────────────────────────────────────────────────

def fetch_crsp_dsi() -> pd.DataFrame:
    import wrds
    print("Connecting to WRDS...")
    db = wrds.Connection()
    print("Fetching crsp.dsi...")
    raw = db.raw_sql(f"""
        SELECT date, vwretd, sprtrn
        FROM crsp.dsi
        WHERE date BETWEEN '{FETCH_START}' AND '{FETCH_END}'
        ORDER BY date
    """)
    db.close()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    raw["vwretd"] = pd.to_numeric(raw["vwretd"], errors="coerce")
    raw["sprtrn"] = pd.to_numeric(raw["sprtrn"], errors="coerce")
    print(f"  crsp.dsi rows: {len(raw):,}  ({raw['date'].min().date()} ~ {raw['date'].max().date()})")
    return raw


def build_market_features(dsi: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling momentum and realized-vol features from daily market returns."""
    d = dsi.copy().set_index("date").sort_index()
    ret = d["vwretd"].fillna(0.0)

    # Cumulative log-return over window (≈ sum of daily returns for short windows)
    for w in [5, 21, 63]:
        d[f"mkt_ret_{w}d"] = ret.rolling(w).sum()

    # Realized vol (annualized): std of daily returns × sqrt(252)
    for w in [21, 63]:
        d[f"mkt_vol_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

    keep = [c for c in d.columns if c.startswith("mkt_")]
    out = d[keep].reset_index().rename(columns={"date": "DlyCalDt"})
    return out


# ── FRED API ────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, col_name: str) -> pd.DataFrame:
    """Download a single FRED series via the public JSON API (no API key needed)."""
    import urllib.request, json

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&vintage_date={FETCH_END}"
    )
    print(f"  Fetching FRED {series_id} ...")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            from io import StringIO
            text = resp.read().decode("utf-8")
        df = pd.read_csv(StringIO(text))
        # FRED CSV header is always the first column for date
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df = df.rename(columns={date_col: "DlyCalDt", val_col: col_name})
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
        df = df.dropna(subset=["DlyCalDt"])
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=[col_name])
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=[col_name])
        print(f"    {col_name}: {len(df):,} rows  ({df['DlyCalDt'].min().date()} ~ {df['DlyCalDt'].max().date()})")
        return df[["DlyCalDt", col_name]]
    except Exception as e:
        print(f"    WARNING: could not fetch {series_id} ({e}); column will be NaN")
        return pd.DataFrame(columns=["DlyCalDt", col_name])


def fetch_fred_macro() -> pd.DataFrame:
    vix   = fetch_fred_series("VIXCLS", "vix")
    dgs10 = fetch_fred_series("DGS10",  "dgs10")
    if vix.empty and dgs10.empty:
        return pd.DataFrame(columns=["DlyCalDt", "vix", "dgs10"])
    if vix.empty:
        return dgs10
    if dgs10.empty:
        return vix
    merged = vix.merge(dgs10, on="DlyCalDt", how="outer").sort_values("DlyCalDt")
    # FRED series are weekly/business-daily; forward-fill weekends & holidays
    merged = merged.set_index("DlyCalDt").resample("B").last().ffill().reset_index()
    return merged


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_fred", action="store_true",
                    help="Skip VIX and DGS10 from FRED (use only crsp.dsi)")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore local cache and re-fetch from WRDS/FRED")
    args = ap.parse_args()

    # 1. Market features — use cache if available and not forced
    if MACRO_CACHE.exists() and not getattr(args, "refresh", False):
        print(f"Loading macro cache: {MACRO_CACHE}")
        macro = pd.read_parquet(MACRO_CACHE)
        macro["DlyCalDt"] = pd.to_datetime(macro["DlyCalDt"])
    else:
        dsi = fetch_crsp_dsi()
        mkt_df = build_market_features(dsi)
        print(f"Market features: {mkt_df.shape}  cols={[c for c in mkt_df.columns if c!='DlyCalDt']}")

        # 2. FRED macro (optional)
        if args.no_fred:
            fred_df = pd.DataFrame(columns=["DlyCalDt", "vix", "dgs10"])
            print("Skipping FRED data (--no_fred)")
        else:
            fred_df = fetch_fred_macro()

        # 3. Combine macro table
        if fred_df.empty or len(fred_df.columns) <= 1:
            macro = mkt_df.copy()
        else:
            macro = mkt_df.merge(fred_df, on="DlyCalDt", how="left")

        macro["DlyCalDt"] = pd.to_datetime(macro["DlyCalDt"])
        macro = macro.sort_values("DlyCalDt").reset_index(drop=True)
        macro.to_parquet(MACRO_CACHE, index=False)
        print(f"Macro cache saved: {MACRO_CACHE}")

    available_macro_cols = [c for c in MACRO_COLS if c in macro.columns]
    print(f"\nMacro table: {len(macro):,} rows, features: {available_macro_cols}")

    # 4. Merge into each split
    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"  {split}: not found at {path}, skipping")
            continue

        df = pd.read_parquet(path)
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])

        # Drop any stale macro cols before re-merge
        drop_cols = [c for c in available_macro_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        before = df.shape
        df = df.merge(macro[["DlyCalDt"] + available_macro_cols],
                      on="DlyCalDt", how="left")

        # Forward-fill within each PERMNO for the rare case where market data
        # has a gap on a given trading day (e.g. public holidays)
        df = df.sort_values(["PERMNO", "DlyCalDt"])
        df[available_macro_cols] = df.groupby("PERMNO")[available_macro_cols].ffill()

        null_rates = df[available_macro_cols].isna().mean()
        print(f"  {split}: {before} → {df.shape}  null rates: "
              + "  ".join(f"{c}={null_rates[c]:.1%}" for c in available_macro_cols))

        df.to_parquet(path, index=False)
        print(f"    saved: {path}")

    print("\nDone. Add these to configs/model.yaml → num_cols:")
    for c in available_macro_cols:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
