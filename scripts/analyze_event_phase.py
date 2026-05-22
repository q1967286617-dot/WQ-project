"""
Event Phase Attribution
-----------------------
Decomposes each trade's holding-period return into:
  Phase 1: entry_date → DCLRDT-1  (pre-announcement drift)
  Phase 2: DCLRDT    → exit_date   (declaration day + post)

Only trades where a DCLRDT falls strictly inside the holding window are
analyzed (these are the y_entry=1 or late-match trades).

Usage:
  python scripts/analyze_event_phase.py --run hpo_v9_d6_lr005_sub09
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_data(run_name: str):
    trades_path = os.path.join(ROOT, "outputs", "runs", run_name, "backtest", "trades.csv")
    tableA_path = os.path.join(ROOT, "data", "raw", "tableA.csv")
    tableB_path = os.path.join(ROOT, "data", "raw", "tableB.parquet")

    trades = pd.read_csv(trades_path, parse_dates=["signal_date", "entry_date", "exit_date"])
    A = pd.read_csv(tableA_path, parse_dates=["DCLRDT", "EXDT"])
    A = A.dropna(subset=["DCLRDT"]).rename(columns={"PERMNO": "permno"})
    A["permno"] = A["permno"].astype(int)

    B = pd.read_parquet(tableB_path, columns=["PERMNO", "DlyCalDt", "DlyRet"])
    B = B.rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "ret"})
    B["permno"] = B["permno"].astype(int)
    B["date"] = pd.to_datetime(B["date"])
    B = B.dropna(subset=["ret"]).sort_values(["permno", "date"])

    return trades, A, B


def match_declaration(trades: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """For each trade find the first DCLRDT in (entry_date, exit_date]."""
    # Merge all dividends onto trades by permno, then filter to window
    merged = trades.merge(
        A[["permno", "DCLRDT", "EXDT", "DISTCD", "DIVAMT"]],
        on="permno",
        how="left",
    )
    # DCLRDT must be strictly after entry and on or before exit
    in_window = (merged["DCLRDT"] > merged["entry_date"]) & (
        merged["DCLRDT"] <= merged["exit_date"]
    )
    merged = merged[in_window].copy()

    # Keep earliest DCLRDT per trade
    merged = merged.sort_values(["trade_id", "DCLRDT"])
    matched = merged.groupby("trade_id", as_index=False).first()
    return matched


def build_cumret(B: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative return index per permno (base = 1.0 at first observation)."""
    B = B.sort_values(["permno", "date"])
    B["log1r"] = np.log1p(B["ret"].clip(-0.99, 10))
    B["cumlogr"] = B.groupby("permno")["log1r"].cumsum()
    return B[["permno", "date", "ret", "cumlogr"]]


def lookup_cumlogr(cumret: pd.DataFrame, keys: pd.DataFrame, date_col: str, label: str) -> pd.Series:
    """
    For each row in keys, look up cumlogr at nearest date on or before date_col,
    grouped by permno.  Processes each permno separately to satisfy merge_asof's
    requirement that the `on` column is monotone within each group.
    Returns a Series aligned to keys.index.
    """
    tmp = keys[["permno", date_col]].rename(columns={date_col: "date"}).copy()
    result = pd.Series(index=keys.index, dtype=float, name=label)

    for pno, grp in tmp.groupby("permno"):
        sub_cum = cumret.loc[cumret["permno"] == pno, ["date", "cumlogr"]].sort_values("date")
        if sub_cum.empty:
            continue
        lkp = grp[["date"]].sort_values("date")
        merged = pd.merge_asof(lkp, sub_cum, on="date", direction="backward")
        merged.index = lkp.index
        result.loc[grp.index] = merged["cumlogr"].values

    return result


def compute_phase_returns(matched: pd.DataFrame, cumret: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: entry_date → day before DCLRDT
    Phase 2: DCLRDT    → exit_date

    Return = exp(cumlogr_end - cumlogr_start) - 1
    """
    # Pre-declaration: [entry_date, DCLRDT-1]
    # We look up cumlogr at the business day just before DCLRDT.
    # Using direction="backward" at (DCLRDT - 1 day) is not reliable because
    # DCLRDT-1 may be a weekend. Instead: look up cumlogr AT entry_date and
    # AT the day just before DCLRDT (by merging backward from DCLRDT - 1 day).
    matched = matched.copy()
    matched["dclrdt_minus1"] = matched["DCLRDT"] - pd.Timedelta(days=1)

    cl_entry = lookup_cumlogr(cumret, matched, "entry_date", "cl_entry")
    cl_predecl = lookup_cumlogr(cumret, matched, "dclrdt_minus1", "cl_predecl")
    cl_decl = lookup_cumlogr(cumret, matched, "DCLRDT", "cl_decl")
    cl_exit = lookup_cumlogr(cumret, matched, "exit_date", "cl_exit")

    matched = matched.join(cl_entry).join(cl_predecl).join(cl_decl).join(cl_exit)

    # Phase 1: entry open → close just before declaration
    matched["ret_phase1"] = np.expm1(matched["cl_predecl"] - matched["cl_entry"])
    # Phase 2: declaration day close → exit close
    matched["ret_phase2"] = np.expm1(matched["cl_exit"] - matched["cl_predecl"])
    # Replication check: (1+p1)*(1+p2) - 1 vs realized_holding_return
    matched["ret_replicated"] = (1 + matched["ret_phase1"]) * (1 + matched["ret_phase2"]) - 1

    return matched


def summarize(matched: pd.DataFrame):
    print("\n=== Event Phase Attribution ===")
    print(f"Matched trades (DCLRDT in window): {len(matched):,}")
    print(f"Unmatched (no declaration in window): available via trades file\n")

    cols = ["realized_holding_return", "ret_phase1", "ret_phase2", "ret_replicated"]
    means = matched[cols].mean() * 100
    meds  = matched[cols].median() * 100
    print("Mean returns (%):")
    for c in cols:
        print(f"  {c:<30s}  mean={means[c]:+.3f}%  median={meds[c]:+.3f}%")

    print("\n--- Phase 2 breakdown by DCLRDT day-of-week ---")
    matched["decl_dow"] = matched["DCLRDT"].dt.day_name()
    print(matched.groupby("decl_dow")[["ret_phase1", "ret_phase2"]].mean().mul(100).round(3))

    print("\n--- Phase 2 breakdown by signal_group ---")
    if "signal_group" in matched.columns:
        print(matched.groupby("signal_group")[["ret_phase1", "ret_phase2", "realized_holding_return"]]
              .mean().mul(100).round(3))

    print("\n--- Replication error (replicated - realized) ---")
    err = (matched["ret_replicated"] - matched["realized_holding_return"]).abs()
    print(f"  Mean abs error: {err.mean()*100:.4f}%  Max: {err.max()*100:.3f}%")

    # Phase contribution
    total_mean = matched["realized_holding_return"].mean()
    p1_mean = matched["ret_phase1"].mean()
    p2_mean = matched["ret_phase2"].mean()
    print(f"\n--- Phase attribution (matched trades only) ---")
    print(f"  Total mean return:  {total_mean*100:+.3f}%")
    print(f"  Phase 1 (pre-decl): {p1_mean*100:+.3f}%  ({p1_mean/total_mean*100:.1f}% of total)")
    print(f"  Phase 2 (decl+post):{p2_mean*100:+.3f}%  ({p2_mean/total_mean*100:.1f}% of total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hpo_v9_d6_lr005_sub09")
    args = parser.parse_args()

    print(f"Loading data for run: {args.run}")
    trades, A, B = load_data(args.run)
    print(f"Trades: {len(trades):,}  |  TableA events: {len(A):,}  |  TableB rows: {len(B):,}")

    print("Matching declarations to trade windows...")
    matched = match_declaration(trades, A)
    print(f"Matched: {len(matched):,} / {len(trades):,} trades ({len(matched)/len(trades)*100:.1f}%)")

    print("Building cumulative return index...")
    # Only need permnos that appear in matched trades
    permnos = matched["permno"].unique()
    B_sub = B[B["permno"].isin(permnos)]
    cumret = build_cumret(B_sub)

    print("Computing phase returns...")
    matched = compute_phase_returns(matched, cumret)

    summarize(matched)

    out_dir = os.path.join(ROOT, "outputs", "runs", args.run, "backtest", "research")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "event_phase_attribution.csv")
    matched.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
