"""
Year-by-year performance attribution for one or more versions.

Reads existing backtest outputs (daily_portfolio.csv + trades.csv) and produces:
  - Per-year Sharpe, annual return, excess return, hit rate
  - Attribution breakdown: which years drove the strategy vs dragged it

Usage:
    python scripts/analyze_yearly_attribution.py --versions v9 v12
    python scripts/analyze_yearly_attribution.py --versions v1 v7 v9 v12 --run_ids v1_run1 v7_run1 v9_run1 v12_run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.paths import load_yaml, resolve_paths, ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_daily(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "backtest" / "daily_portfolio.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _load_trades(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "backtest" / "trades.csv"
    return pd.read_csv(path)


def _yearly_metrics(daily: pd.DataFrame, label: str) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year

    rows = []
    for year, g in daily.groupby("year"):
        g = g.sort_values("date")
        ret_col = "portfolio_ret" if "portfolio_ret" in g.columns else "portfolio_return"
        bench_col = "benchmark_ret" if "benchmark_ret" in g.columns else "benchmark_return"
        strat = g[ret_col].fillna(0)
        bench = g[bench_col].fillna(0)
        excess = strat - bench

        ann_factor = 252 / len(g)
        strat_ann = (1 + strat).prod() ** ann_factor - 1
        excess_ann = (1 + excess).prod() ** ann_factor - 1
        sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else np.nan
        hit = (strat > 0).mean()

        rows.append({
            "version": label,
            "year": int(year),
            "n_days": len(g),
            "ann_return": strat_ann,
            "excess_ann_return": excess_ann,
            "sharpe": sharpe,
            "hit_rate": hit,
        })
    return pd.DataFrame(rows)


def _trade_yearly_metrics(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    df = trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["year"] = df["entry_date"].dt.year
    if "pnl_net" not in df.columns:
        df["pnl_net"] = df.get("pnl", df.get("gross_pnl", np.nan))
    if "return_net" not in df.columns and "trade_return" in df.columns:
        df["return_net"] = df["trade_return"]

    rows = []
    for year, g in df.groupby("year"):
        row = {
            "version": label,
            "year": int(year),
            "n_trades": len(g),
        }
        if "return_net" in g.columns:
            row["mean_trade_return"] = g["return_net"].mean()
            row["trade_hit_rate"] = (g["return_net"] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_yearly_bars(df: pd.DataFrame, col: str, ylabel: str, title: str, out: Path) -> None:
    versions = df["version"].unique()
    years = sorted(df["year"].unique())
    x = np.arange(len(years))
    width = 0.8 / max(len(versions), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ver in enumerate(versions):
        sub = df[df["version"] == ver].set_index("year")
        vals = [sub.loc[y, col] if y in sub.index else np.nan for y in years]
        offset = (i - (len(versions) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, label=ver)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def _plot_sharpe_bars(df: pd.DataFrame, out: Path) -> None:
    versions = df["version"].unique()
    years = sorted(df["year"].unique())
    x = np.arange(len(years))
    width = 0.8 / max(len(versions), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ver in enumerate(versions):
        sub = df[df["version"] == ver].set_index("year")
        vals = [sub.loc[y, "sharpe"] if y in sub.index else np.nan for y in years]
        offset = (i - (len(versions) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=ver)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Yearly Sharpe Ratio by Version")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v9"])
    ap.add_argument("--run_ids", nargs="+", default=None,
                    help="Explicit run_ids (same order as --versions). Defaults to <version>_run1.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--out_dir", default="outputs/yearly_attribution")
    args = ap.parse_args()

    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)
    out_dir = PROJECT_ROOT / args.out_dir
    ensure_dir(out_dir)

    run_ids = args.run_ids or [f"{v}_run1" for v in args.versions]
    if len(run_ids) != len(args.versions):
        raise SystemExit("--run_ids must have same length as --versions")

    all_yearly: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []

    for ver, run_id in zip(args.versions, run_ids):
        run_dir = paths.outputs_dir / run_id
        if not (run_dir / "backtest" / "daily_portfolio.csv").exists():
            print(f"[WARN] missing backtest for {ver}/{run_id}, skipping")
            continue
        daily = _load_daily(run_dir)
        trades = _load_trades(run_dir)
        ym = _yearly_metrics(daily, label=ver)
        tm = _trade_yearly_metrics(trades, label=ver)
        all_yearly.append(ym)
        all_trades.append(tm)
        print(f"\n{ver} ({run_id}) — year-by-year:")
        print(ym[["year", "ann_return", "excess_ann_return", "sharpe", "hit_rate"]].to_string(index=False))

    if not all_yearly:
        raise SystemExit("No valid runs found.")

    yearly_df = pd.concat(all_yearly, ignore_index=True)
    trades_df = pd.concat(all_trades, ignore_index=True)

    yearly_df.to_csv(out_dir / "yearly_metrics.csv", index=False)
    trades_df.to_csv(out_dir / "trade_yearly_metrics.csv", index=False)
    print(f"\nSaved CSVs → {out_dir}")

    _plot_yearly_bars(yearly_df, "ann_return", "Ann. Return", "Yearly Annualized Return by Version",
                      out_dir / "yearly_ann_return.png")
    _plot_yearly_bars(yearly_df, "excess_ann_return", "Excess Ann. Return",
                      "Yearly Excess Annualized Return by Version",
                      out_dir / "yearly_excess_return.png")
    _plot_sharpe_bars(yearly_df, out_dir / "yearly_sharpe.png")

    # Attribution summary
    for ver in args.versions:
        sub = yearly_df[yearly_df["version"] == ver].copy()
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("excess_ann_return", ascending=False)
        best = sub_sorted.iloc[0]
        worst = sub_sorted.iloc[-1]
        pos_years = sub[sub["excess_ann_return"] > 0]
        neg_years = sub[sub["excess_ann_return"] < 0]
        print(f"\n{'='*50}")
        print(f"Attribution summary: {ver}")
        print(f"  Best year:  {int(best['year'])} excess={best['excess_ann_return']:.2%} sharpe={best['sharpe']:.3f}")
        print(f"  Worst year: {int(worst['year'])} excess={worst['excess_ann_return']:.2%} sharpe={worst['sharpe']:.3f}")
        print(f"  Positive-excess years: {len(pos_years)}/{len(sub)}  avg={pos_years['excess_ann_return'].mean():.2%}")
        print(f"  Negative-excess years: {len(neg_years)}/{len(sub)}  avg={neg_years['excess_ann_return'].mean():.2%}")


if __name__ == "__main__":
    main()
