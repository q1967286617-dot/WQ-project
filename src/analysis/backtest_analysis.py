from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def return_metrics(ret: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if x.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    nav = (1.0 + x).cumprod()
    years = max(len(x) / 252.0, 1.0 / 252.0)
    drawdown = nav / nav.cummax() - 1.0
    vol = float(x.std(ddof=1) * np.sqrt(252.0)) if len(x) > 1 else 0.0
    sharpe = float((x.mean() / (x.std(ddof=1) + 1e-12)) * np.sqrt(252.0)) if len(x) > 1 else 0.0
    return {
        "total_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(nav.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def load_run_backtest_frames(project_root: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backtest_dir = project_root / "outputs" / "runs" / run_id / "backtest"
    daily_df = _read_df(backtest_dir / "daily_portfolio.csv")
    trades_df = _read_df(backtest_dir / "trades.csv")
    signals_df = _read_df(backtest_dir / "signals.csv")
    return daily_df, trades_df, signals_df


def enrich_trades_with_split_context(
    trades_df: pd.DataFrame,
    split_df: pd.DataFrame,
    signal_date_col: str = "signal_date",
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    raw_cols = ["DlyCalDt", "PERMNO", "turnover_5d", "industry", "has_div_history", "y_div_10d", "DlyPrc"]
    context = split_df[[c for c in raw_cols if c in split_df.columns]].copy()
    context["DlyCalDt"] = pd.to_datetime(context["DlyCalDt"], errors="coerce")
    context["PERMNO"] = pd.to_numeric(context["PERMNO"], errors="coerce").astype("Int64")
    context = context[context["PERMNO"].notna()].copy()
    context["PERMNO"] = context["PERMNO"].astype(int)

    trades = trades_df.copy()
    trades[signal_date_col] = pd.to_datetime(trades[signal_date_col], errors="coerce")
    trades["permno"] = pd.to_numeric(trades["permno"], errors="coerce").astype("Int64")
    trades = trades[trades["permno"].notna()].copy()
    trades["permno"] = trades["permno"].astype(int)

    merged = trades.merge(
        context,
        left_on=[signal_date_col, "permno"],
        right_on=["DlyCalDt", "PERMNO"],
        how="left",
        suffixes=("", "__split"),
    ).drop(columns=["DlyCalDt", "PERMNO"], errors="ignore")

    for col in ["turnover_5d", "industry", "has_div_history", "y_div_10d", "DlyPrc"]:
        split_col = f"{col}__split"
        if split_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[split_col]
            else:
                merged[col] = merged[split_col].where(merged[split_col].notna(), merged[col])
            merged = merged.drop(columns=[split_col])

    if "entry_date" in merged.columns:
        merged["entry_date"] = pd.to_datetime(merged["entry_date"], errors="coerce")
        merged["entry_year"] = merged["entry_date"].dt.year.astype("Int64")
    merged["signal_date"] = pd.to_datetime(merged["signal_date"], errors="coerce")
    merged["signal_year"] = merged["signal_date"].dt.year.astype("Int64")
    merged["has_div_history"] = pd.to_numeric(merged.get("has_div_history"), errors="coerce").fillna(0).astype(int)
    merged["history_bucket"] = np.where(merged["has_div_history"].gt(0), "has_history", "no_history")
    return merged


def _quantile_edges(values: pd.Series, q: int) -> list[float]:
    ranked = pd.to_numeric(values, errors="coerce").dropna()
    if ranked.empty:
        return []
    probs = np.linspace(0.0, 1.0, q + 1)
    edges = ranked.quantile(probs).to_numpy(dtype=float).tolist()
    deduped: list[float] = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def add_liquidity_bucket(
    df: pd.DataFrame,
    value_col: str = "turnover_5d",
    bucket_col: str = "liquidity_bucket",
    labels: Iterable[str] | None = None,
    edges: list[float] | None = None,
    q: int = 4,
) -> tuple[pd.DataFrame, list[float]]:
    out = df.copy()
    if labels is None:
        labels = [f"liq_q{i}" for i in range(1, q + 1)]
    labels = list(labels)
    if edges is None:
        edges = _quantile_edges(out[value_col], len(labels))

    if len(edges) <= 1:
        out[bucket_col] = "unknown"
        return out, edges

    bins = [-np.inf] + edges[1:-1] + [np.inf]
    out[bucket_col] = pd.cut(
        pd.to_numeric(out[value_col], errors="coerce"),
        bins=bins,
        labels=labels[: len(bins) - 1],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    return out, edges


def compute_trade_group_metrics(trades_df: pd.DataFrame, group_col: str, version: str | None = None) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[group_col, "n_trades", "mean_trade_return", "win_rate", "hit_rate"])

    x = trades_df.copy()
    x[group_col] = x[group_col].astype("string").fillna("unknown")
    out = (
        x.groupby(group_col, dropna=False)
        .agg(
            n_trades=("trade_id", "size"),
            mean_trade_return=("realized_holding_return", "mean"),
            median_trade_return=("realized_holding_return", "median"),
            win_rate=("realized_holding_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            hit_rate=("y_entry", "mean"),
            avg_total_cost=("total_cost_rate", "mean"),
            mean_prob=("prob", "mean"),
        )
        .reset_index()
    )
    if version is not None:
        out.insert(0, "version", version)
    return out


def compute_yearly_backtest_metrics(daily_df: pd.DataFrame, version: str | None = None) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(
            columns=["year", "portfolio_annualized_return", "excess_annualized_return", "portfolio_sharpe", "excess_sharpe"]
        )

    x = daily_df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["year"] = x["date"].dt.year.astype(int)

    rows: list[dict[str, float | int | str]] = []
    for year, g in x.groupby("year", sort=True):
        portfolio = return_metrics(g["portfolio_ret"])
        excess = return_metrics(g["excess_ret"])
        row: dict[str, float | int | str] = {
            "year": int(year),
            "n_days": int(len(g)),
            "avg_turnover": float(pd.to_numeric(g["turnover"], errors="coerce").mean()),
            "avg_positions": float(pd.to_numeric(g["n_positions"], errors="coerce").mean()),
            "signal_hit_rate": float(pd.to_numeric(g["signal_hit_rate"], errors="coerce").mean()),
            "portfolio_total_return": portfolio["total_return"],
            "portfolio_annualized_return": portfolio["annualized_return"],
            "portfolio_sharpe": portfolio["sharpe"],
            "excess_total_return": excess["total_return"],
            "excess_annualized_return": excess["annualized_return"],
            "excess_sharpe": excess["sharpe"],
        }
        if version is not None:
            row["version"] = version
        rows.append(row)
    return pd.DataFrame(rows)
