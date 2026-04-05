from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

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


def _best_forward_return_col(df: pd.DataFrame, holding_td: int | None = None) -> str | None:
    candidates: list[str] = []
    if holding_td is not None:
        candidates.append(f"fwd_ret_{int(holding_td)}d")
    candidates.extend(["fwd_ret_10d", "fwd_ret_5d", "fwd_ret_1d", "exec_ret_1d"])

    seen: set[str] = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in df.columns:
            return col
    return None


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    aligned = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(aligned) < 2:
        return np.nan
    if aligned["x"].nunique() <= 1 or aligned["y"].nunique() <= 1:
        return np.nan
    method = str(method).lower()
    if method == "spearman":
        x_rank = aligned["x"].rank(method="average")
        y_rank = aligned["y"].rank(method="average")
        return float(x_rank.corr(y_rank, method="pearson"))
    return float(aligned["x"].corr(aligned["y"], method=method))

def summarize_score_return_correlations(
    pool_df: pd.DataFrame,
    score_col: str = "prob",
    return_col: str | None = None,
    holding_td: int | None = None,
    methods: Sequence[str] = ("spearman", "pearson"),
    min_obs: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = pool_df.copy()
    if "eligible" in x.columns:
        x = x[x["eligible"]].copy()
    if x.empty:
        return pd.DataFrame(columns=["date", "n_pool"] + [f"{m}_corr" for m in methods]), {
            "return_col": return_col or _best_forward_return_col(pool_df, holding_td),
            "n_days": 0,
            "n_pool_rows": 0,
        }

    if return_col is None:
        return_col = _best_forward_return_col(x, holding_td)
    if return_col is None:
        raise ValueError("No forward return column available for correlation diagnostics")

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for dt, g in x.groupby("date", sort=True):
        row: dict[str, Any] = {"date": dt, "n_pool": int(len(g))}
        for method in methods:
            row[f"{method}_corr"] = _safe_corr(g[score_col], g[return_col], method=method) if len(g) >= min_obs else np.nan
        rows.append(row)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    summary: dict[str, Any] = {
        "return_col": return_col,
        "n_days": int(len(daily)),
        "n_pool_rows": int(len(x)),
        "avg_pool_size": float(daily["n_pool"].mean()) if len(daily) else np.nan,
    }
    for method in methods:
        col = f"{method}_corr"
        vals = pd.to_numeric(daily[col], errors="coerce").dropna()
        summary[f"{method}_mean"] = float(vals.mean()) if len(vals) else np.nan
        summary[f"{method}_median"] = float(vals.median()) if len(vals) else np.nan
        summary[f"{method}_positive_rate"] = float((vals > 0).mean()) if len(vals) else np.nan
        summary[f"{method}_n_days"] = int(len(vals))
    return daily, summary


def compute_score_bucket_metrics(
    pool_df: pd.DataFrame,
    score_col: str = "prob",
    return_col: str | None = None,
    holding_td: int | None = None,
    bucket_count: int = 10,
) -> pd.DataFrame:
    x = pool_df.copy()
    if "eligible" in x.columns:
        x = x[x["eligible"]].copy()
    if x.empty:
        return pd.DataFrame(columns=["score_bucket", "n_rows", "mean_score", "mean_forward_return", "median_forward_return", "hit_rate"])

    if return_col is None:
        return_col = _best_forward_return_col(x, holding_td)
    if return_col is None:
        raise ValueError("No forward return column available for score bucket diagnostics")

    x[score_col] = pd.to_numeric(x[score_col], errors="coerce")
    x[return_col] = pd.to_numeric(x[return_col], errors="coerce")
    ranked = x[score_col].rank(method="first")
    q = max(2, min(int(bucket_count), int(ranked.notna().sum())))
    x["score_bucket"] = pd.qcut(ranked, q=q, labels=[f"q{i}" for i in range(1, q + 1)])
    out = (
        x.groupby("score_bucket", dropna=False)
        .agg(
            n_rows=("permno", "size"),
            mean_score=(score_col, "mean"),
            mean_forward_return=(return_col, "mean"),
            median_forward_return=(return_col, "median"),
            hit_rate=("y_div_10d", "mean"),
        )
        .reset_index()
    )
    out["return_col"] = return_col
    return out


def compute_topk_return_metrics(
    pool_df: pd.DataFrame,
    ks: Sequence[int],
    score_col: str = "prob",
    return_col: str | None = None,
    holding_td: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pool_df.copy()
    if "eligible" in x.columns:
        x = x[x["eligible"]].copy()
    if x.empty:
        empty_daily = pd.DataFrame(columns=["date", "k", "n_pool", "n_selected", "mean_forward_return", "median_forward_return", "hit_rate", "pool_mean_forward_return", "pool_hit_rate"])
        empty_summary = pd.DataFrame(columns=["k", "n_days", "avg_pool_size", "mean_forward_return", "median_forward_return", "mean_hit_rate", "mean_return_spread_vs_pool", "mean_hit_spread_vs_pool"])
        return empty_daily, empty_summary

    if return_col is None:
        return_col = _best_forward_return_col(x, holding_td)
    if return_col is None:
        raise ValueError("No forward return column available for top-k diagnostics")

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x[score_col] = pd.to_numeric(x[score_col], errors="coerce")
    x[return_col] = pd.to_numeric(x[return_col], errors="coerce")
    x["y_div_10d"] = pd.to_numeric(x.get("y_div_10d"), errors="coerce")

    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    rows: list[dict[str, Any]] = []
    for dt, g in x.groupby("date", sort=True):
        ranked = g.sort_values([score_col, "permno"], ascending=[False, True])
        pool_mean_ret = float(pd.to_numeric(g[return_col], errors="coerce").mean())
        pool_hit_rate = float(pd.to_numeric(g["y_div_10d"], errors="coerce").mean()) if "y_div_10d" in g.columns else np.nan
        for k in unique_ks:
            top = ranked.head(k).copy()
            if top.empty:
                continue
            top_ret = pd.to_numeric(top[return_col], errors="coerce")
            top_hit = pd.to_numeric(top["y_div_10d"], errors="coerce") if "y_div_10d" in top.columns else pd.Series(dtype=float)
            rows.append(
                {
                    "date": dt,
                    "k": int(k),
                    "return_col": return_col,
                    "n_pool": int(len(g)),
                    "n_selected": int(len(top)),
                    "mean_forward_return": float(top_ret.mean()),
                    "median_forward_return": float(top_ret.median()),
                    "hit_rate": float(top_hit.mean()) if len(top_hit) else np.nan,
                    "pool_mean_forward_return": pool_mean_ret,
                    "pool_hit_rate": pool_hit_rate,
                    "return_spread_vs_pool": float(top_ret.mean() - pool_mean_ret),
                    "hit_spread_vs_pool": float(top_hit.mean() - pool_hit_rate) if len(top_hit) else np.nan,
                }
            )

    daily = pd.DataFrame(rows).sort_values(["date", "k"]).reset_index(drop=True)
    if daily.empty:
        return daily, pd.DataFrame(columns=["k", "n_days", "avg_pool_size", "mean_forward_return", "median_forward_return", "mean_hit_rate", "mean_return_spread_vs_pool", "mean_hit_spread_vs_pool"])
    summary = (
        daily.groupby("k", sort=True)
        .agg(
            n_days=("date", "nunique"),
            avg_pool_size=("n_pool", "mean"),
            mean_forward_return=("mean_forward_return", "mean"),
            median_forward_return=("mean_forward_return", "median"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_return_spread_vs_pool=("return_spread_vs_pool", "mean"),
            mean_hit_spread_vs_pool=("hit_spread_vs_pool", "mean"),
        )
        .reset_index()
    )
    summary["return_col"] = return_col
    return daily, summary

