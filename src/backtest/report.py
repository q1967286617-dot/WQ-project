from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.paths import ensure_dir


def enrich_daily_report(daily_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    x = daily_df.merge(benchmark_df[["date", "benchmark_ret", "benchmark_nav"]], on="date", how="left")
    x["benchmark_ret"] = x["benchmark_ret"].fillna(0.0)
    x["portfolio_nav"] = (1.0 + x["portfolio_ret"]).cumprod()
    x["excess_ret"] = x["portfolio_ret"] - x["benchmark_ret"]
    x["excess_nav"] = (1.0 + x["excess_ret"]).cumprod()
    x["drawdown"] = x["portfolio_nav"] / x["portfolio_nav"].cummax() - 1.0
    x["industry_exposure_json"] = x["industry_exposure"].apply(
        lambda v: json.dumps(v, ensure_ascii=True, sort_keys=True)
    )
    return x


def _cost_sum(daily_df: pd.DataFrame, cols: list[str]) -> float:
    total = 0.0
    for col in cols:
        total += float(pd.to_numeric(daily_df.get(col), errors="coerce").fillna(0.0).sum())
    return total


def _safe_cost_share(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) < 1e-12:
        return None
    return float(numerator / denominator)


def _trade_cost_detail(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    x = trades_df.copy()
    x["gross_holding_return"] = (
        pd.to_numeric(x.get("realized_holding_return"), errors="coerce").fillna(0.0)
        + pd.to_numeric(x.get("total_cost_rate"), errors="coerce").fillna(0.0)
    )
    x["net_holding_return"] = pd.to_numeric(x.get("realized_holding_return"), errors="coerce").fillna(0.0)
    x["fixed_cost_rate"] = (
        pd.to_numeric(x.get("entry_fixed_cost_rate"), errors="coerce").fillna(0.0)
        + pd.to_numeric(x.get("exit_fixed_cost_rate"), errors="coerce").fillna(0.0)
    )
    x["spread_cost_rate"] = (
        pd.to_numeric(x.get("entry_spread_rate"), errors="coerce").fillna(0.0)
        + pd.to_numeric(x.get("exit_spread_rate"), errors="coerce").fillna(0.0)
    )
    x["total_cost_rate"] = pd.to_numeric(x.get("total_cost_rate"), errors="coerce").fillna(
        x["fixed_cost_rate"] + x["spread_cost_rate"]
    )
    x["cost_drag_rate"] = x["gross_holding_return"] - x["net_holding_return"]
    x["cost_share_of_gross"] = np.where(
        x["gross_holding_return"].abs() > 1e-12,
        x["cost_drag_rate"] / x["gross_holding_return"],
        np.nan,
    )
    return x


def _aggregate_trade_costs(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "n_trades", "mean_gross_holding_return", "mean_net_holding_return"])

    x = df.copy()
    x[group_col] = x[group_col].astype("string").fillna("unknown")
    out = (
        x.groupby(group_col, dropna=False)
        .agg(
            n_trades=("trade_id", "size"),
            mean_gross_holding_return=("gross_holding_return", "mean"),
            median_gross_holding_return=("gross_holding_return", "median"),
            mean_net_holding_return=("net_holding_return", "mean"),
            median_net_holding_return=("net_holding_return", "median"),
            mean_cost_drag_rate=("cost_drag_rate", "mean"),
            mean_total_cost_rate=("total_cost_rate", "mean"),
            mean_fixed_cost_rate=("fixed_cost_rate", "mean"),
            mean_spread_cost_rate=("spread_cost_rate", "mean"),
            total_cost_drag=("cost_drag_rate", "sum"),
            total_gross_holding_return=("gross_holding_return", "sum"),
            total_net_holding_return=("net_holding_return", "sum"),
            win_rate_net=("net_holding_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            win_rate_gross=("gross_holding_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_prob=("prob", "mean"),
        )
        .reset_index()
    )
    out["cost_share_of_gross"] = out.apply(
        lambda row: _safe_cost_share(float(row["total_cost_drag"]), float(row["total_gross_holding_return"])),
        axis=1,
    )
    return out


def _cost_turnover_summary(daily_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    if daily_df.empty:
        return {
            "avg_turnover": 0.0,
            "gross": _return_metrics(pd.Series(dtype=float)),
            "net": _return_metrics(pd.Series(dtype=float)),
            "daily_costs": {},
            "trades": {},
        }

    gross_metrics = _return_metrics(pd.to_numeric(daily_df.get("gross_ret"), errors="coerce").fillna(0.0))
    net_metrics = _return_metrics(pd.to_numeric(daily_df.get("portfolio_ret"), errors="coerce").fillna(0.0))
    total_entry_fixed = _cost_sum(daily_df, ["entry_fixed_cost"])
    total_entry_spread = _cost_sum(daily_df, ["entry_spread_cost"])
    total_exit_fixed = _cost_sum(daily_df, ["exit_fixed_cost"])
    total_exit_spread = _cost_sum(daily_df, ["exit_spread_cost"])
    total_cost = total_entry_fixed + total_entry_spread + total_exit_fixed + total_exit_spread
    total_gross = float(pd.to_numeric(daily_df.get("gross_ret"), errors="coerce").fillna(0.0).sum())

    trade_detail = _trade_cost_detail(trades_df)
    if trade_detail.empty:
        trade_summary = {
            "n_trades": 0,
            "mean_gross_holding_return": np.nan,
            "median_gross_holding_return": np.nan,
            "mean_net_holding_return": np.nan,
            "median_net_holding_return": np.nan,
            "mean_total_cost_rate": np.nan,
            "mean_fixed_cost_rate": np.nan,
            "mean_spread_cost_rate": np.nan,
            "aggregate_cost_share_of_gross": None,
        }
    else:
        total_trade_cost_drag = float(pd.to_numeric(trade_detail["cost_drag_rate"], errors="coerce").fillna(0.0).sum())
        total_trade_gross = float(pd.to_numeric(trade_detail["gross_holding_return"], errors="coerce").fillna(0.0).sum())
        trade_summary = {
            "n_trades": int(len(trade_detail)),
            "mean_gross_holding_return": float(pd.to_numeric(trade_detail["gross_holding_return"], errors="coerce").mean()),
            "median_gross_holding_return": float(pd.to_numeric(trade_detail["gross_holding_return"], errors="coerce").median()),
            "mean_net_holding_return": float(pd.to_numeric(trade_detail["net_holding_return"], errors="coerce").mean()),
            "median_net_holding_return": float(pd.to_numeric(trade_detail["net_holding_return"], errors="coerce").median()),
            "mean_total_cost_rate": float(pd.to_numeric(trade_detail["total_cost_rate"], errors="coerce").mean()),
            "mean_fixed_cost_rate": float(pd.to_numeric(trade_detail["fixed_cost_rate"], errors="coerce").mean()),
            "mean_spread_cost_rate": float(pd.to_numeric(trade_detail["spread_cost_rate"], errors="coerce").mean()),
            "aggregate_cost_share_of_gross": _safe_cost_share(total_trade_cost_drag, total_trade_gross),
        }

    return {
        "avg_turnover": float(pd.to_numeric(daily_df["turnover"], errors="coerce").mean()),
        "median_turnover": float(pd.to_numeric(daily_df["turnover"], errors="coerce").median()),
        "gross": gross_metrics,
        "net": net_metrics,
        "cost_drag": {
            "annualized_return_diff": float(gross_metrics["annualized_return"] - net_metrics["annualized_return"]),
            "total_return_diff": float(gross_metrics["total_return"] - net_metrics["total_return"]),
            "sharpe_diff": float(gross_metrics["sharpe"] - net_metrics["sharpe"]),
        },
        "daily_costs": {
            "total_entry_fixed_cost": total_entry_fixed,
            "total_entry_spread_cost": total_entry_spread,
            "total_exit_fixed_cost": total_exit_fixed,
            "total_exit_spread_cost": total_exit_spread,
            "total_cost": total_cost,
            "avg_daily_total_cost": float(total_cost / max(len(daily_df), 1)),
            "cost_share_of_gross_sum": _safe_cost_share(total_cost, total_gross),
        },
        "trades": trade_summary,
    }


def summarize_backtest(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    research_decision: Dict,
    cost_model: Optional[Dict] = None,
    three_way_comparison: Optional[Dict] = None,
    ranking_comparison: Optional[Dict] = None,
    selection_diagnostics: Optional[Dict] = None,
) -> Dict:
    out = {
        "n_days": int(len(daily_df)),
        "avg_positions": float(daily_df["n_positions"].mean()) if len(daily_df) else 0.0,
        "avg_turnover": float(daily_df["turnover"].mean()) if len(daily_df) else 0.0,
        "signal_hit_rate_on_entries": float(
            pd.to_numeric(daily_df["signal_hit_rate"], errors="coerce").mean()
        ) if len(daily_df) else np.nan,
        "portfolio": _return_metrics(daily_df["portfolio_ret"]),
        "excess_vs_benchmark": _return_metrics(daily_df["excess_ret"]),
        "benchmark_final_nav": float(daily_df["benchmark_nav"].iloc[-1]) if len(daily_df) else 1.0,
        "trades": {
            "n_trades": int(len(trades_df)),
            "win_rate": float(
                (pd.to_numeric(trades_df.get("realized_holding_return"), errors="coerce") > 0).mean()
            ) if len(trades_df) else np.nan,
            "mean_trade_return": float(
                pd.to_numeric(trades_df.get("realized_holding_return"), errors="coerce").mean()
            ) if len(trades_df) else np.nan,
            "truncated_rate": float(
                pd.to_numeric(trades_df.get("truncated"), errors="coerce").mean()
            ) if len(trades_df) else np.nan,
        },
        "research_decision": research_decision,
        "cost_turnover": _cost_turnover_summary(daily_df, trades_df),
    }
    if cost_model is not None:
        out["cost_model"] = cost_model

    if three_way_comparison:
        out["three_way_comparison"] = three_way_comparison
        alpha = three_way_comparison.get("alpha_capture", {})
        ratio = alpha.get("primary_alpha_capture_ratio")
        grade = three_way_comparison.get("grade", "N/A")
        if ratio is not None:
            out["alpha_capture_ratio"] = round(float(ratio), 4)
            out["alpha_capture_grade"] = grade
            out["alpha_capture_pct"] = f"{ratio * 100:.1f}%"
    if ranking_comparison:
        out["ranking_comparison"] = ranking_comparison
    if selection_diagnostics:
        out["selection_diagnostics"] = selection_diagnostics

    return out


def build_trade_attribution(trades_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if trades_df.empty:
        empty = trades_df.copy()
        return {
            "prob_buckets": empty,
            "dividend_groups": empty,
            "industry": empty,
            "trade_cost_detail": empty,
            "cost_by_signal_group": empty,
            "cost_by_industry": empty,
            "cost_by_prob_bucket": empty,
        }

    x = _trade_cost_detail(trades_df)
    q = min(4, len(x))
    x["prob_bucket"] = pd.qcut(
        x["prob"].rank(method="first"), q=q,
        labels=[f"q{i}" for i in range(1, q + 1)]
    ).astype("string")
    x["dividend_group"] = x["signal_group"].astype(str)

    def _agg(col: str) -> pd.DataFrame:
        return (
            x.groupby(col, dropna=False)
            .agg(
                n_trades=("trade_id", "size"),
                mean_trade_return=("realized_holding_return", "mean"),
                median_trade_return=("realized_holding_return", "median"),
                win_rate=(
                    "realized_holding_return",
                    lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()),
                ),
                mean_prob=("prob", "mean"),
                mean_hit=("y_entry", "mean"),
            )
            .reset_index()
        )

    return {
        "prob_buckets": _agg("prob_bucket"),
        "dividend_groups": _agg("dividend_group"),
        "industry": _agg("industry"),
        "trade_cost_detail": x,
        "cost_by_signal_group": _aggregate_trade_costs(x, "signal_group"),
        "cost_by_industry": _aggregate_trade_costs(x, "industry"),
        "cost_by_prob_bucket": _aggregate_trade_costs(x, "prob_bucket"),
    }


def write_backtest_outputs(
    out_dir: Path,
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    summary: Dict,
    research_reports: Dict[str, pd.DataFrame],
    attribution_reports: Dict[str, pd.DataFrame],
    random_daily_df: Optional[pd.DataFrame] = None,
    oracle_return_daily_df: Optional[pd.DataFrame] = None,
    oracle_event_daily_df: Optional[pd.DataFrame] = None,
    random_trades_df: Optional[pd.DataFrame] = None,
    oracle_return_trades_df: Optional[pd.DataFrame] = None,
    oracle_event_trades_df: Optional[pd.DataFrame] = None,
    additional_reference_outputs: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "research")
    ensure_dir(out_dir / "reports")

    _write_df(panel, out_dir / "panel.parquet")
    _write_df(candidates, out_dir / "signals.csv")
    _write_df(
        daily_df.drop(columns=["industry_exposure"], errors="ignore"),
        out_dir / "daily_portfolio.csv",
    )
    _write_df(trades_df, out_dir / "trades.csv")
    _write_df(positions_df, out_dir / "positions.csv")
    _write_json(summary, out_dir / "summary.json")

    for name, df in research_reports.items():
        _write_df(df, out_dir / "research" / f"{name}.csv")
    for name, df in attribution_reports.items():
        _write_df(df, out_dir / "reports" / f"{name}.csv")

    ref_dir = out_dir / "reference"

    if random_daily_df is not None and not random_daily_df.empty:
        ensure_dir(ref_dir)
        _write_df(
            random_daily_df.drop(columns=["industry_exposure"], errors="ignore"),
            ref_dir / "random_baseline_daily.csv",
        )
        if random_trades_df is not None and not random_trades_df.empty:
            _write_df(random_trades_df, ref_dir / "random_baseline_trades.csv")
        _write_json(
            _return_metrics(random_daily_df["portfolio_ret"]),
            ref_dir / "random_baseline_metrics.json",
        )

    if oracle_return_daily_df is not None and not oracle_return_daily_df.empty:
        ensure_dir(ref_dir)
        _write_df(
            oracle_return_daily_df.drop(columns=["industry_exposure"], errors="ignore"),
            ref_dir / "oracle_return_ceiling_daily.csv",
        )
        if oracle_return_trades_df is not None and not oracle_return_trades_df.empty:
            _write_df(oracle_return_trades_df, ref_dir / "oracle_return_ceiling_trades.csv")
        _write_json(
            _return_metrics(oracle_return_daily_df["portfolio_ret"]),
            ref_dir / "oracle_return_ceiling_metrics.json",
        )

    if oracle_event_daily_df is not None and not oracle_event_daily_df.empty:
        ensure_dir(ref_dir)
        _write_df(
            oracle_event_daily_df.drop(columns=["industry_exposure"], errors="ignore"),
            ref_dir / "oracle_event_ceiling_daily.csv",
        )
        if oracle_event_trades_df is not None and not oracle_event_trades_df.empty:
            _write_df(oracle_event_trades_df, ref_dir / "oracle_event_ceiling_trades.csv")
        _write_json(
            _return_metrics(oracle_event_daily_df["portfolio_ret"]),
            ref_dir / "oracle_event_ceiling_metrics.json",
        )

    if additional_reference_outputs:
        ensure_dir(ref_dir)
        for label, payload in additional_reference_outputs.items():
            daily = payload.get("daily")
            trades = payload.get("trades")
            if daily is not None and not daily.empty:
                _write_df(
                    daily.drop(columns=["industry_exposure"], errors="ignore"),
                    ref_dir / f"{label}_daily.csv",
                )
                _write_json(_return_metrics(daily["portfolio_ret"]), ref_dir / f"{label}_metrics.json")
            if trades is not None and not trades.empty:
                _write_df(trades, ref_dir / f"{label}_trades.csv")

    if "three_way_comparison" in summary and summary["three_way_comparison"]:
        ensure_dir(ref_dir)
        _write_json(summary["three_way_comparison"], ref_dir / "three_way_comparison.json")
    if "ranking_comparison" in summary and summary["ranking_comparison"]:
        ensure_dir(ref_dir)
        _write_json(summary["ranking_comparison"], ref_dir / "ranking_comparison.json")


def _return_metrics(ret: pd.Series) -> Dict:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if x.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    nav = (1.0 + x).cumprod()
    years = max(len(x) / 252.0, 1.0 / 252.0)
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    annualized_vol = float(x.std(ddof=1) * np.sqrt(252.0)) if len(x) > 1 else 0.0
    sharpe = float((x.mean() / (x.std(ddof=1) + 1e-12)) * np.sqrt(252.0)) if len(x) > 1 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown + 1e-12)) if max_drawdown < 0 else np.inf
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _write_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)


def _write_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


