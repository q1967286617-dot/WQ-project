from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_PRED_COLS = ("date", "permno", "prob")
REQUIRED_SPLIT_COLS = (
    "DlyCalDt",
    "PERMNO",
    "DlyRet",
    "DlyPrc",
    "turnover_5d",
    "SICCD",
    "industry",
    "has_div_history",
    "div_count_exp",
    "gap_cv_exp",
    "gap_med_exp",
    "days_since_last_div",
    "z_to_med_exp",
    "y_div_10d",
)
BACKTEST_PANEL_COLS = tuple(c for c in REQUIRED_SPLIT_COLS if c not in {"DlyCalDt", "PERMNO"})
RAW_PRICE_COLS = ("DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk")


@dataclass(frozen=True)
class ResearchDecision:
    use_dividend_rules: bool
    support_votes: int
    total_checks: int
    checks: Dict[str, bool]


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def build_backtest_panel(preds_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(preds_df, REQUIRED_PRED_COLS, "preds_df")
    _require_columns(split_df, REQUIRED_SPLIT_COLS, "split_df")

    preds = preds_df.copy()
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    preds["permno"] = preds["permno"].astype(int)

    split = split_df.copy()
    split["DlyCalDt"] = pd.to_datetime(split["DlyCalDt"], errors="coerce")
    split["PERMNO"] = split["PERMNO"].astype(int)

    merged = preds.merge(
        split[list(REQUIRED_SPLIT_COLS)],
        left_on=["date", "permno"],
        right_on=["DlyCalDt", "PERMNO"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__split"),
    )
    merged = merged.drop(columns=["DlyCalDt", "PERMNO"])

    for col in BACKTEST_PANEL_COLS:
        split_col = f"{col}__split"
        if split_col in merged.columns:
            # Split-side fields are the authoritative raw panel values.
            merged[col] = merged[split_col]
            merged = merged.drop(columns=[split_col])

    _require_columns(merged, REQUIRED_PRED_COLS + BACKTEST_PANEL_COLS, "backtest_panel")
    return merged.sort_values(["date", "permno"]).reset_index(drop=True)


def merge_execution_price_data(panel: pd.DataFrame, raw_price_df: pd.DataFrame) -> pd.DataFrame:
    if raw_price_df.empty:
        return panel

    required = ("date", "permno") + RAW_PRICE_COLS
    _require_columns(raw_price_df, required, "raw_price_df")

    raw = raw_price_df.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["permno"] = raw["permno"].astype(int)

    x = panel.copy()
    merged = x.merge(raw[list(required)], on=["date", "permno"], how="left", suffixes=("", "__raw"))
    for col in RAW_PRICE_COLS:
        raw_col = f"{col}__raw"
        if raw_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[raw_col])
                merged = merged.drop(columns=[raw_col])
            else:
                merged = merged.rename(columns={raw_col: col})
    return merged


def infer_execution_basis(panel: pd.DataFrame) -> str:
    if "DlyOpen" in panel.columns and pd.to_numeric(panel["DlyOpen"], errors="coerce").notna().any():
        return "open_to_open"
    return "close_to_close_delayed"


def add_execution_returns(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy().sort_values(["permno", "date"]).reset_index(drop=True)
    basis = infer_execution_basis(x)
    if basis == "open_to_open":
        open_px = pd.to_numeric(x["DlyOpen"], errors="coerce").where(lambda s: s > 0)
        x["exec_ret_1d"] = open_px.groupby(x["permno"]).pct_change(fill_method=None)
    else:
        x["exec_ret_1d"] = pd.to_numeric(x["DlyRet"], errors="coerce")
    return x


def add_forward_returns(
    panel: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 10),
) -> pd.DataFrame:
    x = panel.copy().sort_values(["permno", "date"]).reset_index(drop=True)
    basis = infer_execution_basis(x)

    if basis == "open_to_open":
        open_px = pd.to_numeric(x["DlyOpen"], errors="coerce").where(lambda s: s > 0)
        g = open_px.groupby(x["permno"])
        for h in horizons:
            col = f"fwd_ret_{int(h)}d"
            x[col] = g.shift(-(int(h) + 1)) / g.shift(-1) - 1.0
        return x

    gross = 1.0 + pd.to_numeric(x["DlyRet"], errors="coerce").fillna(0.0)
    for h in horizons:
        col = f"fwd_ret_{int(h)}d"
        x[col] = (
            gross.groupby(x["permno"])
            .transform(lambda s, h=h: s.shift(-1).rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1.0)
        )
    return x


def compute_stable_gap_cv_threshold(
    reference_df: pd.DataFrame,
    stable_div_count_min: int = 4,
    quantile: float = 0.5,
) -> float:
    x = reference_df.copy()
    mask = (pd.to_numeric(x["div_count_exp"], errors="coerce") >= stable_div_count_min) & pd.to_numeric(
        x["gap_cv_exp"], errors="coerce"
    ).notna()
    if not mask.any():
        return float("inf")
    return float(pd.to_numeric(x.loc[mask, "gap_cv_exp"], errors="coerce").quantile(quantile))


def _qbucket(s: pd.Series, labels: Sequence[str]) -> pd.Series:
    valid = pd.to_numeric(s, errors="coerce")
    if valid.notna().sum() < len(labels):
        return pd.Series(["unknown"] * len(s), index=s.index)
    ranked = valid.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype("string").fillna("unknown")


def _near_zero_bucket(z: pd.Series) -> pd.Series:
    x = pd.to_numeric(z, errors="coerce").abs()
    if x.notna().sum() < 3:
        return pd.Series(["unknown"] * len(z), index=z.index)
    ranked = x.rank(method="first")
    return pd.qcut(ranked, q=3, labels=["near", "mid", "far"]).astype("string").fillna("unknown")


def _compare_groups(df: pd.DataFrame, col: str, good_key: str, weak_key: str) -> bool:
    if df.empty or col not in df.columns:
        return False
    x = df.set_index(col)
    if good_key not in x.index or weak_key not in x.index:
        return False
    good = x.loc[good_key]
    weak = x.loc[weak_key]
    return bool(
        pd.notna(good.get("mean_fwd_ret_10d"))
        and pd.notna(weak.get("mean_fwd_ret_10d"))
        and pd.notna(good.get("hit_rate"))
        and pd.notna(weak.get("hit_rate"))
        and float(good["mean_fwd_ret_10d"]) >= float(weak["mean_fwd_ret_10d"])
        and float(good["hit_rate"]) >= float(weak["hit_rate"])
    )


def run_signal_research(
    panel: pd.DataFrame,
    high_prob_threshold: float,
) -> Tuple[Dict[str, pd.DataFrame], ResearchDecision]:
    x = panel.copy()
    x["prob_bucket"] = _qbucket(x["prob"], labels=["low", "mid", "high", "very_high"])
    x["history_group"] = np.where(pd.to_numeric(x["div_count_exp"], errors="coerce") <= 1, "<=1", ">=2")
    x["gap_cv_bucket"] = _qbucket(x["gap_cv_exp"], labels=["low", "mid", "high"])
    x["z_phase_bucket"] = _near_zero_bucket(x["z_to_med_exp"])

    reports: Dict[str, pd.DataFrame] = {}
    reports["prob_buckets"] = (
        x.groupby("prob_bucket", dropna=False)[["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d", "y_div_10d"]]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    high_prob = x[pd.to_numeric(x["prob"], errors="coerce") >= float(high_prob_threshold)].copy()
    if high_prob.empty:
        history_cmp = pd.DataFrame(columns=["history_group", "mean_fwd_ret_10d", "hit_rate", "n"])
        gap_cmp = pd.DataFrame(columns=["gap_cv_bucket", "mean_fwd_ret_10d", "hit_rate", "n"])
        z_cmp = pd.DataFrame(columns=["z_phase_bucket", "mean_fwd_ret_10d", "hit_rate", "n"])
    else:
        history_cmp = (
            high_prob.groupby("history_group", dropna=False)
            .agg(mean_fwd_ret_10d=("fwd_ret_10d", "mean"), hit_rate=("y_div_10d", "mean"), n=("permno", "size"))
            .reset_index()
        )
        gap_cmp = (
            high_prob[high_prob["history_group"] == ">=2"]
            .groupby("gap_cv_bucket", dropna=False)
            .agg(mean_fwd_ret_10d=("fwd_ret_10d", "mean"), hit_rate=("y_div_10d", "mean"), n=("permno", "size"))
            .reset_index()
        )
        z_cmp = (
            high_prob.groupby("z_phase_bucket", dropna=False)
            .agg(mean_fwd_ret_10d=("fwd_ret_10d", "mean"), hit_rate=("y_div_10d", "mean"), n=("permno", "size"))
            .reset_index()
        )
    reports["history_groups"] = history_cmp
    reports["gap_cv_groups"] = gap_cmp
    reports["z_phase_groups"] = z_cmp

    checks = {
        "history_filter": _compare_groups(history_cmp, "history_group", ">=2", "<=1"),
        "gap_cv_filter": _compare_groups(gap_cmp, "gap_cv_bucket", "low", "high"),
        "z_phase_filter": _compare_groups(z_cmp, "z_phase_bucket", "near", "far"),
    }
    support_votes = int(sum(bool(v) for v in checks.values()))
    return reports, ResearchDecision(
        use_dividend_rules=support_votes >= 2,
        support_votes=support_votes,
        total_checks=len(checks),
        checks=checks,
    )


def prepare_candidate_pool(
    panel: pd.DataFrame,
    stable_gap_cv_threshold: float,
    turnover_quantile_min: float,
    exclude_div_count_le: int,
    min_price: float,
    stable_div_count_min: int,
    stable_prob_threshold: float,
    regular_prob_threshold: float,
    use_dividend_rules: bool,
    require_prob_thresholds: bool = True,
) -> pd.DataFrame:
    x = panel.copy().sort_values(["date", "prob"], ascending=[True, False]).reset_index(drop=True)
    turnover_threshold = x.groupby("date")["turnover_5d"].quantile(turnover_quantile_min).rename("turnover_threshold")
    x = x.merge(turnover_threshold, on="date", how="left")
    x["tradable"] = (
        pd.to_numeric(x["DlyPrc"], errors="coerce").abs().ge(float(min_price))
        & pd.to_numeric(x["DlyRet"], errors="coerce").notna()
        & pd.to_numeric(x["turnover_5d"], errors="coerce").ge(pd.to_numeric(x["turnover_threshold"], errors="coerce"))
    )

    div_count = pd.to_numeric(x["div_count_exp"], errors="coerce").fillna(0)
    gap_cv = pd.to_numeric(x["gap_cv_exp"], errors="coerce")
    x["signal_group"] = np.where(
        (div_count >= stable_div_count_min) & gap_cv.notna() & (gap_cv <= stable_gap_cv_threshold),
        "stable",
        np.where(div_count > exclude_div_count_le, "regular", "no_history"),
    )

    if use_dividend_rules and require_prob_thresholds:
        x["passes_dividend_rule"] = np.where(
            x["signal_group"].eq("stable"),
            x["prob"] >= stable_prob_threshold,
            np.where(x["signal_group"].eq("regular"), x["prob"] >= regular_prob_threshold, False),
        )
        x["eligible"] = x["tradable"] & x["passes_dividend_rule"] & x["signal_group"].ne("no_history")
    elif use_dividend_rules:
        x["passes_dividend_rule"] = x["signal_group"].ne("no_history")
        x["eligible"] = x["tradable"] & x["signal_group"].ne("no_history")
    else:
        x["passes_dividend_rule"] = x["prob"] >= regular_prob_threshold
        x["eligible"] = x["tradable"] & x["passes_dividend_rule"] if require_prob_thresholds else x["tradable"]

    return x


def select_top_k_from_pool(
    pool: pd.DataFrame,
    top_k: int,
    max_industry_weight: float,
    ranking_mode: str = "prob",
    seed: int = 42,
) -> pd.DataFrame:
    x = pool[pool["eligible"]].copy()
    if x.empty:
        return pool.head(0).copy()

    ranking_mode = str(ranking_mode).lower()
    rng = np.random.default_rng(int(seed))
    chosen: List[pd.DataFrame] = []
    cap_names = max(1, int(np.floor(max_industry_weight * top_k))) if max_industry_weight > 0 else top_k

    for _, g in x.groupby("date", sort=True):
        rows = []
        counts: Dict[str, int] = {}
        day = g.copy()

        if ranking_mode == "prob":
            ranked = day.sort_values(["prob", "permno"], ascending=[False, True])
        elif ranking_mode == "random":
            day["_rand"] = rng.random(len(day))
            ranked = day.sort_values(["_rand", "permno"], ascending=[False, True])
        elif ranking_mode == "event":
            hit = pd.to_numeric(day.get("y_div_10d"), errors="coerce").fillna(0)
            day["_oracle_event_hit"] = hit.astype(int)
            ranked = day.sort_values(["_oracle_event_hit", "permno"], ascending=[False, True])
        elif ranking_mode == "non_prob":
            div_count = pd.to_numeric(day.get("div_count_exp"), errors="coerce").fillna(0)
            gap_cv = pd.to_numeric(day.get("gap_cv_exp"), errors="coerce").fillna(np.inf)
            abs_z = pd.to_numeric(day.get("z_to_med_exp"), errors="coerce").abs().fillna(np.inf)
            turnover = pd.to_numeric(day.get("turnover_5d"), errors="coerce").fillna(-np.inf)
            group_rank = np.where(day["signal_group"].eq("stable"), 2, np.where(day["signal_group"].eq("regular"), 1, 0))
            day["_non_prob_group_rank"] = group_rank
            day["_non_prob_div_count"] = div_count
            day["_non_prob_gap_cv"] = gap_cv
            day["_non_prob_abs_z"] = abs_z
            day["_non_prob_turnover"] = turnover
            ranked = day.sort_values(
                [
                    "_non_prob_group_rank",
                    "_non_prob_div_count",
                    "_non_prob_gap_cv",
                    "_non_prob_abs_z",
                    "_non_prob_turnover",
                    "permno",
                ],
                ascending=[False, False, True, True, False, True],
            )
        else:
            raise ValueError(f"Unknown ranking_mode: {ranking_mode}")

        for _, r in ranked.iterrows():
            industry = str(r["industry"]) if pd.notna(r["industry"]) else "unknown"
            if counts.get(industry, 0) >= cap_names:
                continue
            rows.append(r)
            counts[industry] = counts.get(industry, 0) + 1
            if len(rows) >= top_k:
                break

        if rows:
            out_day = pd.DataFrame(rows).copy()
            out_day["signal_rank"] = np.arange(1, len(out_day) + 1)
            chosen.append(out_day)

    if not chosen:
        return pool.head(0).copy()

    out = pd.concat(chosen, ignore_index=True)
    out = out.drop(
        columns=[
            "_rand",
            "_oracle_event_hit",
            "_non_prob_group_rank",
            "_non_prob_div_count",
            "_non_prob_gap_cv",
            "_non_prob_abs_z",
            "_non_prob_turnover",
        ],
        errors="ignore",
    )
    return out


def build_daily_candidates(
    panel: pd.DataFrame,
    top_k: int,
    stable_gap_cv_threshold: float,
    turnover_quantile_min: float,
    exclude_div_count_le: int,
    min_price: float,
    stable_div_count_min: int,
    stable_prob_threshold: float,
    regular_prob_threshold: float,
    max_industry_weight: float,
    use_dividend_rules: bool,
) -> pd.DataFrame:
    x = prepare_candidate_pool(
        panel=panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
        min_price=min_price,
        stable_div_count_min=stable_div_count_min,
        stable_prob_threshold=stable_prob_threshold,
        regular_prob_threshold=regular_prob_threshold,
        use_dividend_rules=use_dividend_rules,
    )
    out = select_top_k_from_pool(
        pool=x,
        top_k=top_k,
        max_industry_weight=max_industry_weight,
        ranking_mode="prob",
    )
    if out.empty:
        return x.head(0).copy()
    cols_front = [
        "date",
        "permno",
        "prob",
        "signal_rank",
        "signal_group",
        "industry",
        "div_count_exp",
        "gap_cv_exp",
        "z_to_med_exp",
        "fwd_ret_1d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "y_div_10d",
        "DlyOpen",
        "DlyClose",
    ]
    cols_front = [c for c in cols_front if c in out.columns]
    rest = [c for c in out.columns if c not in cols_front]
    return out[cols_front + rest]

