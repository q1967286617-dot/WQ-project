from __future__ import annotations

import numpy as np
import pandas as pd

from .signal import prepare_candidate_pool, select_top_k_from_pool


def _reference_universe(
    panel: pd.DataFrame,
    min_price: float,
    ret_col: str = "exec_ret_1d",
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
) -> pd.DataFrame:
    x = panel.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    price_ok = pd.to_numeric(x["DlyPrc"], errors="coerce").abs().ge(float(min_price))
    ret_ok = pd.to_numeric(x[ret_col], errors="coerce").notna()
    mask = price_ok & ret_ok

    if turnover_quantile_min > 0.0 and "turnover_5d" in x.columns:
        turnover = pd.to_numeric(x["turnover_5d"], errors="coerce")
        threshold = (
            x.assign(_turnover=turnover)
            .groupby("date")["_turnover"]
            .transform(lambda s: s.quantile(float(turnover_quantile_min)))
        )
        mask &= turnover.ge(threshold)

    if exclude_div_count_le > 0 and "div_count_exp" in x.columns:
        div_count = pd.to_numeric(x["div_count_exp"], errors="coerce").fillna(0)
        mask &= div_count.gt(int(exclude_div_count_le))

    x = x.loc[mask].copy()
    if x.empty:
        return x

    x["permno"] = pd.to_numeric(x["permno"], errors="coerce").astype("Int64")
    x = x[x["permno"].notna()].copy()
    x["permno"] = x["permno"].astype(int)

    if "industry" not in x.columns:
        x["industry"] = "unknown"
    x["industry"] = x["industry"].astype("string").fillna("unknown").astype(str)

    if "y_div_10d" in x.columns:
        x["y_div_10d"] = pd.to_numeric(x["y_div_10d"], errors="coerce").fillna(0).astype(int)

    return x.sort_values(["date", "permno"]).reset_index(drop=True)


def _format_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

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
    cols_front = [c for c in cols_front if c in df.columns]
    rest = [c for c in df.columns if c not in cols_front]
    return df[cols_front + rest].reset_index(drop=True)


def build_equal_weight_benchmark(
    panel: pd.DataFrame,
    min_price: float,
    ret_col: str = "exec_ret_1d",
    cost_bps_one_way: float = 0.0,
    holding_td: int = 1,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
) -> pd.DataFrame:
    x = _reference_universe(
        panel=panel,
        min_price=min_price,
        ret_col=ret_col,
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
    )
    if x.empty:
        return pd.DataFrame(
            columns=["date", "benchmark_ret", "benchmark_n", "is_rebal", "benchmark_nav"]
        )

    dates = sorted(x["date"].dropna().unique())
    if not dates:
        return pd.DataFrame(
            columns=["date", "benchmark_ret", "benchmark_n", "is_rebal", "benchmark_nav"]
        )

    day_ret_map: dict[pd.Timestamp, dict[int, float]] = {}
    day_permno_set: dict[pd.Timestamp, set[int]] = {}
    for dt, g in x.groupby("date"):
        rets = pd.to_numeric(g[ret_col], errors="coerce")
        day_ret_map[dt] = dict(zip(g["permno"].astype(int), rets))
        day_permno_set[dt] = set(g["permno"].astype(int).tolist())

    cost_rate = float(cost_bps_one_way) / 10_000.0
    holding_td = max(1, int(holding_td))

    current_permnos: set[int] = set()
    rows = []

    for i, dt in enumerate(dates):
        is_rebal = i % holding_td == 0

        if is_rebal:
            new_permnos = day_permno_set.get(dt, set())

            if not current_permnos:
                cost = cost_rate
            else:
                n_exit = len(current_permnos - new_permnos)
                n_enter = len(new_permnos - current_permnos)
                side_turnover = (
                    n_exit / max(len(current_permnos), 1)
                    + n_enter / max(len(new_permnos), 1)
                ) / 2.0
                cost = 2.0 * cost_rate * float(min(side_turnover, 1.0))

            current_permnos = new_permnos
        else:
            cost = 0.0

        day_map = day_ret_map.get(dt, {})
        held_rets = [
            v for k, v in day_map.items()
            if k in current_permnos and not pd.isna(v)
        ]
        gross_ret = float(np.mean(held_rets)) if held_rets else 0.0

        rows.append(
            {
                "date": dt,
                "benchmark_ret": gross_ret - cost,
                "benchmark_n": len(current_permnos),
                "is_rebal": int(is_rebal),
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["benchmark_ret"] = out["benchmark_ret"].fillna(0.0)
    out["benchmark_nav"] = (1.0 + out["benchmark_ret"]).cumprod()
    return out


def build_random_candidates(
    panel: pd.DataFrame,
    top_k: int,
    seed: int = 42,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
) -> pd.DataFrame:
    x = _reference_universe(
        panel=panel,
        min_price=min_price,
        ret_col="exec_ret_1d",
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
    )
    if x.empty:
        return x

    rng = np.random.default_rng(int(seed))
    out_parts: list[pd.DataFrame] = []

    for _, g in x.groupby("date", sort=True):
        idx = rng.permutation(len(g))
        day = g.iloc[idx].copy()
        day["prob"] = rng.random(len(day))
        day = day.sort_values(["prob", "permno"], ascending=[False, True]).head(int(top_k)).copy()
        day["signal_rank"] = np.arange(1, len(day) + 1)
        day["signal_group"] = "random"
        out_parts.append(day)

    if not out_parts:
        return x.head(0).copy()
    return _format_candidates(pd.concat(out_parts, ignore_index=True))


def build_non_prob_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
    stable_gap_cv_threshold: float = np.inf,
    stable_div_count_min: int = 4,
    stable_prob_threshold: float = 0.45,
    regular_prob_threshold: float = 0.55,
    max_industry_weight: float = 1.0,
    use_dividend_rules: bool = False,
) -> pd.DataFrame:
    pool = prepare_candidate_pool(
        panel=panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
        min_price=min_price,
        stable_div_count_min=stable_div_count_min,
        stable_prob_threshold=stable_prob_threshold,
        regular_prob_threshold=regular_prob_threshold,
        use_dividend_rules=use_dividend_rules,
        require_prob_thresholds=False,
    )
    selected = select_top_k_from_pool(
        pool=pool,
        top_k=top_k,
        max_industry_weight=max_industry_weight,
        ranking_mode="non_prob",
    )
    if selected.empty:
        return selected
    return _format_candidates(selected)


def build_random_prob_candidates(
    panel: pd.DataFrame,
    top_k: int,
    seed: int = 42,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
    stable_gap_cv_threshold: float = np.inf,
    stable_div_count_min: int = 4,
    stable_prob_threshold: float = 0.45,
    regular_prob_threshold: float = 0.55,
    max_industry_weight: float = 1.0,
    use_dividend_rules: bool = False,
) -> pd.DataFrame:
    pool = prepare_candidate_pool(
        panel=panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
        min_price=min_price,
        stable_div_count_min=stable_div_count_min,
        stable_prob_threshold=stable_prob_threshold,
        regular_prob_threshold=regular_prob_threshold,
        use_dividend_rules=use_dividend_rules,
        require_prob_thresholds=True,
    )
    selected = select_top_k_from_pool(
        pool=pool,
        top_k=top_k,
        max_industry_weight=max_industry_weight,
        ranking_mode="random",
        seed=seed,
    )
    if selected.empty:
        return selected
    return _format_candidates(selected)


def _best_return_col(x: pd.DataFrame, holding_td: int | None = None) -> str | None:
    candidates: list[str] = []
    if holding_td is not None:
        candidates.append(f"fwd_ret_{int(holding_td)}d")
    candidates.extend(["fwd_ret_10d", "fwd_ret_5d", "fwd_ret_1d", "exec_ret_1d"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in x.columns:
            return candidate
    return None


def build_oracle_event_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
    stable_gap_cv_threshold: float | None = None,
    stable_div_count_min: int = 4,
    stable_prob_threshold: float = 0.45,
    regular_prob_threshold: float = 0.55,
    max_industry_weight: float = 1.0,
    use_dividend_rules: bool = False,
) -> pd.DataFrame:
    if stable_gap_cv_threshold is not None:
        x = prepare_candidate_pool(
            panel=panel,
            stable_gap_cv_threshold=float(stable_gap_cv_threshold),
            turnover_quantile_min=turnover_quantile_min,
            exclude_div_count_le=exclude_div_count_le,
            min_price=min_price,
            stable_div_count_min=stable_div_count_min,
            stable_prob_threshold=stable_prob_threshold,
            regular_prob_threshold=regular_prob_threshold,
            use_dividend_rules=use_dividend_rules,
        )
        selected = select_top_k_from_pool(
            pool=x,
            top_k=top_k,
            max_industry_weight=max_industry_weight,
            ranking_mode="event",
        )
        if selected.empty:
            return selected
        selected["oracle_event_hit"] = pd.to_numeric(selected["y_div_10d"], errors="coerce").fillna(0).astype(int)
        return _format_candidates(selected)

    x = _reference_universe(
        panel=panel,
        min_price=min_price,
        ret_col="exec_ret_1d",
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
    )
    if x.empty:
        return x

    if "y_div_10d" not in x.columns:
        x["y_div_10d"] = 0

    x["_oracle_hit"] = pd.to_numeric(x["y_div_10d"], errors="coerce").fillna(0).astype(int)

    out_parts: list[pd.DataFrame] = []
    for _, g in x.groupby("date", sort=True):
        day = g.sort_values(
            ["_oracle_hit", "permno"],
            ascending=[False, True],
        ).head(int(top_k)).copy()
        if day.empty:
            continue
        day["prob"] = np.where(day["_oracle_hit"].gt(0), 1.0, 0.0)
        day["signal_group"] = np.where(day["_oracle_hit"].gt(0), "oracle_hit", "oracle_miss")
        day["signal_rank"] = np.arange(1, len(day) + 1)
        out_parts.append(day)

    if not out_parts:
        return x.head(0).copy()

    out = pd.concat(out_parts, ignore_index=True).drop(columns=["_oracle_hit"], errors="ignore")
    return _format_candidates(out)


def build_oracle_return_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
    holding_td: int = 10,
) -> pd.DataFrame:
    x = _reference_universe(
        panel=panel,
        min_price=min_price,
        ret_col="exec_ret_1d",
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
    )
    if x.empty:
        return x

    score_ret_col = _best_return_col(x, holding_td=holding_td)
    if score_ret_col is None:
        x["_oracle_ret"] = -np.inf
    else:
        x["_oracle_ret"] = pd.to_numeric(x[score_ret_col], errors="coerce").fillna(-np.inf)

    out_parts: list[pd.DataFrame] = []
    for _, g in x.groupby("date", sort=True):
        day = g.sort_values(
            ["_oracle_ret", "permno"],
            ascending=[False, True],
        ).head(int(top_k)).copy()
        if day.empty:
            continue
        rank = np.arange(len(day), 0, -1, dtype=float)
        day["prob"] = rank / rank.max()
        day["signal_rank"] = np.arange(1, len(day) + 1)
        day["signal_group"] = "oracle_return"
        out_parts.append(day)

    if not out_parts:
        return x.head(0).copy()

    out = pd.concat(out_parts, ignore_index=True).drop(columns=["_oracle_ret"], errors="ignore")
    return _format_candidates(out)


def build_oracle_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float = 0.0,
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
    mode: str = "event",
    holding_td: int = 10,
    stable_gap_cv_threshold: float | None = None,
    stable_div_count_min: int = 4,
    stable_prob_threshold: float = 0.45,
    regular_prob_threshold: float = 0.55,
    max_industry_weight: float = 1.0,
    use_dividend_rules: bool = False,
) -> pd.DataFrame:
    mode = str(mode).lower()
    if mode == "event":
        return build_oracle_event_candidates(
            panel=panel,
            top_k=top_k,
            min_price=min_price,
            turnover_quantile_min=turnover_quantile_min,
            exclude_div_count_le=exclude_div_count_le,
            stable_gap_cv_threshold=stable_gap_cv_threshold,
            stable_div_count_min=stable_div_count_min,
            stable_prob_threshold=stable_prob_threshold,
            regular_prob_threshold=regular_prob_threshold,
            max_industry_weight=max_industry_weight,
            use_dividend_rules=use_dividend_rules,
        )
    if mode != "return":
        raise ValueError(f"Unknown oracle mode: {mode}")
    return build_oracle_return_candidates(
        panel=panel,
        top_k=top_k,
        min_price=min_price,
        turnover_quantile_min=turnover_quantile_min,
        exclude_div_count_le=exclude_div_count_le,
        holding_td=holding_td,
    )


def compute_alpha_capture(
    strategy_metrics: dict,
    random_metrics: dict,
    oracle_metrics: dict,
) -> dict:
    def _ratio(metric: str) -> tuple[float | None, float | None]:
        strategy = strategy_metrics.get(metric)
        floor = random_metrics.get(metric)
        ceiling = oracle_metrics.get(metric)
        if strategy is None or floor is None or ceiling is None:
            return None, None

        strategy = float(strategy)
        floor = float(floor)
        ceiling = float(ceiling)
        span = ceiling - floor
        if not np.isfinite(span) or abs(span) < 1e-12:
            return None, None

        raw = (strategy - floor) / span
        clipped = float(np.clip(raw, 0.0, 1.0))
        return raw, clipped

    primary_raw, primary_ratio = _ratio("annualized_return")
    sharpe_raw, sharpe_ratio = _ratio("sharpe")
    total_raw, total_ratio = _ratio("total_return")

    grade = "N/A"
    if primary_ratio is not None:
        if primary_ratio >= 0.8:
            grade = "A"
        elif primary_ratio >= 0.6:
            grade = "B"
        elif primary_ratio >= 0.4:
            grade = "C"
        elif primary_ratio >= 0.2:
            grade = "D"
        else:
            grade = "F"

    return {
        "primary_metric": "annualized_return",
        "primary_alpha_capture_ratio": primary_ratio,
        "primary_alpha_capture_raw": primary_raw,
        "secondary_capture_ratios": {
            "sharpe": sharpe_ratio,
            "total_return": total_ratio,
        },
        "secondary_capture_raw": {
            "sharpe": sharpe_raw,
            "total_return": total_raw,
        },
        "overall_grade": grade,
    }
