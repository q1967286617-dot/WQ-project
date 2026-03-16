from __future__ import annotations

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. 等权基准（修复三项系统性缺陷，向后兼容原有接口）
# ══════════════════════════════════════════════════════════════════════════════

def build_equal_weight_benchmark(
    panel: pd.DataFrame,
    min_price: float,
    ret_col: str = "exec_ret_1d",
    cost_bps_one_way: float = 0.0,       # Fix 1: 与策略成本口径对齐
    holding_td: int = 1,                  # Fix 2: 与策略持仓期对齐
    turnover_quantile_min: float = 0.0,   # Fix 3: 与策略宇宙对齐
    exclude_div_count_le: int = 0,
) -> pd.DataFrame:
    x = panel.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    price_ok = pd.to_numeric(x["DlyPrc"], errors="coerce").abs().ge(float(min_price))
    ret_ok   = pd.to_numeric(x[ret_col],  errors="coerce").notna()

    if turnover_quantile_min > 0.0 and "turnover_5d" in x.columns:
        thr = (
            x.groupby("date")["turnover_5d"]
            .quantile(float(turnover_quantile_min))
            .rename("_bm_thr")
        )
        x = x.merge(thr, on="date", how="left")
        liq_ok = pd.to_numeric(x["turnover_5d"], errors="coerce").ge(
            pd.to_numeric(x["_bm_thr"], errors="coerce")
        )
        x = x.drop(columns=["_bm_thr"])
    else:
        liq_ok = pd.Series(True, index=x.index)

    if exclude_div_count_le > 0 and "div_count_exp" in x.columns:
        div_ok = pd.to_numeric(x["div_count_exp"], errors="coerce").fillna(0).gt(
            int(exclude_div_count_le)
        )
    else:
        div_ok = pd.Series(True, index=x.index)

    x = x[price_ok & ret_ok & liq_ok & div_ok].copy()

    dates = sorted(x["date"].dropna().unique())
    if not dates:
        return pd.DataFrame(
            columns=["date", "benchmark_ret", "benchmark_n", "is_rebal", "benchmark_nav"]
        )

    day_ret_map: dict = {}
    day_permno_set: dict = {}
    for dt, g in x.groupby("date"):
        rets = pd.to_numeric(g[ret_col], errors="coerce")
        day_ret_map[dt]    = dict(zip(g["permno"].astype(int), rets))
        day_permno_set[dt] = set(g["permno"].astype(int).tolist())

    cost_rate  = float(cost_bps_one_way) / 10_000.0
    holding_td = max(1, int(holding_td))
    rows = []

    # ── 选项一：买入持有（Buy-and-Hold within holding period） ────────────────
    # 逻辑：每隔 holding_td 天重新建仓，期间不再平衡。
    # 收益：对每只持仓股票追踪其累乘收益，最终取等权均值。
    # 成本：仅在建仓当天扣一次双边成本，持有期内不再扣。
    # 这与策略的处理方式一致：simulate_portfolio 中 cum_mult 追踪持仓累乘，
    # 建仓和平仓各扣一次固定成本。
    #
    # 具体实现：
    #   - 每个周期开始时，记录每只股票的"基准价格乘数" = 1.0
    #   - 每天将乘数 *= (1 + daily_ret)
    #   - 当天基准组合收益 = 当天所有持仓的乘数变化率的等权均值
    #     即 mean( ret_i ) —— 因为在等权买入持有下，
    #     组合日收益就是各股日收益的等权均值（权重漂移可忽略，
    #     与 simulate_portfolio 的处理口径一致）

    current_permnos: set  = set()
    cum_mults: dict       = {}   # permno -> 自建仓以来的累乘收益乘数

    for i, dt in enumerate(dates):
        is_rebal = (i % holding_td == 0)

        if is_rebal:
            new_permnos = day_permno_set.get(dt, set())

            # 计算换手成本（与原版相同）
            if not current_permnos:
                cost = cost_rate          # 首次建仓：单边买入成本
            else:
                n_exit  = len(current_permnos - new_permnos)
                n_enter = len(new_permnos - current_permnos)
                side_turnover = (
                    n_exit  / max(len(current_permnos), 1)
                  + n_enter / max(len(new_permnos), 1)
                ) / 2.0
                cost = 2.0 * cost_rate * float(min(side_turnover, 1.0))

            # 重置持仓和累乘乘数
            current_permnos = new_permnos
            cum_mults = {p: 1.0 for p in current_permnos}
        else:
            cost = 0.0   # 持有期内不扣成本

        # 当日收益：对每只持仓股取日收益，等权平均
        # （买入持有下，当日组合收益 = 各股日收益等权均值，与策略口径一致）
        day_map   = day_ret_map.get(dt, {})
        held_rets = []
        for p in current_permnos:
            r = day_map.get(p)
            if r is not None and not pd.isna(r):
                cum_mults[p] = cum_mults.get(p, 1.0) * (1.0 + float(r))
                held_rets.append(float(r))

        gross_ret = float(np.mean(held_rets)) if held_rets else 0.0
        rows.append({
            "date":          dt,
            "benchmark_ret": gross_ret - cost,
            "benchmark_n":   len(current_permnos),
            "is_rebal":      int(is_rebal),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["benchmark_ret"] = out["benchmark_ret"].fillna(0.0)
    out["benchmark_nav"] = (1.0 + out["benchmark_ret"]).cumprod()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. 三点坐标系：随机基准（下界）与 Oracle 基准（上界）
# ══════════════════════════════════════════════════════════════════════════════

def _build_tradable_universe(
    panel: pd.DataFrame,
    min_price: float,
    turnover_quantile_min: float,
    exclude_div_count_le: int,
) -> pd.DataFrame:
    """
    提取每日可交易宇宙，过滤条件与 build_daily_candidates() 完全对齐。
    随机基准和 Oracle 基准共用此函数，确保三者在相同宇宙内比较。
    """
    x = panel.copy()
    turnover_threshold = (
        x.groupby("date")["turnover_5d"]
        .quantile(float(turnover_quantile_min))
        .rename("turnover_threshold")
    )
    x = x.merge(turnover_threshold, on="date", how="left")

    price_ok  = pd.to_numeric(x["DlyPrc"],       errors="coerce").abs().ge(float(min_price))
    ret_ok    = pd.to_numeric(x["DlyRet"],        errors="coerce").notna()
    liq_ok    = pd.to_numeric(x["turnover_5d"],   errors="coerce").ge(
                    pd.to_numeric(x["turnover_threshold"], errors="coerce"))
    div_count = pd.to_numeric(x["div_count_exp"], errors="coerce").fillna(0)
    div_ok    = div_count.gt(int(exclude_div_count_le))

    return x[price_ok & ret_ok & liq_ok & div_ok].drop(columns=["turnover_threshold"]).copy()


def _to_candidates_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一列顺序，与 build_daily_candidates() 输出格式一致。"""
    cols_front = [
        "date", "permno", "prob", "signal_rank", "signal_group",
        "industry", "div_count_exp", "gap_cv_exp", "z_to_med_exp",
        "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d", "y_div_10d",
        "DlyOpen", "DlyClose",
    ]
    cols_front = [c for c in cols_front if c in df.columns]
    rest = [c for c in df.columns if c not in cols_front]
    return df[cols_front + rest]


def build_random_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float,
    turnover_quantile_min: float,
    exclude_div_count_le: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    随机基准候选生成器（策略下界）。

    每日从与策略相同的可交易宇宙中随机抽取 top_k 只股票，
    完全不使用任何模型信号，代表「零选股能力」水平。

    经济含义：策略若无法跑赢此基准，说明模型贡献为负。
    返回与 build_daily_candidates() 相同格式的 DataFrame，
    可直接传入 simulate_portfolio()。
    """
    universe = _build_tradable_universe(
        panel, min_price, turnover_quantile_min, exclude_div_count_le
    )
    rng = np.random.default_rng(int(seed))
    chosen = []

    for dt, g in universe.groupby("date", sort=True):
        if len(g) == 0:
            continue
        n   = min(int(top_k), len(g))
        idx = rng.choice(len(g), size=n, replace=False)
        day = g.iloc[idx].copy()
        day["prob"]         = 0.5
        day["signal_group"] = "random"
        day["signal_rank"]  = np.arange(1, len(day) + 1)
        chosen.append(day)

    if not chosen:
        return universe.head(0).copy()
    return _to_candidates_df(pd.concat(chosen, ignore_index=True))


def build_oracle_candidates(
    panel: pd.DataFrame,
    top_k: int,
    min_price: float,
    turnover_quantile_min: float,
    exclude_div_count_le: int,
    label_col: str = "y_div_10d",
) -> pd.DataFrame:
    """
    Oracle 基准候选生成器（模型预测能力上界）。

    每日只选 label=1（真实将在未来 H 日内分红）的可交易股票，
    代表「完美预测」场景，即模型能力的理论天花板。

    ⚠️  使用了未来标签，严格用于事后评估，绝不可用于实盘。
    返回与 build_daily_candidates() 相同格式的 DataFrame。
    """
    if label_col not in panel.columns:
        raise ValueError(
            f"Oracle benchmark requires column '{label_col}' in panel. "
            "Ensure forward labels are computed before calling this function."
        )
    universe = _build_tradable_universe(
        panel, min_price, turnover_quantile_min, exclude_div_count_le
    )
    chosen = []

    for dt, g in universe.groupby("date", sort=True):
        oracle_pool = (
            g[pd.to_numeric(g[label_col], errors="coerce").fillna(0).eq(1)]
            .sort_values("div_count_exp", ascending=False)
            .head(int(top_k))
        )
        if len(oracle_pool) == 0:
            continue
        day = oracle_pool.copy()
        day["prob"]         = 1.0
        day["signal_group"] = "oracle"
        day["signal_rank"]  = np.arange(1, len(day) + 1)
        chosen.append(day)

    if not chosen:
        return universe.head(0).copy()
    return _to_candidates_df(pd.concat(chosen, ignore_index=True))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Alpha 捕获率
# ══════════════════════════════════════════════════════════════════════════════

def compute_alpha_capture(
    strategy_metrics: dict,
    random_metrics: dict,
    oracle_metrics: dict,
) -> dict:
    """
    计算策略在随机基准（下界）与 Oracle 基准（上界）之间的 Alpha 捕获率。

    公式：
        Alpha 捕获率 = (策略指标 - 随机基准) / (Oracle 基准 - 随机基准)

    解读：
        1.0  → 达到完美预测水平
        0.5  → 捕获了一半可用 Alpha
        0.0  → 与随机无异，模型无贡献
        <0   → 差于随机，策略设计存在问题

    参数中的 *_metrics 来自 _return_metrics()，
    包含 total_return, annualized_return, sharpe 等字段。
    """
    def _one(key: str) -> dict:
        s = strategy_metrics.get(key)
        r = random_metrics.get(key)
        o = oracle_metrics.get(key)
        if any(v is None for v in [s, r, o]):
            return {"error": "missing metric"}
        denom = float(o) - float(r)
        if abs(denom) < 1e-9:
            return {
                "strategy":            float(s),
                "random_baseline":     float(r),
                "oracle_ceiling":      float(o),
                "alpha_capture_ratio": None,
                "note":                "oracle ≈ random, capture undefined",
            }
        capture = (float(s) - float(r)) / denom
        return {
            "strategy":            round(float(s), 6),
            "random_baseline":     round(float(r), 6),
            "oracle_ceiling":      round(float(o), 6),
            "alpha_capture_ratio": round(float(capture), 4),
            "pct_of_oracle_gap":   f"{capture * 100:.1f}%",
        }

    result = {k: _one(k) for k in ["total_return", "annualized_return", "sharpe"]}

    main  = result.get("annualized_return", {})
    ratio = main.get("alpha_capture_ratio")
    if ratio is not None:
        if ratio >= 0.8:
            grade = "Excellent  (captures >=80% of oracle gap)"
        elif ratio >= 0.5:
            grade = "Good       (captures 50-80% of oracle gap)"
        elif ratio >= 0.2:
            grade = "Fair       (captures 20-50% of oracle gap)"
        elif ratio >= 0.0:
            grade = "Weak       (captures 0-20% of oracle gap)"
        else:
            grade = "Below random baseline -- strategy design needs review"
        result["overall_grade"] = grade
        result["primary_alpha_capture_ratio"] = ratio

    return result