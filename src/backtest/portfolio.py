from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ActivePosition:
    permno: int
    industry: str
    signal_group: str
    prob: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_idx: int
    active_end_idx: int
    planned_active_end_idx: int
    hit_label: int
    trade_id: str
    entry_spread_rate: float = 0.0
    cum_mult: float = 1.0


def simulate_portfolio(
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int,
    holding_td: int,
    cooldown_td: int,
    cost_bps_one_way: float,
    max_industry_weight: float,
    ret_col: str = "exec_ret_1d",
    use_bid_ask_spread: bool = False,
    spread_cost_cap_bps_one_way: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = panel.copy().sort_values(["date", "permno"]).reset_index(drop=True)
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    dates = list(pd.Index(sorted(x["date"].dropna().unique())))
    date_to_idx = {dt: i for i, dt in enumerate(dates)}

    day_frames = {dt: g.set_index("permno", drop=False) for dt, g in x.groupby("date", sort=True)}
    day_permnos = {dt: set(frame.index.tolist()) for dt, frame in day_frames.items()}
    last_idx_by_permno = x.groupby("permno")["date"].max().map(date_to_idx).to_dict()
    planned_entries = _plan_entries(candidates, dates)

    weight = 1.0 / float(top_k)
    cost_rate = float(cost_bps_one_way) / 10000.0
    spread_cap = None if spread_cost_cap_bps_one_way is None else float(spread_cost_cap_bps_one_way) / 10000.0
    cap_names = max(1, int(np.floor(max_industry_weight * top_k))) if max_industry_weight > 0 else top_k

    active: Dict[int, ActivePosition] = {}
    cooldown_until: Dict[int, int] = {}
    trade_rows: List[Dict] = []
    position_rows: List[Dict] = []
    daily_rows: List[Dict] = []

    for idx, dt in enumerate(dates):
        entries_today: List[ActivePosition] = []
        industry_counts = _industry_counts(active)
        available_permnos = day_permnos.get(dt, set())
        day_frame = day_frames.get(dt)

        entry_fixed_cost = 0.0
        entry_spread_cost = 0.0
        exit_fixed_cost = 0.0
        exit_spread_cost = 0.0
        exit_spread_by_trade: Dict[str, float] = {}

        for row in planned_entries.get(dt, []):
            permno = int(row["permno"])
            if len(active) >= top_k:
                break
            if permno in active:
                continue
            if idx <= cooldown_until.get(permno, -10**9):
                continue
            if permno not in available_permnos:
                continue

            industry = str(row.get("industry", "unknown"))
            if industry_counts.get(industry, 0) >= cap_names:
                continue

            spread_rate = 0.0
            if use_bid_ask_spread and day_frame is not None and permno in day_frame.index:
                spread_rate = _quote_half_spread_rate(_select_row(day_frame, permno), spread_cap)

            planned_end_idx = idx + holding_td
            active_end_idx = min(planned_end_idx, int(last_idx_by_permno.get(permno, idx)))
            pos = ActivePosition(
                permno=permno,
                industry=industry,
                signal_group=str(row.get("signal_group", "regular")),
                prob=float(row["prob"]),
                signal_date=pd.Timestamp(row["date"]),
                entry_date=dt,
                entry_idx=idx,
                active_end_idx=active_end_idx,
                planned_active_end_idx=planned_end_idx,
                hit_label=int(row.get("y_div_10d", 0)),
                trade_id=f"{permno}_{pd.Timestamp(dt).date()}",
                entry_spread_rate=spread_rate,
            )
            active[permno] = pos
            entries_today.append(pos)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            entry_fixed_cost += weight * cost_rate
            entry_spread_cost += weight * spread_rate

        active_today = list(active.values())
        gross_ret = 0.0
        exit_positions = [p for p in active_today if p.active_end_idx == idx]
        for pos in exit_positions:
            exit_fixed_cost += weight * cost_rate
            spread_rate = 0.0
            if use_bid_ask_spread and day_frame is not None and pos.permno in day_frame.index:
                spread_rate = _quote_half_spread_rate(_select_row(day_frame, pos.permno), spread_cap)
            exit_spread_cost += weight * spread_rate
            exit_spread_by_trade[pos.trade_id] = spread_rate

        industry_exposure: Dict[str, float] = {}

        for pos in active_today:
            in_return_window = idx > pos.entry_idx
            if in_return_window and day_frame is not None and pos.permno in day_frame.index:
                row = _select_row(day_frame, pos.permno)
                stock_ret = float(row[ret_col]) if pd.notna(row[ret_col]) else 0.0
                pos.cum_mult *= 1.0 + stock_ret
            else:
                stock_ret = 0.0
            weighted_ret = weight * stock_ret
            gross_ret += weighted_ret
            industry_exposure[pos.industry] = industry_exposure.get(pos.industry, 0.0) + weight
            position_rows.append(
                {
                    "date": dt,
                    "permno": pos.permno,
                    "trade_id": pos.trade_id,
                    "weight": weight,
                    "stock_ret": stock_ret,
                    "weighted_ret": weighted_ret,
                    "in_return_window": int(in_return_window),
                    "entry_spread_rate": pos.entry_spread_rate,
                    "exit_spread_rate": exit_spread_by_trade.get(pos.trade_id, 0.0),
                    "prob": pos.prob,
                    "signal_group": pos.signal_group,
                    "industry": pos.industry,
                    "signal_date": pos.signal_date,
                    "entry_date": pos.entry_date,
                    "planned_exit_date": dates[pos.planned_active_end_idx] if pos.planned_active_end_idx < len(dates) else dates[-1],
                    "active_end_date": dates[pos.active_end_idx],
                }
            )

        entry_cost = entry_fixed_cost + entry_spread_cost
        exit_cost = exit_fixed_cost + exit_spread_cost
        daily_rows.append(
            {
                "date": dt,
                "portfolio_ret": gross_ret - entry_cost - exit_cost,
                "gross_ret": gross_ret,
                "entry_cost": entry_cost,
                "exit_cost": exit_cost,
                "entry_fixed_cost": entry_fixed_cost,
                "entry_spread_cost": entry_spread_cost,
                "exit_fixed_cost": exit_fixed_cost,
                "exit_spread_cost": exit_spread_cost,
                "turnover": (len(entries_today) + len(exit_positions)) * weight,
                "n_positions": len(active_today),
                "cash_weight": max(0.0, 1.0 - len(active_today) * weight),
                "signal_hit_rate": float(np.mean([p.hit_label for p in entries_today])) if entries_today else np.nan,
                "industry_exposure": industry_exposure,
            }
        )

        for pos in exit_positions:
            exit_spread_rate = exit_spread_by_trade.get(pos.trade_id, 0.0)
            trade_rows.append(
                {
                    "trade_id": pos.trade_id,
                    "permno": pos.permno,
                    "signal_date": pos.signal_date,
                    "entry_date": pos.entry_date,
                    "exit_date": dates[pos.active_end_idx],
                    "holding_days_realized": int(max(0, pos.active_end_idx - pos.entry_idx)),
                    "holding_days_planned": int(holding_td),
                    "truncated": int(pos.active_end_idx < pos.planned_active_end_idx),
                    "prob": pos.prob,
                    "signal_group": pos.signal_group,
                    "industry": pos.industry,
                    "entry_weight": weight,
                    "y_entry": pos.hit_label,
                    "entry_fixed_cost_rate": cost_rate,
                    "entry_spread_rate": pos.entry_spread_rate,
                    "exit_fixed_cost_rate": cost_rate,
                    "exit_spread_rate": exit_spread_rate,
                    "total_cost_rate": (2.0 * cost_rate) + pos.entry_spread_rate + exit_spread_rate,
                    "realized_holding_return": pos.cum_mult - 1.0 - (2.0 * cost_rate) - pos.entry_spread_rate - exit_spread_rate,
                }
            )
            cooldown_until[pos.permno] = pos.active_end_idx + int(cooldown_td)
            del active[pos.permno]

    daily_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(trade_rows).sort_values(["entry_date", "permno"]).reset_index(drop=True)
    positions_df = pd.DataFrame(position_rows).sort_values(["date", "permno"]).reset_index(drop=True)
    return daily_df, trades_df, positions_df


def _select_row(day_frame: pd.DataFrame, permno: int) -> pd.Series:
    row = day_frame.loc[permno]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _quote_half_spread_rate(row: pd.Series, spread_cap: float | None) -> float:
    bid = pd.to_numeric(pd.Series([row.get("DlyBid")]), errors="coerce").iloc[0]
    ask = pd.to_numeric(pd.Series([row.get("DlyAsk")]), errors="coerce").iloc[0]
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or ask < bid:
        return 0.0
    mid = (ask + bid) / 2.0
    if mid <= 0:
        return 0.0
    rate = (ask - bid) / (2.0 * mid)
    rate = max(float(rate), 0.0)
    if spread_cap is not None:
        rate = min(rate, spread_cap)
    return rate


def _industry_counts(active: Dict[int, ActivePosition]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pos in active.values():
        counts[pos.industry] = counts.get(pos.industry, 0) + 1
    return counts


def _plan_entries(candidates: pd.DataFrame, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, List[Dict]]:
    date_to_idx = {dt: i for i, dt in enumerate(dates)}
    planned: Dict[pd.Timestamp, List[Dict]] = {}
    for dt, g in candidates.groupby("date", sort=True):
        dt = pd.Timestamp(dt)
        idx = date_to_idx.get(dt)
        if idx is None or idx + 1 >= len(dates):
            continue
        planned.setdefault(dates[idx + 1], []).extend(g.sort_values("signal_rank").to_dict(orient="records"))
    return planned
