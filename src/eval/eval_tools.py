from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)

CensoringMode = Literal["count_as_false", "exclude"]


# -------------------------
# Canonical schema helpers
# -------------------------

def validate_eval_df(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure eval_df includes the required columns:
      - date: datetime
      - permno: int
      - y: {0,1}
      - prob: float in [0,1]
    """
    df = eval_df.copy()

    required = ["date", "permno", "y", "prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"eval_df 缺少必要列: {missing}. 需要至少包含 {required}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["permno"] = df["permno"].astype(int)

    df["y"] = df["y"].astype(int)
    if not set(df["y"].unique()).issubset({0, 1}):
        raise ValueError(f"y 必须是 0/1，但发现: {sorted(df['y'].unique())}")

    df["prob"] = df["prob"].astype(float)
    if (df["prob"] < -1e-6).any() or (df["prob"] > 1 + 1e-6).any():
        raise ValueError("prob 必须在 [0,1] 范围内")

    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    return df


# -------------------------
# Global & per-stock metrics
# -------------------------

def global_metrics(eval_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    df = validate_eval_df(eval_df)
    y = df["y"].values
    p = df["prob"].values

    out: Dict[str, float] = {}
    out["n"] = int(len(y))
    out["pos_rate"] = float(y.mean())

    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    out["aucpr"] = float(average_precision_score(y, p))
    out["logloss"] = float(log_loss(y, p))

    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    out.update({
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "alert_rate": float(pred.mean()),
        "fpr": float(fp / (fp + tn + 1e-12)),
    })
    return out


def per_stock_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = validate_eval_df(eval_df)

    def one_stock(g: pd.DataFrame) -> pd.Series:
        y = g["y"].values
        p = g["prob"].values
        n = len(g)
        n_pos = int(y.sum())

        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        aucpr = float(average_precision_score(y, p)) if n_pos > 0 else np.nan

        return pd.Series({
            "n": int(n),
            "n_pos": int(n_pos),
            "pos_rate": float(y.mean()),
            "auc": auc,
            "aucpr": aucpr,
        })

    out = df.groupby("permno", sort=False).apply(one_stock).reset_index()
    return out


def stock_aucpr_best_worst(
    eval_df: pd.DataFrame,
    stock_cohorts_cutoff: Optional[pd.DataFrame],
    stock_cohorts_full: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank stocks by AUC-PR and return the best/worst N with key cohort stats.

    We only rank stocks that have at least one positive label (n_pos > 0),
    because AUC-PR is not meaningful otherwise.
    """
    metrics = per_stock_metrics(eval_df)
    metrics = metrics[(metrics["n_pos"] > 0) & metrics["aucpr"].notna()].copy()
    if metrics.empty:
        return metrics

    feat_cols = ["gap_cv", "gap_med", "n_events"]
    base_permnos = pd.DataFrame({"permno": metrics["permno"].unique()})

    def _prep_feats(cohorts: Optional[pd.DataFrame], suffix: str) -> pd.DataFrame:
        if cohorts is None or cohorts.empty:
            return base_permnos.copy()

        keep = ["permno"] + [c for c in feat_cols if c in cohorts.columns]
        f = cohorts[keep].copy()

        rename_map = {c: f"{c}_{suffix}" for c in feat_cols if c in f.columns}
        f = f.rename(columns=rename_map)

        # Provide the requested div_count alias for each variant.
        n_events_col = f"n_events_{suffix}"
        div_count_col = f"div_count_{suffix}"
        if n_events_col in f.columns and div_count_col not in f.columns:
            f[div_count_col] = f[n_events_col]

        return f

    feats_cutoff = _prep_feats(stock_cohorts_cutoff, "cutoff")
    feats_full = _prep_feats(stock_cohorts_full, "full")

    merged = metrics.merge(feats_cutoff, on="permno", how="left")
    merged = merged.merge(feats_full, on="permno", how="left")

    n = int(max(top_n, 1))

    worst = (
        merged.sort_values(["aucpr", "n_pos", "n"], ascending=[True, False, False])
        .head(n)
        .reset_index(drop=True)
    )
    worst["rank_group"] = "worst"
    worst["rank_in_group"] = worst.index + 1

    best = (
        merged.sort_values(["aucpr", "n_pos", "n"], ascending=[False, False, False])
        .head(n)
        .reset_index(drop=True)
    )
    best["rank_group"] = "best"
    best["rank_in_group"] = best.index + 1

    cols_front = ["rank_group", "rank_in_group", "permno", "aucpr", "auc", "n", "n_pos", "pos_rate"]
    cols_feat = [
        c
        for c in [
            "gap_cv_cutoff",
            "gap_med_cutoff",
            "div_count_cutoff",
            "n_events_cutoff",
            "gap_cv_full",
            "gap_med_full",
            "div_count_full",
            "n_events_full",
        ]
        if c in merged.columns
    ]
    cols = cols_front + [c for c in cols_feat if c not in cols_front]

    out = pd.concat([best[cols], worst[cols]], ignore_index=True)
    # Keep "best" first, then "worst".
    out["rank_group"] = pd.Categorical(out["rank_group"], categories=["best", "worst"], ordered=True)
    out = out.sort_values(["rank_group", "rank_in_group"]).reset_index(drop=True)
    out["rank_group"] = out["rank_group"].astype(str)

    # Replace NaN with a readable token for cohort-style fields.
    unknown_cols = [
        "gap_cv_cutoff",
        "gap_med_cutoff",
        "div_count_cutoff",
        "n_events_cutoff",
        "gap_cv_full",
        "gap_med_full",
        "div_count_full",
        "n_events_full",
    ]
    for c in unknown_cols:
        if c in out.columns:
            out[c] = out[c].where(~out[c].isna(), "unknown")
    return out


# -------------------------
# Daily Top-K report + alerts
# -------------------------

def daily_topk_report(eval_df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    df = validate_eval_df(eval_df)
    daily = []

    for dt, g in df.groupby("date", sort=True):
        g = g.sort_values("prob", ascending=False)
        top = g.head(k)

        pos_total = int(g["y"].sum())
        pos_rate = float(g["y"].mean())

        tp_at_k = int(top["y"].sum())
        prec_at_k = float(tp_at_k / max(k, 1))
        rec_at_k = float(tp_at_k / (pos_total + 1e-12))
        lift = float(prec_at_k / (pos_rate + 1e-12))

        daily.append({
            "date": pd.to_datetime(dt),
            "k": int(k),
            "n": int(len(g)),
            "pos_total": int(pos_total),
            "pos_rate": float(pos_rate),
            "tp_at_k": int(tp_at_k),
            "precision_at_k": prec_at_k,
            "recall_at_k": rec_at_k,
            "lift_at_k": lift,
        })

    return pd.DataFrame(daily).sort_values("date").reset_index(drop=True)


def make_daily_topk_alerts(eval_df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """
    Turn "daily top-k" into an alerts_df with schema:
      permno, date, prob

    This mimics an operational policy: "each day, we alert on the top-k tickers by score".
    """
    df = validate_eval_df(eval_df)
    alerts = []
    for dt, g in df.groupby("date", sort=True):
        top = g.sort_values("prob", ascending=False).head(k)
        for _, r in top.iterrows():
            alerts.append({"permno": int(r["permno"]), "date": pd.to_datetime(r["date"]), "prob": float(r["prob"])})
    return pd.DataFrame(alerts).sort_values(["permno", "date"]).reset_index(drop=True)


# -------------------------
# Event filtering and event-level evaluation
# -------------------------

def filter_events_for_eval(eval_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events to the eligible set for event-level evaluation:
      1) event_date within eval_df [min_date, max_date]
      2) permno in eval_df
      3) for each permno, there exists at least one trading day t < event_date in eval_df (opportunity to alert)
    """
    df = validate_eval_df(eval_df)

    ev = events_df.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev["permno"] = ev["permno"].astype(int)
    ev = ev.dropna(subset=["permno", "event_date"]).drop_duplicates()

    min_d = df["date"].min()
    max_d = df["date"].max()

    ev = ev[(ev["event_date"] >= min_d) & (ev["event_date"] <= max_d)].copy()

    perm_set = set(df["permno"].unique())
    ev = ev[ev["permno"].isin(perm_set)].copy()

    first_date = df.groupby("permno")["date"].min()
    ev = ev[ev.apply(lambda r: first_date.loc[r["permno"]] < r["event_date"], axis=1)].copy()

    return ev.sort_values(["permno", "event_date"]).reset_index(drop=True)


def generate_alerts_threshold(eval_df: pd.DataFrame, threshold: float, cooldown_td: int = 0) -> pd.DataFrame:
    """
    For each stock, raise an alert when prob >= threshold.
    If cooldown_td>0, suppress alerts for the next cooldown_td trading days after an alert.
    Output columns:
      permno, date, prob, aidx (position index in that stock's date series)
    """
    df = validate_eval_df(eval_df)
    rows = []

    for permno, g in df.groupby("permno", sort=False):
        probs = g["prob"].values
        dates = g["date"].values
        last_alert_i = -10**18

        for i in range(len(g)):
            if probs[i] >= threshold:
                if cooldown_td > 0 and i <= last_alert_i + cooldown_td:
                    continue
                rows.append({"permno": int(permno), "date": pd.to_datetime(dates[i]), "prob": float(probs[i]), "aidx": int(i)})
                last_alert_i = i

    return pd.DataFrame(rows).sort_values(["permno", "date"]).reset_index(drop=True)


def _attach_aidx(eval_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach aidx (the within-permno row index) to alerts_df by matching on (permno, date).
    Assumes eval_df is unique on (permno, date).
    """
    df = validate_eval_df(eval_df).copy()
    df["_aidx"] = df.groupby("permno").cumcount().astype(int)

    a = alerts_df.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a["permno"] = a["permno"].astype(int)

    merged = a.merge(df[["permno", "date", "_aidx"]], on=["permno", "date"], how="left")
    merged = merged.rename(columns={"_aidx": "aidx"})
    return merged


def evaluate_alerts_forward_window(
    eval_df: pd.DataFrame,
    events_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    H: int,
    censoring_mode: CensoringMode = "exclude",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Core event-level evaluation consistent with the labeling convention:
      - alert at date t defines a window (t, t_end], where t_end is the H-th future trading day for that stock
      - the alert is 'true' if any event_date falls into that window

    Right censoring:
      - if an alert is too close to the end (no t_end), it is *censored*
      - censoring_mode="exclude": do not count censored alerts as false alerts (recommended)
      - censoring_mode="count_as_false": treat censored alerts as false (not recommended; inflates false alerts near tail)
    """
    df = validate_eval_df(eval_df)
    ev = events_df.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev["permno"] = ev["permno"].astype(int)
    ev = ev.dropna(subset=["permno", "event_date"]).drop_duplicates()
    ev = ev.sort_values(["permno", "event_date"]).reset_index(drop=True)

    alerts = alerts_df.copy()
    if "aidx" not in alerts.columns:
        alerts = _attach_aidx(df, alerts)

    alerts = alerts.sort_values(["permno", "date"]).reset_index(drop=True)

    # trading-day sequences per stock
    dates_by_perm = {p: g["date"].values.astype("datetime64[ns]") for p, g in df.groupby("permno", sort=False)}
    events_by_perm = {p: g["event_date"].values.astype("datetime64[ns]") for p, g in ev.groupby("permno", sort=False)}

    is_true = []
    matched_event_date = []
    lead_times = []
    is_censored = []

    best_lead_for_event: Dict[Tuple[int, pd.Timestamp], int] = {}

    for permno, g in alerts.groupby("permno", sort=False):
        p = int(permno)
        stock_dates = dates_by_perm.get(p)
        stock_events = events_by_perm.get(p)

        # If no event history in eval range, alerts are false (or censored)
        for t, aidx, prob in zip(g["date"].values.astype("datetime64[ns]"), g["aidx"].values, g["prob"].values):
            if stock_dates is None or pd.isna(aidx):
                is_true.append(0)
                matched_event_date.append(pd.NaT)
                lead_times.append(np.nan)
                is_censored.append(1)
                continue

            aidx = int(aidx)
            if aidx + H >= len(stock_dates):
                # right-censored: no well-defined H trading-day window
                is_true.append(0)
                matched_event_date.append(pd.NaT)
                lead_times.append(np.nan)
                is_censored.append(1)
                continue

            is_censored.append(0)

            if stock_events is None or len(stock_events) == 0:
                is_true.append(0)
                matched_event_date.append(pd.NaT)
                lead_times.append(np.nan)
                continue

            t_end = stock_dates[aidx + H]
            left = np.searchsorted(stock_events, t, side="right")
            right = np.searchsorted(stock_events, t_end, side="right") - 1

            if left <= right:
                e_date = stock_events[left]
                is_true.append(1)
                matched_event_date.append(pd.to_datetime(e_date))

                eidx = np.searchsorted(stock_dates, e_date, side="left")
                lead = int(eidx - aidx)
                lead = max(1, min(H, lead))
                lead_times.append(float(lead))

                key = (p, pd.to_datetime(e_date))
                prev = best_lead_for_event.get(key)
                if prev is None or lead > prev:
                    best_lead_for_event[key] = lead
            else:
                is_true.append(0)
                matched_event_date.append(pd.NaT)
                lead_times.append(np.nan)

    alerts_out = alerts.copy()
    alerts_out["is_true_alert"] = np.array(is_true, dtype=int)
    alerts_out["matched_event_date"] = matched_event_date
    alerts_out["lead_time_td"] = lead_times
    alerts_out["is_censored"] = np.array(is_censored, dtype=int)

    # event-level output
    event_rows = []
    for permno, g in ev.groupby("permno", sort=False):
        p = int(permno)
        for e_date in g["event_date"].values:
            key = (p, pd.to_datetime(e_date))
            best = best_lead_for_event.get(key)
            event_rows.append({
                "permno": p,
                "event_date": pd.to_datetime(e_date),
                "hit": 0 if best is None else 1,
                "best_lead_time_td": np.nan if best is None else float(best),
            })
    events_out = pd.DataFrame(event_rows).sort_values(["permno", "event_date"]).reset_index(drop=True)

    # summary counts (censoring-aware)
    n_events = int(len(events_out))
    hit_events = int(events_out["hit"].sum())
    hit_rate = float(hit_events / (n_events + 1e-12))

    n_alerts_total = int(len(alerts_out))
    n_censored = int(alerts_out["is_censored"].sum())

    if censoring_mode == "exclude":
        alerts_eval = alerts_out[alerts_out["is_censored"] == 0].copy()
    else:
        alerts_eval = alerts_out

    n_alerts = int(len(alerts_eval))
    false_alerts = int((alerts_eval["is_true_alert"] == 0).sum())
    false_alert_rate = float(false_alerts / (n_alerts + 1e-12)) if n_alerts > 0 else np.nan

    stock_days = int(len(df))
    false_per_1000 = float(false_alerts / (stock_days / 1000.0 + 1e-12))

    lead_desc = events_out.loc[events_out["hit"] == 1, "best_lead_time_td"].describe()

    summary = {
        "H": int(H),
        "censoring_mode": censoring_mode,
        "n_events": n_events,
        "hit_events": hit_events,
        "hit_rate_events": hit_rate,
        "n_alerts_total": n_alerts_total,
        "n_alerts_eval": n_alerts,
        "n_censored_alerts": n_censored,
        "false_alerts": false_alerts,
        "false_alert_rate": false_alert_rate,
        "false_alerts_per_1000_stockdays": false_per_1000,
        "lead_time_td_desc_on_hits": lead_desc.to_dict() if hit_events > 0 else {},
    }
    return events_out, alerts_out.sort_values(["permno", "date"]).reset_index(drop=True), summary


def event_level_report_v2(
    eval_df: pd.DataFrame,
    events_df: pd.DataFrame,
    H: int = 10,
    threshold: float = 0.5,
    cooldown_td: int = 0,
    censoring_mode: CensoringMode = "exclude",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience wrapper: generate threshold-based alerts (with cooldown),
    then run evaluate_alerts_forward_window with censoring correction.
    """
    df = validate_eval_df(eval_df)
    alerts = generate_alerts_threshold(df, threshold=threshold, cooldown_td=cooldown_td)
    alerts = alerts.drop(columns=[], errors="ignore")
    return evaluate_alerts_forward_window(df, events_df, alerts, H=H, censoring_mode=censoring_mode)


# -------------------------
# Cadence stats & cohorts
# -------------------------

def compute_cadence_stats(
    events_df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-stock cadence stats using historical events (optionally only before cutoff_date).

    Output columns:
      permno, n_events, gap_mean, gap_med, gap_std, gap_cv
    """
    ev = events_df.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev["permno"] = ev["permno"].astype(int)
    ev = ev.dropna(subset=["permno", "event_date"]).drop_duplicates()
    ev = ev.sort_values(["permno", "event_date"]).reset_index(drop=True)

    if cutoff_date is not None:
        ev = ev[ev["event_date"] < pd.to_datetime(cutoff_date)].copy()

    def _one(g: pd.DataFrame) -> pd.Series:
        dates = g["event_date"].sort_values()
        gaps = dates.diff().dt.days.dropna()
        n = int(len(dates))
        if len(gaps) == 0:
            return pd.Series({"n_events": n, "gap_mean": np.nan, "gap_med": np.nan, "gap_std": np.nan, "gap_cv": np.nan})
        gap_mean = float(gaps.mean())
        gap_med = float(gaps.median())
        gap_std = float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
        gap_cv = float(gap_std / (gap_mean + 1e-12))
        return pd.Series({"n_events": n, "gap_mean": gap_mean, "gap_med": gap_med, "gap_std": gap_std, "gap_cv": gap_cv})

    out = ev.groupby("permno", sort=False).apply(_one).reset_index()
    return out


def build_stock_cohorts(
    eval_df: pd.DataFrame,
    cadence_stats: pd.DataFrame,
    log_mkt_cap_col: str = "log_mkt_cap",
    turnover_col: str = "turnover_5d",
    sic_col: str = "SICCD",
    vol_col: str = "vol_21d",
) -> pd.DataFrame:
    """
    Build per-stock cohort tags for slicing event-level performance.

    Cohort dimensions (as requested):
      - n_events buckets: 0, 1-5, 6-20, 20+
      - gap_cv buckets: low/med/high (tertiles; NaN -> "unknown")
      - size, liquidity: quantiles (by per-stock median log_mkt_cap / turnover_5d)
      - industry: SIC two-digit if available; else use existing 'industry' column
      - vol regime: high/low by median vol_21d
      - clockwork quarterly: gap_med≈91 and gap_std small
    """
    df = validate_eval_df(eval_df)

    # per-stock medians from eval_df
    per_stock = df.groupby("permno").agg(
        mkt_cap_med=(log_mkt_cap_col, "median") if log_mkt_cap_col in df.columns else ("prob", "median"),
        turnover_med=(turnover_col, "median") if turnover_col in df.columns else ("prob", "median"),
        vol_med=(vol_col, "median") if vol_col in df.columns else ("prob", "median"),
    ).reset_index()

    # bring SIC/industry
    if sic_col in df.columns:
        sic2 = (
            df.assign(_sic2=df[sic_col].fillna(0).astype(int).astype(str).str.zfill(4).str[:2])
              .groupby("permno")["_sic2"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "00")
              .reset_index()
              .rename(columns={"_sic2": "industry"})
        )
    elif "industry" in df.columns:
        sic2 = df.groupby("permno")["industry"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "00").reset_index()
    else:
        sic2 = per_stock[["permno"]].assign(industry="00")

    cohorts = per_stock.merge(sic2, on="permno", how="left")
    cohorts = cohorts.merge(cadence_stats, on="permno", how="left")

    # n_events buckets
    def _bucket_n(n):
        if pd.isna(n) or n == 0:
            return "0"
        n = int(n)
        if 1 <= n <= 5:
            return "1-5"
        if 6 <= n <= 20:
            return "6-20"
        return "20+"
    cohorts["bucket_n_events"] = cohorts["n_events"].apply(_bucket_n)

    # gap_cv buckets (tertiles over non-null)
    cv = cohorts["gap_cv"]
    nonnull = cv.dropna()
    if len(nonnull) >= 3:
        q1, q2 = nonnull.quantile([1/3, 2/3]).values
        def _bucket_cv(x):
            if pd.isna(x):
                return "unknown"
            if x <= q1:
                return "low"
            if x <= q2:
                return "mid"
            return "high"
    else:
        def _bucket_cv(x):
            return "unknown" if pd.isna(x) else "mid"
    cohorts["bucket_gap_cv"] = cv.apply(_bucket_cv)

    # size/liquidity quantiles (by per-stock median)
    def _qbucket(series: pd.Series, name: str) -> pd.Series:
        s = series.copy()
        qs = s.dropna()
        if len(qs) < 4:
            return pd.Series(["unknown"] * len(s), index=s.index)
        q = qs.quantile([0.25, 0.5, 0.75]).values
        def _b(x):
            if pd.isna(x):
                return "unknown"
            if x <= q[0]:
                return "q1"
            if x <= q[1]:
                return "q2"
            if x <= q[2]:
                return "q3"
            return "q4"
        return s.apply(_b)

    cohorts["bucket_size"] = _qbucket(cohorts["mkt_cap_med"], "size")
    cohorts["bucket_liquidity"] = _qbucket(cohorts["turnover_med"], "liq")

    # vol regime: high/low by median across stocks
    vol_med_all = cohorts["vol_med"].median(skipna=True)
    cohorts["bucket_vol_regime"] = cohorts["vol_med"].apply(lambda x: "high" if (not pd.isna(x) and x >= vol_med_all) else "low")

    # clockwork quarterly stock heuristic
    cohorts["is_quarterly_clockwork"] = (
        (cohorts["gap_med"].between(85, 97)) &
        (cohorts["gap_std"].fillna(999).le(10)) &
        (cohorts["n_events"].fillna(0).ge(6))
    ).astype(int)

    return cohorts


def cohort_event_metrics(
    eval_df: pd.DataFrame,
    events_out: pd.DataFrame,
    alerts_out: pd.DataFrame,
    cohorts: pd.DataFrame,
    cohort_cols: List[str],
    censoring_mode: CensoringMode = "exclude",
) -> pd.DataFrame:
    """
    Slice event-level performance by cohort attributes.
    Returns a table with per-bucket:
      - n_events, hit_rate_events
      - n_alerts_eval, false_alert_rate, false_alerts_per_1000_stockdays
      - lead_time mean/median among hits
    """
    df = validate_eval_df(eval_df)
    c = cohorts[["permno"] + cohort_cols].copy()

    e = events_out.merge(c, on="permno", how="left")
    a = alerts_out.merge(c, on="permno", how="left")
    # apply censoring if needed
    if censoring_mode == "exclude" and "is_censored" in a.columns:
        a_eval = a[a["is_censored"] == 0].copy()
    else:
        a_eval = a.copy()

    # stock-days per cohort: sum of rows in eval_df for stocks in cohort bucket
    df2 = df.merge(c, on="permno", how="left")

    rows = []
    for col in cohort_cols:
        for key, g in df2.groupby(col, dropna=False):
            permnos = set(g["permno"].unique())
            stock_days = int(len(g))

            e_sub = e[e["permno"].isin(permnos)]
            a_sub = a_eval[a_eval["permno"].isin(permnos)]

            n_events = int(len(e_sub))
            hit_events = int(e_sub["hit"].sum()) if "hit" in e_sub.columns else 0
            hit_rate = float(hit_events / (n_events + 1e-12)) if n_events > 0 else np.nan

            n_alerts = int(len(a_sub))
            false_alerts = int((a_sub["is_true_alert"] == 0).sum()) if "is_true_alert" in a_sub.columns else 0
            false_rate = float(false_alerts / (n_alerts + 1e-12)) if n_alerts > 0 else np.nan
            false_per_1000 = float(false_alerts / (stock_days / 1000.0 + 1e-12))

            lead = e_sub.loc[e_sub.get("hit", 0) == 1, "best_lead_time_td"] if "best_lead_time_td" in e_sub.columns else pd.Series(dtype=float)

            rows.append({
                "cohort_dim": col,
                "bucket": key,
                "n_stocks": int(len(permnos)),
                "stock_days": stock_days,
                "n_events": n_events,
                "hit_events": hit_events,
                "hit_rate_events": hit_rate,
                "n_alerts": n_alerts,
                "false_alerts": false_alerts,
                "false_alert_rate": false_rate,
                "false_alerts_per_1000_stockdays": false_per_1000,
                "lead_time_mean": float(lead.mean()) if len(lead) else np.nan,
                "lead_time_median": float(lead.median()) if len(lead) else np.nan,
            })

    return pd.DataFrame(rows).sort_values(["cohort_dim", "bucket"]).reset_index(drop=True)


# -------------------------
# Right-censoring diagnostics
# -------------------------

def censoring_diagnostics(alerts_out: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnose whether alerts are concentrated near the tail and being censored.

    Returns per-month counts:
      - n_alerts
      - n_censored
      - censored_rate
      - n_false (among non-censored if possible)
    """
    a = alerts_out.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a["month"] = a["date"].dt.to_period("M").astype(str)

    if "is_censored" not in a.columns:
        a["is_censored"] = 0

    # non-censored false rate if possible
    a_nc = a[a["is_censored"] == 0].copy()
    false_by_m = a_nc.groupby("month")["is_true_alert"].apply(lambda s: int((s == 0).sum()) if "is_true_alert" in a_nc.columns else 0)

    out = (
        a.groupby("month").agg(
            n_alerts=("permno", "size"),
            n_censored=("is_censored", "sum"),
        ).reset_index()
    )
    out["censored_rate"] = out["n_censored"] / (out["n_alerts"] + 1e-12)
    out = out.merge(false_by_m.rename("n_false_non_censored"), on="month", how="left")
    return out.sort_values("month").reset_index(drop=True)


# -------------------------
# Operational daily simulation
# -------------------------

def simulate_daily_ops(
    eval_df: pd.DataFrame,
    alerts_out: pd.DataFrame,
    events_out: pd.DataFrame,
) -> pd.DataFrame:
    """
    Daily operational summary (for "day-by-day" monitoring):
      - how many alerts fire
      - how many are true/false
      - how many events occur that day; how many are 'hit' in the model sense

    This helps answer: "how noisy is the system per day?"
    """
    df = validate_eval_df(eval_df)
    a = alerts_out.copy()
    a["date"] = pd.to_datetime(a["date"])
    e = events_out.copy()
    e["event_date"] = pd.to_datetime(e["event_date"])

    daily_alerts = a.groupby("date").agg(
        n_alerts=("permno", "size"),
        n_true=("is_true_alert", "sum") if "is_true_alert" in a.columns else ("permno", "size"),
        n_censored=("is_censored", "sum") if "is_censored" in a.columns else ("permno", "size"),
    ).reset_index()

    if "is_true_alert" not in a.columns:
        daily_alerts["n_true"] = np.nan
    if "is_censored" not in a.columns:
        daily_alerts["n_censored"] = 0

    daily_alerts["n_false"] = daily_alerts["n_alerts"] - daily_alerts["n_true"]

    daily_events = e.groupby("event_date").agg(
        n_events=("permno", "size"),
        n_hit_events=("hit", "sum") if "hit" in e.columns else ("permno", "size"),
    ).reset_index().rename(columns={"event_date": "date"})

    out = pd.merge(daily_alerts, daily_events, on="date", how="outer").fillna(0)
    out = out.sort_values("date").reset_index(drop=True)
    return out


# -------------------------
# Phase / lead-time analysis
# -------------------------

def phase_table(events_out: pd.DataFrame, H: int) -> pd.DataFrame:
    """
    Summarize lead-time distribution and "phase bias".
      - if lead times concentrate near H => the model tends to alert *very early*
      - if lead times concentrate near 1-2 => the model tends to alert *late*
    """
    e = events_out.copy()
    if "best_lead_time_td" not in e.columns:
        return pd.DataFrame()

    e_hit = e[e["hit"] == 1].copy()
    if len(e_hit) == 0:
        return pd.DataFrame()

    e_hit["phase_bucket"] = pd.cut(
        e_hit["best_lead_time_td"],
        bins=[0, 2, 5, 10],
        labels=["late(1-2)", "mid(3-5)", "early(6-10)"],#, f"very_early(11-{H})"],
        include_lowest=True,
        right=True
    )

    tab = e_hit.groupby("phase_bucket").agg(
        n_events=("permno", "size"),
        lead_mean=("best_lead_time_td", "mean"),
        lead_median=("best_lead_time_td", "median"),
    ).reset_index()

    return tab



def event_recall_by_event_date(events_out: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event-date recall table:
      event_recall = hit_events / n_events on each event_date

    This is usually what you want for "每日 EventRecall@K" style reporting.
    """
    e = events_out.copy()
    e["event_date"] = pd.to_datetime(e["event_date"], errors="coerce")
    e = e.dropna(subset=["event_date"])
    out = e.groupby("event_date").agg(
        n_events=("permno", "size"),
        hit_events=("hit", "sum"),
    ).reset_index()
    out["event_recall"] = out["hit_events"] / (out["n_events"] + 1e-12)
    return out.sort_values("event_date").reset_index(drop=True)
