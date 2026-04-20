"""
Deep SHAP comparison (50r vs 200r) + tradability test on selected names.

Goal:
    Test the hypothesis that 200r's extra rounds learn microstructure/liquidity
    alpha that predicts y=1 but is not tradable (worse spreads, higher vol,
    smaller caps, worse realized fwd returns) — i.e. 200r overfits to the
    classification target in a way that hurts backtest Sharpe.

Outputs under outputs/analysis/calibration_shap_50v200/:
    shap_scatter.png              full per-feature importance scatter, 50r vs 200r
    shap_group_share.csv          importance share by feature group
    shap_group_share.png          bar chart of the above
    shap_topfeat_dist.png         SHAP value distribution for top-3 shared features
    shap_concentration.csv        top-k cumulative share of |SHAP|
    tradability_topk.csv          mean cohort stats of top-K daily picks
    tradability_topk_disjoint.csv same, restricted to names where exactly one model picks
    tradability_fwd_ret.csv       realized fwd_ret_10d (and win rate) for top-K picks
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[2]))

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.modeling.preprocess import apply_imputer_and_scaler
from src.modeling.train import build_dmatrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("outputs/analysis/calibration_shap_50v200")
OUT.mkdir(parents=True, exist_ok=True)

SHAP_SAMPLE = 80_000
RNG = np.random.default_rng(42)

# Feature grouping (stable, hand-curated from the current feature list)
GROUPS = {
    "div_cycle": {
        "days_since_last_div", "time_to_mean_rN", "time_to_med_exp",
        "z_to_mean_rN", "z_to_med_exp", "decl_to_exdt_days",
        "decl_to_exdt_mean_exp", "gap_std_rN", "gap_mean_rN",
        "gap_std_exp", "gap_mean_exp", "div_no_cut_streak",
        "div_amt_chg", "div_amt_mean_exp", "has_div_history",
    },
    "microstructure": {
        "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd",
        "num_trd_spike", "daily_range_pct", "turnover_5d",
    },
    "vol_risk": {
        "vol_5d", "vol_21d", "atr_5d", "atr_21d",
    },
    "return_momentum": {
        "ret_5d", "ret_10d", "ret_21d", "ret_rel_to_ind",
    },
    "fundamental": {
        "log_mkt_cap", "log_price", "dividend_yield_proxy",
    },
    "seasonality": {
        "doy_sin", "doy_cos", "month_sin", "month_cos",
        "weekday", "weekday_sin", "weekday_cos",
    },
    "industry": {"industry", "SICCD"},
}


def _group_of(feat: str) -> str:
    for g, s in GROUPS.items():
        if feat in s:
            return g
    return "other"


# ─────────── preprocessing helper ───────────
def _prep_for_artifact(df: pd.DataFrame, art) -> pd.DataFrame:
    x = df.copy()
    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)
    for c in art.cat_cols:
        if c not in x.columns:
            x[c] = "<<MISSING>>"
        x[c] = x[c].astype("string").fillna("<<MISSING>>")
        cats = art.cat_categories.get(c)
        if cats:
            cats_idx = pd.Index(cats)
            unk = "<<UNK>>"
            if unk not in cats_idx:
                cats_idx = cats_idx.append(pd.Index([unk]))
            x[c] = pd.Categorical(x[c].where(x[c].isin(cats_idx), unk), categories=cats_idx)
        else:
            x[c] = x[c].astype("category")
    feat_names = art.feature_names or (art.num_cols + art.cat_cols)
    for c in feat_names:
        if c not in x.columns:
            x[c] = np.nan
    return x[feat_names]


def _contribs(df: pd.DataFrame, art) -> tuple[np.ndarray, list]:
    X = _prep_for_artifact(df, art)
    feat_names = art.feature_names or (art.num_cols + art.cat_cols)
    dm = build_dmatrix(X, y=None, enable_categorical=True, feature_names=feat_names)
    c = art.booster.predict(dm, pred_contribs=True)
    return c[:, :-1], feat_names  # drop bias


# ─────────── SHAP deep dive ───────────
def shap_deep(sample: pd.DataFrame) -> dict:
    contribs = {}
    feats = None
    for tag in ["dual_50r", "dual_200r"]:
        art = joblib.load(f"models/xgb_{tag}.joblib")
        c, f = _contribs(sample, art)
        contribs[tag] = c
        feats = f

    c50 = contribs["dual_50r"]
    c200 = contribs["dual_200r"]

    # ── (1) Full scatter: per-feature mean|SHAP|
    abs50 = np.abs(c50).mean(axis=0)
    abs200 = np.abs(c200).mean(axis=0)
    scatter_df = pd.DataFrame({"feature": feats, "abs_50r": abs50, "abs_200r": abs200})
    scatter_df["group"] = scatter_df["feature"].apply(_group_of)

    fig, ax = plt.subplots(figsize=(8, 8))
    groups = scatter_df["group"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    for g, col in zip(groups, colors):
        m = scatter_df["group"] == g
        ax.scatter(scatter_df.loc[m, "abs_50r"], scatter_df.loc[m, "abs_200r"],
                   label=g, s=40, alpha=0.8, color=col)
    lim = max(abs50.max(), abs200.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4)
    ax.plot([0, lim], [0, 2 * lim], "k:", lw=0.8, alpha=0.3, label="2×")
    # Annotate features where |diff| is large
    scatter_df["diff"] = scatter_df["abs_200r"] - scatter_df["abs_50r"]
    for _, r in scatter_df.nlargest(8, "diff").iterrows():
        ax.annotate(r["feature"], (r["abs_50r"], r["abs_200r"]),
                    fontsize=8, alpha=0.8)
    for _, r in scatter_df.nsmallest(4, "diff").iterrows():
        ax.annotate(r["feature"], (r["abs_50r"], r["abs_200r"]),
                    fontsize=8, alpha=0.8, color="darkred")
    ax.set_xlabel("mean |SHAP| — 50r")
    ax.set_ylabel("mean |SHAP| — 200r")
    ax.set_title("Per-feature SHAP importance: 50r vs 200r")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(OUT / "shap_scatter.png", dpi=120)
    plt.close(fig)

    # ── (2) Group-level share
    scatter_df["share_50r"] = scatter_df["abs_50r"] / scatter_df["abs_50r"].sum()
    scatter_df["share_200r"] = scatter_df["abs_200r"] / scatter_df["abs_200r"].sum()
    group_share = scatter_df.groupby("group")[["abs_50r", "abs_200r", "share_50r", "share_200r"]].sum()
    group_share["share_delta"] = group_share["share_200r"] - group_share["share_50r"]
    group_share = group_share.sort_values("share_200r", ascending=False)
    group_share.to_csv(OUT / "shap_group_share.csv")
    print("\n=== importance share by group ===")
    print(group_share.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(group_share))
    w = 0.38
    ax.bar(x - w/2, group_share["share_50r"], w, label="50r", color="tab:blue")
    ax.bar(x + w/2, group_share["share_200r"], w, label="200r", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(group_share.index, rotation=20, ha="right")
    ax.set_ylabel("share of total mean|SHAP|")
    ax.set_title("SHAP importance share by feature group")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "shap_group_share.png", dpi=120)
    plt.close(fig)

    # ── (3) Concentration: cumulative share of top-K features
    conc_rows = []
    for tag, abs_arr in [("50r", abs50), ("200r", abs200)]:
        s = np.sort(abs_arr)[::-1]
        tot = s.sum()
        for k in [1, 3, 5, 10, 20]:
            conc_rows.append(dict(model=tag, top_k=k, cum_share=float(s[:k].sum() / tot)))
    conc = pd.DataFrame(conc_rows)
    conc.to_csv(OUT / "shap_concentration.csv", index=False)
    print("\n=== SHAP concentration ===")
    print(conc.pivot(index="top_k", columns="model", values="cum_share").to_string())

    # ── (4) Per-row SHAP distribution for top-3 shared features
    top3 = scatter_df.nlargest(3, "abs_200r")["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat in zip(axes, top3):
        i = feats.index(feat)
        ax.hist(c50[:, i], bins=60, alpha=0.55, label="50r", color="tab:blue", density=True)
        ax.hist(c200[:, i], bins=60, alpha=0.55, label="200r", color="tab:red", density=True)
        ax.set_title(feat)
        ax.set_xlabel("SHAP (log-odds contribution)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Per-row SHAP distribution — top-3 shared features")
    fig.tight_layout()
    fig.savefig(OUT / "shap_topfeat_dist.png", dpi=120)
    plt.close(fig)

    # ── (5) Feature dispersion: how often is a feature "active" (|shap|>eps)
    eps = 0.01
    active = pd.DataFrame({
        "feature": feats,
        "active_frac_50r": (np.abs(c50) > eps).mean(axis=0),
        "active_frac_200r": (np.abs(c200) > eps).mean(axis=0),
    })
    active["delta"] = active["active_frac_200r"] - active["active_frac_50r"]
    active = active.sort_values("delta", ascending=False)
    active.to_csv(OUT / "shap_active_frac.csv", index=False)

    return {"scatter_df": scatter_df, "c50": c50, "c200": c200, "feats": feats}


# ─────────── tradability test ───────────
def tradability_test():
    test_full = pd.read_parquet("data/processed/test.parquet")
    # We'll join cohort/trade cols into preds by (date, permno)
    join_cols = ["DlyCalDt", "PERMNO", "bid_ask_spread_5d", "bid_ask_spread",
                 "vol_21d", "atr_21d", "log_mkt_cap", "turnover_5d",
                 "daily_range_pct", "fwd_ret_10d"]
    t = test_full[join_cols].copy()
    t["DlyCalDt"] = pd.to_datetime(t["DlyCalDt"])
    t = t.rename(columns={"DlyCalDt": "date", "PERMNO": "permno"})
    t["permno"] = t["permno"].astype(int)

    preds = {}
    for tag in ["dual_50r", "dual_200r"]:
        p = pd.read_parquet(f"outputs/runs/{tag}/preds/test_preds.parquet")
        p["date"] = pd.to_datetime(p["date"])
        p["permno"] = p["permno"].astype(int)
        # drop cohort cols already in preds so the join columns win
        drop = [c for c in t.columns if c in p.columns and c not in ("date", "permno")]
        p = p.drop(columns=drop)
        preds[tag] = p.merge(t, on=["date", "permno"], how="left")

    # Top-K daily picks by score_cls
    def _topk(df, k):
        return (
            df.sort_values("score_cls", ascending=False)
              .groupby("date", sort=False)
              .head(k)
        )

    cohort_cols = ["bid_ask_spread_5d", "vol_21d", "atr_21d", "log_mkt_cap",
                   "turnover_5d", "daily_range_pct"]
    rows = []
    for k in [50, 100, 200]:
        for tag in ["dual_50r", "dual_200r"]:
            picks = _topk(preds[tag], k)
            row = {"model": tag, "k": k, "n_picks": len(picks)}
            for c in cohort_cols:
                row[f"{c}_mean"] = float(picks[c].mean())
                row[f"{c}_median"] = float(picks[c].median())
            row["fwd_ret_mean"] = float(picks["fwd_ret_10d"].mean())
            row["fwd_ret_median"] = float(picks["fwd_ret_10d"].median())
            row["y_mean"] = float(picks["y"].mean())
            row["fwd_ret_win_rate"] = float((picks["fwd_ret_10d"] > 0).mean())
            rows.append(row)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "tradability_topk.csv", index=False)
    print("\n=== top-K cohort stats (both models) ===")
    print(tr.to_string(index=False))

    # Disjoint subset: names that ONLY 50r picks vs ONLY 200r picks (at k=100)
    for k in [50, 100]:
        a = _topk(preds["dual_50r"], k)[["date", "permno"]].assign(m50=1)
        b = _topk(preds["dual_200r"], k)[["date", "permno"]].assign(m200=1)
        m = a.merge(b, on=["date", "permno"], how="outer").fillna(0)
        only50 = m[(m["m50"] == 1) & (m["m200"] == 0)][["date", "permno"]]
        only200 = m[(m["m50"] == 0) & (m["m200"] == 1)][["date", "permno"]]
        both = m[(m["m50"] == 1) & (m["m200"] == 1)][["date", "permno"]]

        def _stats(df_pick, tag):
            joined = df_pick.merge(t, on=["date", "permno"], how="left")
            # Get y from either preds table (same for both)
            joined = joined.merge(
                preds["dual_50r"][["date", "permno", "y"]], on=["date", "permno"], how="left"
            )
            out = {"group": tag, "k": k, "n": len(joined)}
            for c in cohort_cols:
                out[f"{c}_mean"] = float(joined[c].mean())
            out["fwd_ret_mean"] = float(joined["fwd_ret_10d"].mean())
            out["fwd_ret_win_rate"] = float((joined["fwd_ret_10d"] > 0).mean())
            out["y_mean"] = float(joined["y"].mean())
            return out

        drows = [_stats(both, "shared"),
                 _stats(only50, "only_50r"),
                 _stats(only200, "only_200r")]
        dj = pd.DataFrame(drows)
        fn = OUT / f"tradability_topk_disjoint_k{k}.csv"
        dj.to_csv(fn, index=False)
        print(f"\n=== disjoint top-{k} groups (shared / only-50r / only-200r) ===")
        print(dj.to_string(index=False))

    # Separate fwd-ret summary table
    fr_rows = []
    for k in [50, 100, 200]:
        for tag in ["dual_50r", "dual_200r"]:
            picks = _topk(preds[tag], k)
            fr_rows.append(dict(
                model=tag, k=k,
                fwd_ret_mean=float(picks["fwd_ret_10d"].mean()),
                fwd_ret_median=float(picks["fwd_ret_10d"].median()),
                fwd_ret_std=float(picks["fwd_ret_10d"].std()),
                fwd_ret_sharpe_like=float(picks["fwd_ret_10d"].mean() / picks["fwd_ret_10d"].std() * np.sqrt(252 / 10))
                if picks["fwd_ret_10d"].std() > 0 else np.nan,
                win_rate=float((picks["fwd_ret_10d"] > 0).mean()),
                y_hit_rate=float(picks["y"].mean()),
            ))
    fr = pd.DataFrame(fr_rows)
    fr.to_csv(OUT / "tradability_fwd_ret.csv", index=False)
    print("\n=== realized forward-return summary ===")
    print(fr.to_string(index=False))


if __name__ == "__main__":
    # Load sample once, reuse for both artifacts
    test = pd.read_parquet("data/processed/test.parquet")
    idx = RNG.choice(len(test), size=min(SHAP_SAMPLE, len(test)), replace=False)
    sample = test.iloc[idx].reset_index(drop=True)
    print(f"SHAP sample: {len(sample)} rows")

    shap_deep(sample)
    tradability_test()
    print(f"\nAll outputs under {OUT}")
