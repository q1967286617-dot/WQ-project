"""
Probability calibration + SHAP attribution drift: 50r vs 200r classifier.

Inputs:
    outputs/runs/dual_50r/preds/test_preds.parquet   (has score_cls, y)
    outputs/runs/dual_200r/preds/test_preds.parquet
    models/xgb_dual_50r.joblib,  models/xgb_dual_200r.joblib
    data/processed/test.parquet

Outputs (under outputs/analysis/calibration_shap_50v200/):
    calibration_summary.csv       Brier / logloss / ECE / top-K precision
    reliability_50r.csv           reliability table (per-bin), 50r
    reliability_200r.csv          reliability table (per-bin), 200r
    reliability.png               both reliability curves
    shap_top_50r.csv              top-20 features by mean|SHAP|, 50r
    shap_top_200r.csv             same, 200r
    shap_compare.csv              merged table with rank deltas
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

SHAP_SAMPLE = 80_000      # rows for SHAP (keeps it under a minute)
RNG = np.random.default_rng(42)


# ─────────────────────────── calibration ────────────────────────────
def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _logloss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _reliability(p, y, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append((b, edges[b], edges[b + 1], n, np.nan, np.nan))
            continue
        rows.append(
            (b, edges[b], edges[b + 1], n, float(p[m].mean()), float(y[m].mean()))
        )
    rel = pd.DataFrame(
        rows, columns=["bin", "lo", "hi", "count", "avg_pred", "avg_true"]
    )
    valid = rel.dropna()
    ece = float(
        (valid["count"] / valid["count"].sum()
         * (valid["avg_pred"] - valid["avg_true"]).abs()).sum()
    )
    return rel, ece


def _topk_daily_precision(df: pd.DataFrame, k: int) -> float:
    """Mean precision@k per trading day using score_cls ranking."""
    def _p(g):
        g = g.nlargest(k, "score_cls")
        return g["y"].mean()
    return float(df.groupby("date").apply(_p).mean())


def calibration_block():
    summary = []
    for tag in ["dual_50r", "dual_200r"]:
        p = pd.read_parquet(f"outputs/runs/{tag}/preds/test_preds.parquet")
        pr = p["score_cls"].to_numpy()
        y = p["y"].to_numpy().astype(float)
        rel, ece = _reliability(pr, y, n_bins=15)
        rel.to_csv(OUT / f"reliability_{tag}.csv", index=False)
        summary.append(dict(
            model=tag,
            n=len(y),
            base_rate=float(y.mean()),
            pred_mean=float(pr.mean()),
            pred_std=float(pr.std()),
            brier=_brier(pr, y),
            logloss=_logloss(pr, y),
            ece=ece,
            prec_at_50=_topk_daily_precision(p, 50),
            prec_at_100=_topk_daily_precision(p, 100),
            prec_at_200=_topk_daily_precision(p, 200),
        ))
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "calibration_summary.csv", index=False)
    print("\n=== calibration summary ===")
    print(summ.to_string(index=False))

    # Reliability plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for tag, color in [("dual_50r", "tab:blue"), ("dual_200r", "tab:red")]:
        rel = pd.read_csv(OUT / f"reliability_{tag}.csv").dropna()
        ax.plot(rel["avg_pred"], rel["avg_true"], "o-", color=color, label=tag)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
    ax.set_xlabel("mean predicted prob (bin)")
    ax.set_ylabel("empirical frequency of y=1 (bin)")
    ax.set_title("Reliability: 50r vs 200r classifier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT/'reliability.png'}")


# ─────────────────────────── SHAP ────────────────────────────
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


def shap_block():
    test = pd.read_parquet("data/processed/test.parquet")
    n = len(test)
    idx = RNG.choice(n, size=min(SHAP_SAMPLE, n), replace=False)
    sample = test.iloc[idx].reset_index(drop=True)
    print(f"SHAP sample: {len(sample)} rows from test ({n})")

    shap_tables = {}
    for tag in ["dual_50r", "dual_200r"]:
        art = joblib.load(f"models/xgb_{tag}.joblib")
        X = _prep_for_artifact(sample, art)
        feat_names = art.feature_names or (art.num_cols + art.cat_cols)
        dm = build_dmatrix(X, y=None, enable_categorical=True, feature_names=feat_names)
        # pred_contribs returns (n, n_features+1); last column is bias
        contribs = art.booster.predict(dm, pred_contribs=True)
        contribs = contribs[:, :-1]
        mean_abs = np.abs(contribs).mean(axis=0)
        mean_signed = contribs.mean(axis=0)
        tbl = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        tbl["rank"] = tbl.index + 1
        tbl.head(20).to_csv(OUT / f"shap_top_{tag}.csv", index=False)
        shap_tables[tag] = tbl
        print(f"\n=== top-10 SHAP {tag} ===")
        print(tbl.head(10)[["feature", "mean_abs_shap", "mean_signed_shap", "rank"]].to_string(index=False))

    # Merge for drift comparison
    a = shap_tables["dual_50r"].rename(
        columns={"mean_abs_shap": "abs_50r", "mean_signed_shap": "signed_50r", "rank": "rank_50r"}
    )
    b = shap_tables["dual_200r"].rename(
        columns={"mean_abs_shap": "abs_200r", "mean_signed_shap": "signed_200r", "rank": "rank_200r"}
    )
    m = a.merge(b, on="feature", how="outer")
    m["abs_delta_200_minus_50"] = m["abs_200r"] - m["abs_50r"]
    m["rank_shift"] = m["rank_50r"] - m["rank_200r"]  # positive = moved up in 200r
    m = m.sort_values("abs_200r", ascending=False).reset_index(drop=True)
    m.to_csv(OUT / "shap_compare.csv", index=False)
    print(f"\nwrote {OUT/'shap_compare.csv'} ({len(m)} features)")

    # Top movers
    top_up = m.dropna(subset=["rank_50r", "rank_200r"]).nlargest(10, "rank_shift")
    top_down = m.dropna(subset=["rank_50r", "rank_200r"]).nsmallest(10, "rank_shift")
    print("\n=== features rising most (rank 50r → 200r) ===")
    print(top_up[["feature", "rank_50r", "rank_200r", "rank_shift", "abs_50r", "abs_200r"]].to_string(index=False))
    print("\n=== features falling most ===")
    print(top_down[["feature", "rank_50r", "rank_200r", "rank_shift", "abs_50r", "abs_200r"]].to_string(index=False))


if __name__ == "__main__":
    calibration_block()
    shap_block()
    print(f"\nAll outputs under {OUT}")
