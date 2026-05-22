"""
SHAP analysis for v12_run1 — macro vs fundamental vs rhythm features.

Outputs (saved to outputs/runs/<run_id>/research/shap/):
  summary_bar.png        : mean |SHAP| bar chart, all features, colored by group
  summary_beeswarm.png   : beeswarm plot top-30 features
  dependence_dgs10.png   : SHAP dependence plot for dgs10
  dependence_vix.png     : SHAP dependence plot for vix
  dependence_mkt_vol21.png
  group_importance.csv   : aggregated mean |SHAP| by feature group

Usage:
    python scripts/analyze_shap_macro.py --run_id v12_run1
    python scripts/analyze_shap_macro.py --run_id v12_macro
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.modeling.preprocess import apply_imputer_and_scaler
from src.modeling.train import build_dmatrix
from src.utils.paths import load_yaml, resolve_paths

# ── feature groups ───────────────────────────────────────────────────────────
RHYTHM_FEATURES = [
    "days_since_last_div", "gap_mean_exp", "gap_med_exp", "gap_std_exp",
    "gap_mean_rN", "gap_std_rN", "gap_cv_exp", "time_to_med_exp",
    "z_to_med_exp", "time_to_mean_rN", "z_to_mean_rN", "div_count_exp",
    "div_amt_last", "div_amt_mean_exp", "div_amt_chg", "div_amt_dir_mean_r3",
    "div_no_cut_streak", "decl_to_exdt_days", "decl_to_exdt_mean_exp",
]
PRICE_FEATURES = [
    "log_mkt_cap", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "mom_12_1",
    "vol_5d", "vol_21d", "turnover_5d", "turnover_21d", "price_to_high",
    "volume_spike", "vol_ratio", "turnover_ratio", "ret_rel_to_ind",
    "daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct",
    "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike",
]
FUNDAMENTAL_FEATURES = [
    "roe", "leverage", "profit_margin", "payout_ratio", "div_coverage", "dvpsxq",
]
MACRO_FEATURES = [
    "mkt_ret_5d", "mkt_ret_21d", "mkt_ret_63d",
    "mkt_vol_21d", "mkt_vol_63d", "vix", "dgs10",
]
CALENDAR_FEATURES = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "quarter", "weekday", "is_month_start", "is_month_end",
]

GROUP_COLORS = {
    "Dividend Rhythm": "#E74C3C",
    "Price/Volume":    "#3498DB",
    "Fundamentals":    "#2ECC71",
    "Macro":           "#F39C12",
    "Calendar":        "#9B59B6",
    "Other":           "#95A5A6",
}

def _assign_group(feat: str) -> str:
    if feat in RHYTHM_FEATURES:      return "Dividend Rhythm"
    if feat in PRICE_FEATURES:       return "Price/Volume"
    if feat in FUNDAMENTAL_FEATURES: return "Fundamentals"
    if feat in MACRO_FEATURES:       return "Macro"
    if feat in CALENDAR_FEATURES:    return "Calendar"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id",    default="v12_run1")
    ap.add_argument("--paths",     default="configs/paths.yaml")
    ap.add_argument("--n_samples", type=int, default=5000,
                    help="Number of test rows to compute SHAP on (speed vs accuracy).")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    paths     = resolve_paths(paths_cfg, project_root=Path.cwd())

    out_dir = paths.outputs_dir / args.run_id / "research" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────────────
    art_path = paths.models_dir / f"xgb_{args.run_id}.joblib"
    if not art_path.exists():
        raise FileNotFoundError(art_path)
    art = joblib.load(art_path)
    print(f"Loaded: {art_path}  features={len(art.feature_names)}")

    # ── load & preprocess test data ──────────────────────────────────────────
    test_df = pd.read_parquet(paths.processed_dir / "test.parquet")
    test_df["DlyCalDt"] = pd.to_datetime(test_df["DlyCalDt"])

    test_df = apply_imputer_and_scaler(test_df, art.num_cols, art.impute_stats,
                                       do_scale=art.do_scale)
    for c in art.cat_cols:
        if c not in test_df.columns:
            test_df[c] = "<<MISSING>>"
        test_df[c] = test_df[c].astype("string").fillna("<<MISSING>>")
        cats = art.cat_categories.get(c)
        if cats:
            cats_idx = pd.Index(cats)
            unk = "<<UNK>>"
            if unk not in cats_idx:
                cats_idx = cats_idx.append(pd.Index([unk]))
            test_df[c] = pd.Categorical(
                test_df[c].where(test_df[c].isin(cats_idx), unk),
                categories=cats_idx
            )
        else:
            test_df[c] = test_df[c].astype("category")

    feat_names = art.feature_names or (art.num_cols + art.cat_cols)
    X_full = test_df[feat_names].copy()
    for c in feat_names:
        if c not in X_full.columns:
            X_full[c] = np.nan

    # sample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_full), size=min(args.n_samples, len(X_full)), replace=False)
    X_sample = X_full.iloc[idx].reset_index(drop=True)

    print(f"Computing SHAP on {len(X_sample):,} samples ...")
    dm = build_dmatrix(X_sample, y=None, enable_categorical=True,
                       feature_names=feat_names)

    # shap.Explainer handles XGBoost version quirks better than TreeExplainer
    try:
        explainer = shap.Explainer(art.booster)
        shap_out  = explainer(dm)
        shap_values = shap_out.values          # (n, p)
    except Exception as e:
        print(f"shap.Explainer failed ({e}), falling back to predict_contributions")
        # XGBoost native SHAP via predict
        import xgboost as xgb
        contribs = art.booster.predict(dm, pred_contribs=True)
        shap_values = contribs[:, :-1]          # drop bias column

    shap_df = pd.DataFrame(shap_values, columns=feat_names)

    # ── 1. group-level importance ────────────────────────────────────────────
    mean_abs = shap_df.abs().mean()
    group_imp = mean_abs.groupby(mean_abs.index.map(_assign_group)).sum()
    group_imp = group_imp.sort_values(ascending=False)
    group_imp.to_csv(out_dir / "group_importance.csv")
    print("\nGroup-level mean |SHAP|:")
    for g, v in group_imp.items():
        print(f"  {g:<20} {v:.5f}")

    # ── 2. summary bar chart (all features, colored by group) ────────────────
    feat_imp = mean_abs.sort_values(ascending=False)
    colors   = [GROUP_COLORS.get(_assign_group(f), "#95A5A6") for f in feat_imp.index]

    fig, ax = plt.subplots(figsize=(10, max(8, len(feat_imp) * 0.22)))
    bars = ax.barh(range(len(feat_imp)), feat_imp.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp.index[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Feature Importance — {args.run_id}", fontsize=13)
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'summary_bar.png'}")

    # ── 3. beeswarm top-30 ───────────────────────────────────────────────────
    top30 = feat_imp.head(30).index.tolist()
    shap_top = shap_df[top30].values
    X_top    = X_sample[top30].copy()
    # encode cats to numeric for display
    for c in top30:
        if X_top[c].dtype.name == "category":
            X_top[c] = X_top[c].cat.codes.astype(float)
        else:
            X_top[c] = pd.to_numeric(X_top[c], errors="coerce")

    shap.summary_plot(shap_top, X_top, feature_names=top30,
                      show=False, plot_size=(10, 9))
    plt.title(f"SHAP Beeswarm — Top 30 Features ({args.run_id})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_beeswarm.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'summary_beeswarm.png'}")

    # ── 4. dependence plots for macro features ───────────────────────────────
    macro_present = [f for f in MACRO_FEATURES if f in feat_names]
    for feat in macro_present:
        if feat not in shap_df.columns:
            continue
        x_vals = pd.to_numeric(X_sample[feat], errors="coerce")
        y_vals = shap_df[feat].values
        valid  = ~np.isnan(x_vals)

        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(x_vals[valid], y_vals[valid],
                        c=y_vals[valid], cmap="RdYlGn",
                        alpha=0.3, s=8, rasterized=True)
        plt.colorbar(sc, ax=ax, label="SHAP value")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel(f"SHAP value for {feat}", fontsize=11)
        ax.set_title(f"SHAP Dependence — {feat}  ({args.run_id})", fontsize=12)

        # annotate direction
        slope = np.polyfit(x_vals[valid], y_vals[valid], 1)[0]
        direction = "↑ higher → more likely dividend" if slope > 0 \
                    else "↑ higher → less likely dividend"
        ax.text(0.05, 0.95, direction, transform=ax.transAxes,
                fontsize=9, va="top",
                color="green" if slope > 0 else "red")

        plt.tight_layout()
        safe = feat.replace("/", "_")
        fig.savefig(out_dir / f"dependence_{safe}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: dependence_{feat}.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
