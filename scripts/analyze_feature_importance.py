"""
scripts/analyze_feature_importance.py

对比 v1 和 v2 模型的特征重要性，找出哪些新特征实际上有害。

用法：
    python scripts/analyze_feature_importance.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import numpy as np

from src.utils.paths import load_yaml, resolve_paths

# ── 新增的16个特征分组 ────────────────────────────────────────────────────────
NEW_FEATURE_GROUPS = {
    "DIVAMT金额特征": [
        "div_amt_last", "div_amt_mean_exp", "div_amt_chg",
        "div_amt_dir_mean_r3", "div_no_cut_streak",
    ],
    "EXDT时间结构特征": [
        "decl_to_exdt_days", "decl_to_exdt_mean_exp",
    ],
    "OHLC衍生特征": [
        "daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct",
    ],
    "流动性特征": [
        "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike",
    ],
}

ALL_NEW_FEATURES = [f for fs in NEW_FEATURE_GROUPS.values() for f in fs]


def load_booster(run_id: str, models_dir: Path):
    path = models_dir / f"xgb_{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    art = joblib.load(path)
    return art.booster, art.num_cols + art.cat_cols


def get_importance(booster, feature_names: list, importance_type: str = "gain") -> pd.Series:
    raw = booster.get_score(importance_type=importance_type)
    # 补全未出现在树中的特征（重要性为0）
    s = pd.Series({f: raw.get(f, 0.0) for f in feature_names}, name=importance_type)
    return s.sort_values(ascending=False)


def main():
    paths_cfg = load_yaml(Path("configs/paths.yaml"))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    print("=" * 60)
    print("加载 v1 和 v2 模型...")
    print("=" * 60)

    booster_v1, features_v1 = load_booster("v1", paths.models_dir)
    booster_v2, features_v2 = load_booster("v2", paths.models_dir)

    imp_v1 = get_importance(booster_v1, features_v1)
    imp_v2 = get_importance(booster_v2, features_v2)

    # ── 1. 整体前20名对比 ─────────────────────────────────────────────────────
    print("\n【1. 特征重要性 Top 20 对比（按 gain）】")
    top20_v1 = imp_v1.head(20).reset_index()
    top20_v1.columns = ["feature", "v1_gain"]
    top20_v2 = imp_v2.head(20).reset_index()
    top20_v2.columns = ["feature", "v2_gain"]
    comparison = top20_v2.merge(
        top20_v1.rename(columns={"v1_gain": "v1_gain"}),
        on="feature", how="outer"
    ).fillna(0.0)
    comparison["v2_rank"]  = comparison["v2_gain"].rank(ascending=False).astype(int)
    comparison["is_new"]   = comparison["feature"].isin(ALL_NEW_FEATURES).map({True: "★新增", False: ""})
    comparison = comparison.sort_values("v2_gain", ascending=False)
    print(comparison[["feature", "is_new", "v2_gain", "v1_gain"]].to_string(index=False))

    # ── 2. 新增特征在 v2 中的排名和重要性 ─────────────────────────────────────
    print("\n【2. 16个新增特征在 v2 中的排名】")
    total_features = len(features_v2)
    for group_name, feats in NEW_FEATURE_GROUPS.items():
        print(f"\n  {group_name}：")
        for f in feats:
            gain = imp_v2.get(f, 0.0)
            rank = (imp_v2 > gain).sum() + 1
            pct  = gain / imp_v2.sum() * 100 if imp_v2.sum() > 0 else 0
            flag = "⚠️  低重要性" if pct < 1.0 else "✓"
            print(f"    {f:<30} rank={rank:>3}/{total_features}  gain={gain:.2f}  占比={pct:.2f}%  {flag}")

    # ── 3. v1核心特征在v2中是否被稀释 ────────────────────────────────────────
    print("\n【3. v1 核心特征在 v2 中的重要性变化】")
    core_v1_features = imp_v1.head(10).index.tolist()
    rows = []
    for f in core_v1_features:
        v1_gain = imp_v1.get(f, 0.0)
        v2_gain = imp_v2.get(f, 0.0)
        v1_rank = (imp_v1 > v1_gain).sum() + 1
        v2_rank = (imp_v2 > v2_gain).sum() + 1
        change  = v2_rank - v1_rank
        flag    = f"↓ 下降{change}名" if change > 3 else ("→ 持平" if abs(change) <= 3 else f"↑ 上升{abs(change)}名")
        rows.append({
            "feature": f,
            "v1_rank": v1_rank,
            "v2_rank": v2_rank,
            "rank_change": flag,
            "v1_gain": round(v1_gain, 2),
            "v2_gain": round(v2_gain, 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # ── 4. 输出建议 ───────────────────────────────────────────────────────────
    print("\n【4. 特征筛选建议】")
    low_importance_new = []
    for f in ALL_NEW_FEATURES:
        gain = imp_v2.get(f, 0.0)
        pct  = gain / imp_v2.sum() * 100 if imp_v2.sum() > 0 else 0
        if pct < 1.0:
            low_importance_new.append((f, round(pct, 3)))

    if low_importance_new:
        print(f"  以下新增特征重要性占比 < 1%，建议在消融实验中优先考虑剔除：")
        for f, pct in sorted(low_importance_new, key=lambda x: x[1]):
            print(f"    - {f}  (占比={pct}%)")
    else:
        print("  所有新增特征重要性均 >= 1%，特征质量尚可，问题可能在策略层。")

    # ── 5. 保存结果 ───────────────────────────────────────────────────────────
    out_path = Path("outputs/feature_importance_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_features = sorted(set(features_v1) | set(features_v2))
    result = pd.DataFrame({
        "feature": all_features,
        "v1_gain": [imp_v1.get(f, 0.0) for f in all_features],
        "v2_gain": [imp_v2.get(f, 0.0) for f in all_features],
        "is_new":  [f in ALL_NEW_FEATURES for f in all_features],
    }).sort_values("v2_gain", ascending=False)
    result.to_csv(out_path, index=False)
    print(f"\n  完整结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
