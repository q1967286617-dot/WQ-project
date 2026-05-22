"""
Week 17 实验二：波动率目标化（Vol-Targeting）

对现有回测的日收益序列施加动态杠杆，使组合波动率稳定在 target_vol，
从而降低最大回撤，同时观察对 Sharpe 的影响。

Usage:
    python scripts/run_voltarget.py --run_ids v12_run1 v12_s0 v12_s7 --target_vol 0.15
    python scripts/run_voltarget.py --run_id v12_run1 --target_vol 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.paths import ensure_dir, resolve_paths, load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_daily(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "backtest" / "daily_portfolio.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _apply_voltarget(daily: pd.DataFrame, target_vol: float,
                     window: int = 21, max_leverage: float = 2.0) -> pd.DataFrame:
    """
    动态杠杆：每日杠杆 = target_vol / rolling_vol，上限为 max_leverage。
    第一个窗口内杠杆=1（无足够历史）。
    """
    df = daily.copy()
    ret = df["portfolio_ret"].fillna(0)
    bench_ret = df["benchmark_ret"].fillna(0)

    # 用前一天的滚动波动率决定今天杠杆（无前视偏差）
    roll_vol = ret.shift(1).rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
    leverage = (target_vol / roll_vol).clip(upper=max_leverage)
    leverage = leverage.fillna(1.0)

    df["leverage"] = leverage
    df["vt_ret"] = ret * leverage
    df["excess_ret"] = ret - bench_ret
    df["vt_excess_ret"] = df["vt_ret"] - bench_ret
    return df


def _metrics(ret: pd.Series, label: str = "") -> dict:
    ret = ret.fillna(0)
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    cum = (1 + ret).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return {"label": label, "ann_return": ann,
            "sharpe": sharpe, "max_drawdown": mdd}


def _plot_nav(df: pd.DataFrame, run_id: str, out_dir: Path) -> None:
    orig_nav = (1 + df["portfolio_ret"].fillna(0)).cumprod()
    vt_nav   = (1 + df["vt_ret"].fillna(0)).cumprod()
    bench    = (1 + df["benchmark_ret"].fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], orig_nav,  label="Original", linewidth=1.5)
    ax.plot(df["date"], vt_nav,    label="Vol-Targeted", linewidth=1.5, linestyle="--")
    ax.plot(df["date"], bench,     label="Benchmark", linewidth=1, color="gray", alpha=0.7)
    ax.set_title(f"Vol-Targeting: {run_id}")
    ax.set_ylabel("NAV")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"nav_{run_id}.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_ids", nargs="+",
                    default=["v12_run1", "v12_s0", "v12_s7", "v12_s123", "v12_s999", "v12_s2025"])
    ap.add_argument("--target_vol", type=float, default=0.15)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--max_leverage", type=float, default=2.0)
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--out_dir", default="outputs/voltarget")
    args = ap.parse_args()

    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)
    out_dir = PROJECT_ROOT / args.out_dir
    ensure_dir(out_dir)

    rows = []
    vt_daily_list = []

    for run_id in args.run_ids:
        run_dir = paths.outputs_dir / run_id
        daily_path = run_dir / "backtest" / "daily_portfolio.csv"
        if not daily_path.exists():
            print(f"[WARN] {run_id}: daily_portfolio.csv not found, skipping")
            continue

        daily = _load_daily(run_dir)
        df_vt = _apply_voltarget(daily, args.target_vol, args.window, args.max_leverage)

        orig = _metrics(df_vt["portfolio_ret"], label=f"{run_id}_original")
        vt   = _metrics(df_vt["vt_ret"],        label=f"{run_id}_voltarget")
        rows.extend([orig, vt])

        _plot_nav(df_vt, run_id, out_dir)
        df_vt["run_id"] = run_id
        vt_daily_list.append(df_vt)

        print(f"\n{run_id}:")
        print(f"  Original   → Sharpe={orig['sharpe']:.3f}  Ann={orig['ann_return']:.2%}  MDD={orig['max_drawdown']:.2%}")
        print(f"  Vol-Target → Sharpe={vt['sharpe']:.3f}  Ann={vt['ann_return']:.2%}  MDD={vt['max_drawdown']:.2%}")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "voltarget_metrics.csv", index=False)

    # Ensemble: 6 종子的 vt_ret 均值
    if vt_daily_list:
        all_vt = pd.concat(vt_daily_list)
        ens_vt = all_vt.groupby("date")[["portfolio_ret", "vt_ret", "benchmark_ret"]].mean()
        ens_orig = _metrics(ens_vt["portfolio_ret"], "ensemble_original")
        ens_vt_m = _metrics(ens_vt["vt_ret"],        "ensemble_voltarget")
        print(f"\n{'='*50}")
        print("Ensemble (6-seed mean):")
        print(f"  Original   → Sharpe={ens_orig['sharpe']:.3f}  Ann={ens_orig['ann_return']:.2%}  MDD={ens_orig['max_drawdown']:.2%}")
        print(f"  Vol-Target → Sharpe={ens_vt_m['sharpe']:.3f}  Ann={ens_vt_m['ann_return']:.2%}  MDD={ens_vt_m['max_drawdown']:.2%}")

        ens_df_out = ens_vt.reset_index()
        ens_df_out.to_csv(out_dir / "ensemble_daily.csv", index=False)

    print(f"\nResults saved → {out_dir}")


if __name__ == "__main__":
    main()
