"""
Val-split parameter sweep (any run_id).

Sweeps top_k × holding_td on the VAL split only.
Picks the best config by val Sharpe, then runs ONCE on test.

Usage:
    python scripts/sweep_val_params.py --source_run v12_run1
    python scripts/sweep_val_params.py --source_run v12_macro --run_test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CFG_PATH = PROJECT_ROOT / "configs" / "backtest.yaml"

TOP_K_GRID      = [10, 12, 15, 20]
HOLDING_TD_GRID = [5, 7, 10]


def run_backtest(run_id: str, preds_path: Path, split: str, cfg_path: Path) -> dict | None:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_backtest.py"),
        "--run_id", run_id,
        "--split", split,
        "--preds_path", str(preds_path),
        "--backtest_cfg", str(cfg_path),
        "--skip_reference",
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR (run_id={run_id}):\n{result.stderr[-800:]}")
        return None
    out_dir = PROJECT_ROOT / "outputs" / "runs" / run_id / "backtest"
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        print(f"  no summary.json for {run_id}")
        return None
    with open(summary_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", default="v12_run1",
                    help="Run-id whose val/test_preds.parquet to use (default: v12_run1).")
    ap.add_argument("--run_test", action="store_true",
                    help="After sweep, run best config on test split.")
    args = ap.parse_args()

    preds_dir    = PROJECT_ROOT / "outputs" / "runs" / args.source_run / "preds"
    val_preds    = preds_dir / "val_preds.parquet"
    test_preds   = preds_dir / "test_preds.parquet"
    sweep_prefix = args.source_run  # tag prefix for sweep output dirs

    if not val_preds.exists():
        raise FileNotFoundError(f"val_preds not found: {val_preds}")

    with open(BASE_CFG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    print("=" * 60)
    print(f"Source run:  {args.source_run}")
    print(f"Val-split sweep: top_k={TOP_K_GRID}  holding_td={HOLDING_TD_GRID}")
    print(f"Val preds:   {val_preds}")
    print("=" * 60)

    results = []
    for top_k, holding_td in product(TOP_K_GRID, HOLDING_TD_GRID):
        cfg = {**base_cfg, "top_k": top_k, "holding_td": holding_td}
        tag = f"{sweep_prefix}_val_k{top_k}_h{holding_td}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = Path(tmp.name)

        print(f"\n[{tag}]  top_k={top_k}, holding_td={holding_td}")
        summary = run_backtest(tag, val_preds, "val", tmp_path)
        tmp_path.unlink(missing_ok=True)

        if summary is None:
            continue
        port = summary.get("portfolio", {})
        sharpe = port.get("sharpe", float("nan"))
        ann_ret = port.get("annualized_return", float("nan"))
        mdd = port.get("max_drawdown", float("nan"))
        print(f"  Sharpe={sharpe:.4f}  AnnRet={ann_ret:.4f}  MDD={mdd:.4f}")
        results.append({"top_k": top_k, "holding_td": holding_td, "sharpe": sharpe,
                        "ann_ret": ann_ret, "mdd": mdd, "tag": tag})

    if not results:
        print("No results collected — check errors above.")
        return

    print("\n" + "=" * 60)
    print("Val sweep results (sorted by Sharpe):")
    print(f"{'top_k':>6} {'holding_td':>10} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8}")
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['top_k']:>6} {r['holding_td']:>10} {r['sharpe']:>8.4f} {r['ann_ret']:>8.4f} {r['mdd']:>8.4f}")

    best = max(results, key=lambda x: x["sharpe"])
    print(f"\nBest val config: top_k={best['top_k']}, holding_td={best['holding_td']}, "
          f"Sharpe={best['sharpe']:.4f}")

    if args.run_test:
        print("\n" + "=" * 60)
        print(f"Running ONCE on test split with best config: {best}")
        best_cfg = {**base_cfg, "top_k": best["top_k"], "holding_td": best["holding_td"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(best_cfg, tmp)
            tmp_path = Path(tmp.name)

        test_tag = f"{sweep_prefix}_valsel_k{best['top_k']}_h{best['holding_td']}"
        test_summary = run_backtest(test_tag, test_preds, "test", tmp_path)
        tmp_path.unlink(missing_ok=True)

        if test_summary:
            port = test_summary.get("portfolio", {})
            print(f"\n  TEST RESULT (val-selected params):")
            print(f"  Sharpe    = {port.get('sharpe', float('nan')):.4f}")
            print(f"  AnnRet    = {port.get('annualized_return', float('nan')):.4f}")
            print(f"  MDD       = {port.get('max_drawdown', float('nan')):.4f}")
            print(f"\n  Baseline: v12_run1 val-selected test Sharpe = 0.638")
            print(f"  This run ({args.source_run}) val-selected test Sharpe = {port.get('sharpe', float('nan')):.4f}")


if __name__ == "__main__":
    main()
