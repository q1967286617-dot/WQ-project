"""
Walk-Forward parameter sweep: top_k × holding_td on WF val folds.

Week 18 WF used top_k=20/hold=10 from single-train sweep — not re-validated
under WF.  This script:
  1. Runs WF (3 folds, 2022-2024) while capturing per-fold val predictions.
  2. Sweeps top_k × holding_td on the combined WF val predictions.
  3. Reports the best val config and runs the test backtest ONCE with those params.

Usage (from project root):
    python scripts/run_wf_param_sweep.py

Optional flags:
    --skip_wf          Skip WF training if already done (reuse existing preds)
    --combined_id      Name for the WF run folder  [default: wf_v12_sweep]
    --test_start       First test year             [default: 2022]
    --test_end         Last test year              [default: 2024]
    --train_years      Rolling train window        [default: 5]
    --seed             XGBoost seed                [default: 42]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY           = sys.executable

# ── Sweep grid ────────────────────────────────────────────────────────────────
TOP_K_GRID      = [10, 12, 15, 20]
HOLDING_TD_GRID = [5, 7, 10]

BASE_MODEL_CFG   = PROJECT_ROOT / "configs" / "versions" / "v12_model.yaml"
BASE_BACKTEST_CFG = PROJECT_ROOT / "configs" / "backtest.yaml"


def _run_backtest(run_id: str, preds_path: Path, split: str, cfg_path: Path,
                  timeout: int = 300) -> dict | None:
    cmd = [
        PY, str(PROJECT_ROOT / "scripts" / "run_backtest.py"),
        "--run_id",      run_id,
        "--split",       split,
        "--preds_path",  str(preds_path),
        "--backtest_cfg", str(cfg_path),
        "--skip_reference",
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(f"  backtest error [{run_id}]:\n{res.stderr[-600:]}")
        return None
    summary = PROJECT_ROOT / "outputs" / "runs" / run_id / "backtest" / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else None


def _sweep(preds_path: Path, split: str, prefix: str) -> list[dict]:
    with open(BASE_BACKTEST_CFG) as f:
        base_cfg = yaml.safe_load(f)

    results = []
    for top_k, holding_td in product(TOP_K_GRID, HOLDING_TD_GRID):
        cfg = {**base_cfg, "top_k": top_k, "holding_td": holding_td}
        tag = f"{prefix}_{split}_k{top_k}_h{holding_td}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = Path(tmp.name)

        print(f"  [{tag}]  top_k={top_k}  holding_td={holding_td}")
        summary = _run_backtest(tag, preds_path, split, tmp_path)
        tmp_path.unlink(missing_ok=True)

        if summary is None:
            continue
        port = summary.get("portfolio", {})
        sharpe  = port.get("sharpe", float("nan"))
        ann_ret = port.get("annualized_return", float("nan"))
        mdd     = port.get("max_drawdown", float("nan"))
        print(f"    Sharpe={sharpe:.4f}  AnnRet={ann_ret*100:.2f}%  MDD={mdd*100:.2f}%")
        results.append({"top_k": top_k, "holding_td": holding_td,
                        "sharpe": sharpe, "ann_ret": ann_ret, "mdd": mdd, "tag": tag})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_wf",    action="store_true",
                    help="Skip WF training; reuse existing combined preds under --combined_id")
    ap.add_argument("--combined_id", default="wf_v12_sweep")
    ap.add_argument("--test_start",  type=int, default=2022)
    ap.add_argument("--test_end",    type=int, default=2024)
    ap.add_argument("--train_years", type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    run_dir    = PROJECT_ROOT / "outputs" / "runs" / args.combined_id
    preds_dir  = run_dir / "preds"
    test_preds = preds_dir / "test_preds.parquet"
    val_preds  = preds_dir / "val_preds.parquet"

    # ── Step 1: Run WF (or skip) ──────────────────────────────────────────────
    if args.skip_wf and test_preds.exists() and val_preds.exists():
        print(f"[skip_wf] reusing preds in {preds_dir}")
    else:
        print("=" * 64)
        print(f"Step 1 — Walk-Forward training  (id={args.combined_id})")
        print(f"  folds: {args.test_start}..{args.test_end}  "
              f"train_years={args.train_years}  seed={args.seed}")
        print("=" * 64)
        wf_cmd = [
            PY, str(PROJECT_ROOT / "scripts" / "run_walk_forward.py"),
            "--paths",          "configs/paths.yaml",
            "--model_cfg",      str(BASE_MODEL_CFG),
            "--backtest_cfg",   str(BASE_BACKTEST_CFG),
            "--combined_id",    args.combined_id,
            "--data_suffix",    "_with_fundamentals",
            "--test_start",     str(args.test_start),
            "--test_end",       str(args.test_end),
            "--train_years",    str(args.train_years),
            "--seed",           str(args.seed),
            "--warmup_days",    "30",
            "--capture_val_preds",
        ]
        log_path = run_dir / "walk_forward_logs" / "wf_run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  logging to {log_path}")
        with log_path.open("w") as f:
            res = subprocess.run(wf_cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print(f"WF run failed — check {log_path}")
            sys.exit(res.returncode)
        print("  WF run complete.")

    if not val_preds.exists():
        print(f"ERROR: val_preds not found at {val_preds}")
        print("Re-run without --skip_wf or add --capture_val_preds to the WF command.")
        sys.exit(1)

    # ── Step 2: Sweep on combined WF val predictions ──────────────────────────
    print()
    print("=" * 64)
    print("Step 2 — Val-fold parameter sweep")
    print(f"  top_k grid:      {TOP_K_GRID}")
    print(f"  holding_td grid: {HOLDING_TD_GRID}")
    print(f"  val preds:       {val_preds}  ({len(pd.read_parquet(val_preds)):,} rows)")
    print("  NOTE: WF val years (2021-2023) lie within test.parquet → using split=test")
    print("=" * 64)

    # WF val fold years (2021-2023) are contained in test.parquet (2015-2024).
    # The backtest merges predictions with the split file on (date, permno), so
    # we must point at test.parquet to get matching price rows.
    results = _sweep(val_preds, "test", args.combined_id)

    if not results:
        print("No sweep results — check backtest errors above.")
        sys.exit(1)

    print()
    print("Val sweep results (sorted by Sharpe):")
    print(f"{'top_k':>6} {'holding_td':>10} {'Sharpe':>8} {'AnnRet%':>9} {'MDD%':>8}")
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['top_k']:>6} {r['holding_td']:>10} {r['sharpe']:>8.4f} "
              f"{r['ann_ret']*100:>9.2f} {r['mdd']*100:>8.2f}")

    best = max(results, key=lambda x: x["sharpe"])
    print(f"\nBest val config: top_k={best['top_k']}, holding_td={best['holding_td']}, "
          f"val Sharpe={best['sharpe']:.4f}")

    # ── Step 3: Single test run with best val params ───────────────────────────
    print()
    print("=" * 64)
    print("Step 3 — Test backtest with val-selected params (run ONCE)")
    print(f"  top_k={best['top_k']}  holding_td={best['holding_td']}")
    print("=" * 64)

    with open(BASE_BACKTEST_CFG) as f:
        best_cfg = yaml.safe_load(f)
    best_cfg["top_k"]      = best["top_k"]
    best_cfg["holding_td"] = best["holding_td"]

    test_tag = f"{args.combined_id}_valsel_k{best['top_k']}_h{best['holding_td']}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(best_cfg, tmp)
        tmp_path = Path(tmp.name)

    test_summary = _run_backtest(test_tag, test_preds, "test", tmp_path, timeout=600)
    tmp_path.unlink(missing_ok=True)

    print()
    print("=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)

    if test_summary:
        port = test_summary.get("portfolio", {})
        ts   = port.get("sharpe", float("nan"))
        ta   = port.get("annualized_return", float("nan"))
        td   = port.get("max_drawdown", float("nan"))
        print(f"  WF baseline (top_k=20/hold=10, fixed)  Sharpe = 0.744  (Week 18 ref)")
        print(f"  WF val-selected params                  Sharpe = {ts:.4f}")
        print(f"    top_k={best['top_k']}  holding_td={best['holding_td']}")
        print(f"    AnnRet={ta*100:.2f}%  MDD={td*100:.2f}%")
        print(f"  Output run_id: {test_tag}")

        # Save sweep summary for report
        sweep_out = {
            "combined_id":  args.combined_id,
            "best_val": best,
            "test_result": {
                "run_id":  test_tag,
                "sharpe":  ts,
                "ann_ret": ta,
                "max_drawdown": td,
            },
            "all_val_results": sorted(results, key=lambda x: x["sharpe"], reverse=True),
        }
        out_path = run_dir / "wf_param_sweep_summary.json"
        out_path.write_text(json.dumps(sweep_out, indent=2, ensure_ascii=False))
        print(f"\n  Full sweep summary saved → {out_path}")
    else:
        print("  Test backtest failed — check run logs.")


if __name__ == "__main__":
    main()
