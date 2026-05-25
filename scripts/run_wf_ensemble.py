"""
Walk-Forward multi-seed ensemble.

Runs WF with seeds {42, 7, 13}, averages per-day per-stock prob scores,
then backtests the ensemble predictions.  Averaging across seeds reduces
run-to-run variance without adding training data or new features.

Usage (from project root):
    python scripts/run_wf_ensemble.py

Optional flags:
    --skip_wf          Skip WF training for seeds already completed
    --seeds            Space-separated seed list          [default: 42 7 13]
    --ensemble_id      Output folder name                 [default: wf_v12_ens3]
    --test_start       First test year                    [default: 2022]
    --test_end         Last test year                     [default: 2024]
    --train_years      Rolling train window               [default: 5]
    --top_k            Top-K stocks to hold               [default: 20]
    --holding_td       Holding period (trading days)      [default: 10]

To use val-selected params from run_wf_param_sweep.py, pass the best
top_k and holding_td explicitly, e.g.:
    python scripts/run_wf_ensemble.py --top_k 15 --holding_td 7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY           = sys.executable

BASE_MODEL_CFG    = PROJECT_ROOT / "configs" / "versions" / "v12_model.yaml"
BASE_BACKTEST_CFG = PROJECT_ROOT / "configs" / "backtest.yaml"


def _run_wf_for_seed(seed: int, combined_id: str, test_start: int, test_end: int,
                     train_years: int) -> Path:
    """Run walk-forward for one seed; return path to combined test_preds.parquet."""
    run_dir    = PROJECT_ROOT / "outputs" / "runs" / combined_id
    preds_path = run_dir / "preds" / "test_preds.parquet"

    if preds_path.exists():
        print(f"  [seed={seed}] reusing existing preds at {preds_path}")
        return preds_path

    print(f"  [seed={seed}] running WF → {combined_id}")
    log_path = run_dir / "walk_forward_logs" / "wf_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, str(PROJECT_ROOT / "scripts" / "run_walk_forward.py"),
        "--paths",        "configs/paths.yaml",
        "--model_cfg",    str(BASE_MODEL_CFG),
        "--backtest_cfg", str(BASE_BACKTEST_CFG),
        "--combined_id",  combined_id,
        "--data_suffix",  "_with_fundamentals",
        "--test_start",   str(test_start),
        "--test_end",     str(test_end),
        "--train_years",  str(train_years),
        "--seed",         str(seed),
        "--warmup_days",  "30",
    ]
    with log_path.open("w") as f:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f"  WF failed for seed={seed} — check {log_path}")
        sys.exit(res.returncode)

    print(f"  [seed={seed}] WF done.")
    return preds_path


def _run_backtest(run_id: str, preds_path: Path, cfg_path: Path,
                  timeout: int = 600) -> dict | None:
    cmd = [
        PY, str(PROJECT_ROOT / "scripts" / "run_backtest.py"),
        "--run_id",       run_id,
        "--split",        "test",
        "--preds_path",   str(preds_path),
        "--backtest_cfg", str(cfg_path),
        "--skip_reference",
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(f"  backtest error:\n{res.stderr[-600:]}")
        return None
    summary = PROJECT_ROOT / "outputs" / "runs" / run_id / "backtest" / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_wf",    action="store_true",
                    help="Reuse per-seed WF runs if already completed")
    ap.add_argument("--seeds",      type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--ensemble_id", default="wf_v12_ens3")
    ap.add_argument("--test_start",  type=int, default=2022)
    ap.add_argument("--test_end",    type=int, default=2024)
    ap.add_argument("--train_years", type=int, default=5)
    ap.add_argument("--top_k",       type=int, default=20)
    ap.add_argument("--holding_td",  type=int, default=10)
    args = ap.parse_args()

    seeds = args.seeds
    print("=" * 64)
    print(f"WF Multi-Seed Ensemble")
    print(f"  seeds:       {seeds}")
    print(f"  ensemble_id: {args.ensemble_id}")
    print(f"  folds:       {args.test_start}..{args.test_end}  train_years={args.train_years}")
    print(f"  backtest:    top_k={args.top_k}  holding_td={args.holding_td}")
    print("=" * 64)

    # ── Step 1: Per-seed WF runs ───────────────────────────────────────────────
    seed_preds: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        combined_id = f"wf_v12_s{seed}"
        preds_path  = _run_wf_for_seed(seed, combined_id, args.test_start,
                                        args.test_end, args.train_years)
        df = pd.read_parquet(preds_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        seed_preds[seed] = df
        print(f"  seed={seed}: {len(df):,} rows  date={df['date'].min().date()}..{df['date'].max().date()}")

    # ── Step 2: Ensemble — average prob across seeds ───────────────────────────
    print()
    print("Averaging prob scores across seeds...")

    # Use seed-42 as the spine; join others on (date, permno)
    base_df = seed_preds[seeds[0]].copy()
    prob_sum = base_df["prob"].values.copy()

    for seed in seeds[1:]:
        other = seed_preds[seed][["date", "permno", "prob"]].rename(columns={"prob": f"prob_{seed}"})
        base_df = base_df.merge(other, on=["date", "permno"], how="left")
        filled = base_df[f"prob_{seed}"].fillna(base_df["prob"])
        prob_sum = prob_sum + filled.values

    base_df["prob"] = prob_sum / len(seeds)

    # Drop any per-seed helper columns
    drop_cols = [c for c in base_df.columns if c.startswith("prob_")]
    ensemble_preds = base_df.drop(columns=drop_cols)

    n_missing = ensemble_preds["prob"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows with NaN prob after ensemble — filling with seed-42 prob")
        ensemble_preds["prob"] = ensemble_preds["prob"].fillna(seed_preds[seeds[0]]["prob"])

    print(f"  Ensemble preds: {len(ensemble_preds):,} rows  "
          f"prob mean={ensemble_preds['prob'].mean():.4f} "
          f"std={ensemble_preds['prob'].std():.4f}")

    # ── Step 3: Save ensemble preds ───────────────────────────────────────────
    ens_run_dir = PROJECT_ROOT / "outputs" / "runs" / args.ensemble_id
    ens_preds_dir = ens_run_dir / "preds"
    ens_preds_dir.mkdir(parents=True, exist_ok=True)
    ens_preds_path = ens_preds_dir / "test_preds.parquet"
    ensemble_preds.to_parquet(ens_preds_path, index=False)
    print(f"  Saved ensemble preds → {ens_preds_path}")

    # Record metadata
    meta = {
        "seeds":       seeds,
        "top_k":       args.top_k,
        "holding_td":  args.holding_td,
        "test_start":  args.test_start,
        "test_end":    args.test_end,
        "train_years": args.train_years,
        "seed_run_ids": [f"wf_v12_s{s}" for s in seeds],
    }
    (ens_run_dir / "ensemble_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # ── Step 4: Per-seed individual backtests (for paired comparison) ──────────
    print()
    print("Running per-seed individual backtests for comparison...")

    with open(BASE_BACKTEST_CFG) as f:
        bt_cfg = yaml.safe_load(f)
    bt_cfg["top_k"]      = args.top_k
    bt_cfg["holding_td"] = args.holding_td

    seed_sharpes: dict[int, float] = {}
    for seed in seeds:
        sid = f"wf_v12_s{seed}"
        sp  = PROJECT_ROOT / "outputs" / "runs" / sid / "preds" / "test_preds.parquet"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(bt_cfg, tmp)
            tmp_path = Path(tmp.name)
        bt_tag = f"{sid}_bt_k{args.top_k}_h{args.holding_td}"
        s = _run_backtest(bt_tag, sp, tmp_path)
        tmp_path.unlink(missing_ok=True)
        if s:
            sharpe = s.get("portfolio", {}).get("sharpe", float("nan"))
            seed_sharpes[seed] = sharpe
            print(f"  seed={seed}  Sharpe={sharpe:.4f}")

    # ── Step 5: Ensemble backtest ──────────────────────────────────────────────
    print()
    print("Running ensemble backtest...")

    with open(BASE_BACKTEST_CFG) as f:
        bt_cfg = yaml.safe_load(f)
    bt_cfg["top_k"]      = args.top_k
    bt_cfg["holding_td"] = args.holding_td

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(bt_cfg, tmp)
        tmp_path = Path(tmp.name)

    ens_bt_id = f"{args.ensemble_id}_bt_k{args.top_k}_h{args.holding_td}"
    ens_summary = _run_backtest(ens_bt_id, ens_preds_path, tmp_path)
    tmp_path.unlink(missing_ok=True)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("ENSEMBLE RESULTS SUMMARY")
    print("=" * 64)

    mean_seed_sharpe = (sum(seed_sharpes.values()) / len(seed_sharpes)) if seed_sharpes else float("nan")
    print(f"  Per-seed Sharpe:  {dict(seed_sharpes)}")
    print(f"  Mean seed Sharpe: {mean_seed_sharpe:.4f}")

    if ens_summary:
        port = ens_summary.get("portfolio", {})
        es   = port.get("sharpe", float("nan"))
        ea   = port.get("annualized_return", float("nan"))
        ed   = port.get("max_drawdown", float("nan"))
        print(f"  Ensemble Sharpe:  {es:.4f}  (+{es - mean_seed_sharpe:+.4f} vs mean seed)")
        print(f"  AnnRet={ea*100:.2f}%  MDD={ed*100:.2f}%")
        print(f"  Output run_id:    {ens_bt_id}")

        final = {
            "seed_sharpes":     seed_sharpes,
            "mean_seed_sharpe": mean_seed_sharpe,
            "ensemble_sharpe":  es,
            "delta_vs_mean":    es - mean_seed_sharpe,
            "ensemble_ann_ret": ea,
            "ensemble_mdd":     ed,
            "top_k":            args.top_k,
            "holding_td":       args.holding_td,
        }
        out_path = ens_run_dir / "wf_ensemble_summary.json"
        out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False))
        print(f"\n  Full summary saved → {out_path}")
    else:
        print("  Ensemble backtest failed — check run logs.")


if __name__ == "__main__":
    main()
