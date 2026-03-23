from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments.versioning import (
    DEFAULT_VERSION_REGISTRY,
    build_backtest_cfg as build_backtest_cfg_from_spec,
    build_model_cfg as build_model_cfg_from_spec,
    load_version_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _run_cmd(cmd: list[str], log_path: Path) -> tuple[int, float]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started

    log_text = [
        f"$ {' '.join(cmd)}",
        "",
        f"exit_code={proc.returncode}",
        f"elapsed_sec={elapsed:.2f}",
        "",
        "[stdout]",
        proc.stdout or "",
        "",
        "[stderr]",
        proc.stderr or "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_text), encoding="utf-8")
    return proc.returncode, elapsed


def _read_pred_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"auc": None, "aucpr": None, "rows": None}

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=["y", "prob"])
    else:
        df = pd.read_parquet(path, columns=["y", "prob"])

    return {
        "auc": float(roc_auc_score(df["y"], df["prob"])),
        "aucpr": float(average_precision_score(df["y"], df["prob"])),
        "rows": int(len(df)),
    }


def _read_backtest_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    research = data.get("research_decision", {})
    excess = data.get("excess_vs_benchmark", {})
    portfolio = data.get("portfolio", {})
    trades = data.get("trades", {})

    return {
        "test_portfolio_total_return": portfolio.get("total_return"),
        "test_portfolio_annualized_return": portfolio.get("annualized_return"),
        "test_portfolio_sharpe": portfolio.get("sharpe"),
        "test_portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "test_excess_total_return": excess.get("total_return"),
        "test_excess_annualized_return": excess.get("annualized_return"),
        "test_excess_sharpe": excess.get("sharpe"),
        "test_excess_max_drawdown": excess.get("max_drawdown"),
        "test_avg_turnover": data.get("avg_turnover"),
        "test_avg_positions": data.get("avg_positions"),
        "test_n_trades": trades.get("n_trades"),
        "test_win_rate": trades.get("win_rate"),
        "test_mean_trade_return": trades.get("mean_trade_return"),
        "policy_mode": research.get("mode"),
        "policy_use_dividend_rules": research.get("use_dividend_rules"),
        "policy_source": research.get("policy_source"),
        "policy_support_votes": research.get("support_votes"),
        "policy_total_checks": research.get("total_checks"),
        "policy_recommended": research.get("research_recommended_dividend_rules"),
    }


def _round_or_none(value: Any, digits: int = 6) -> Any:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _write_summary_files(results: list[dict[str, Any]], suite_dir: Path) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        return

    preferred_cols = [
        "version",
        "description",
        "n_num",
        "n_cat",
        "num_boost_round",
        "early_stopping_rounds",
        "dividend_rules_mode",
        "runner_kind",
        "val_aucpr",
        "test_aucpr",
        "test_portfolio_annualized_return",
        "test_portfolio_sharpe",
        "test_excess_annualized_return",
        "test_excess_sharpe",
        "policy_support_votes",
        "policy_use_dividend_rules",
        "status",
        "run_id",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    df.to_csv(suite_dir / "suite_results.csv", index=False)
    ranked = df.sort_values(["test_excess_annualized_return", "test_portfolio_sharpe"], ascending=[False, False]).reset_index(drop=True)
    ranked.to_csv(suite_dir / "suite_results_ranked.csv", index=False)

    lines = [
        f"batch_id: {suite_dir.name}",
        "",
        df.to_string(index=False),
        "",
        "Per-version configs are stored under configs/ and command logs under logs/.",
    ]
    (suite_dir / "suite_results.txt").write_text("\n".join(lines), encoding="utf-8")

    finding_lines = [
        "Ranked by test_excess_annualized_return descending:",
    ]
    for _, row in ranked.iterrows():
        finding_lines.append(
            f"{row['version']}: excess_ann={row['test_excess_annualized_return']:.6f}, "
            f"sharpe={row['test_portfolio_sharpe']:.6f}, val_aucpr={row['val_aucpr']:.6f}, "
            f"policy_votes={int(row['policy_support_votes']) if pd.notna(row['policy_support_votes']) else 'NA'}/"
            f"{int(row['policy_total_checks']) if pd.notna(row['policy_total_checks']) else 'NA'}, "
            f"policy_mode={row.get('policy_mode')}, use_rules={row.get('policy_use_dividend_rules')}"
        )
    (suite_dir / "suite_findings.txt").write_text("\n".join(finding_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all report versions on the current data snapshot.")
    parser.add_argument("--batch_id", default="report_version_suite_20260323")
    parser.add_argument("--versions", nargs="*", default=None)
    parser.add_argument("--version_registry", default=str(DEFAULT_VERSION_REGISTRY))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_reference", action="store_true", help="Skip random/oracle reference backtests to save time.")
    args = parser.parse_args()

    registry_path = PROJECT_ROOT / args.version_registry
    version_specs = load_version_specs(registry_path)
    report_targets = {version: spec.report_targets for version, spec in version_specs.items()}
    requested_versions = args.versions or list(version_specs.keys())

    requested = []
    for version in requested_versions:
        if version not in version_specs:
            raise SystemExit(f"Unknown version: {version}")
        requested.append(version)

    suite_dir = PROJECT_ROOT / "outputs" / "report_version_suite" / args.batch_id
    configs_dir = suite_dir / "configs"
    logs_dir = suite_dir / "logs"
    suite_dir.mkdir(parents=True, exist_ok=True)

    with (PROJECT_ROOT / "configs" / "model.yaml").open("r", encoding="utf-8") as f:
        base_model_cfg = yaml.safe_load(f)
    with (PROJECT_ROOT / "configs" / "backtest.yaml").open("r", encoding="utf-8") as f:
        base_backtest_cfg = yaml.safe_load(f)

    manifest = {
        "batch_id": args.batch_id,
        "versions": requested,
        "project_root": str(PROJECT_ROOT),
        "skip_reference": bool(args.skip_reference),
    }
    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    definitions = {}
    for version in requested:
        spec = version_specs[version]
        definitions[version] = {
            "description": spec.description,
            "label": spec.label,
            "num_cols": spec.num_cols,
            "cat_cols": spec.cat_cols,
            "num_boost_round": spec.num_boost_round,
            "early_stopping_rounds": spec.early_stopping_rounds,
            "dividend_rules_mode": spec.dividend_rules_mode,
            "runner_kind": spec.runner_kind,
            "random_seed": spec.random_seed if spec.runner_kind == "random" else None,
            **report_targets.get(version, {}),
        }
    (suite_dir / "version_definitions.json").write_text(
        json.dumps(definitions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for version in requested:
        spec = version_specs[version]
        run_id = f"{args.batch_id}_{version}"

        model_cfg_path = configs_dir / f"{version}_model.yaml"
        backtest_cfg_path = configs_dir / f"{version}_backtest.yaml"
        _write_yaml(model_cfg_path, build_model_cfg_from_spec(base_model_cfg, spec))
        _write_yaml(backtest_cfg_path, build_backtest_cfg_from_spec(base_backtest_cfg, spec))

        version_row: dict[str, Any] = {
            "version": version,
            "description": spec.description,
            "n_num": len(spec.num_cols),
            "n_cat": len(spec.cat_cols),
            "num_boost_round": spec.num_boost_round,
            "early_stopping_rounds": spec.early_stopping_rounds,
            "dividend_rules_mode": spec.dividend_rules_mode,
            "runner_kind": spec.runner_kind,
            "random_seed": spec.random_seed if spec.runner_kind == "random" else None,
            "run_id": run_id,
            "status": "ok",
        }
        version_row.update(report_targets.get(version, {}))

        run_dir = PROJECT_ROOT / "outputs" / "runs" / run_id
        model_artifact = PROJECT_ROOT / "models" / f"xgb_{run_id}.joblib"
        val_preds = run_dir / "preds" / "val_preds.parquet"
        test_preds = run_dir / "preds" / "test_preds.parquet"
        backtest_summary = run_dir / "backtest" / "summary.json"

        if spec.runner_kind == "random":
            version_row["train_elapsed_sec"] = None
            steps = [
                ("predict_val", ["python", "scripts/run_random_predict.py", "--run_id", run_id, "--split", "val", "--seed", str(spec.random_seed)], val_preds),
                ("predict_test", ["python", "scripts/run_random_predict.py", "--run_id", run_id, "--split", "test", "--seed", str(spec.random_seed)], test_preds),
            ]
        else:
            steps = [
                ("train", ["python", "scripts/run_train.py", "--run_id", run_id, "--model_cfg", str(model_cfg_path)], model_artifact),
                ("predict_val", ["python", "scripts/run_predict.py", "--run_id", run_id, "--split", "val"], val_preds),
                ("predict_test", ["python", "scripts/run_predict.py", "--run_id", run_id, "--split", "test"], test_preds),
            ]

        for step_name, cmd, expected_path in steps:
            if expected_path.exists() and not args.force:
                version_row[f"{step_name}_elapsed_sec"] = None
                continue
            code, elapsed = _run_cmd(cmd, logs_dir / f"{version}_{step_name}.log")
            version_row[f"{step_name}_elapsed_sec"] = round(elapsed, 2)
            if code != 0:
                version_row["status"] = f"{step_name}_failed"
                _write_summary_files(results + [version_row], suite_dir)
                results.append(version_row)
                break
        else:
            backtest_cmd = [
                "python",
                "scripts/run_backtest.py",
                "--run_id",
                run_id,
                "--split",
                "test",
                "--backtest_cfg",
                str(backtest_cfg_path),
            ]
            if args.skip_reference:
                backtest_cmd.append("--skip_reference")

            if backtest_summary.exists() and not args.force:
                version_row["backtest_elapsed_sec"] = None
            else:
                code, elapsed = _run_cmd(backtest_cmd, logs_dir / f"{version}_backtest_test.log")
                version_row["backtest_elapsed_sec"] = round(elapsed, 2)
                if code != 0:
                    version_row["status"] = "backtest_failed"

            val_metrics = _read_pred_metrics(val_preds)
            test_metrics = _read_pred_metrics(test_preds)
            bt_metrics = _read_backtest_metrics(backtest_summary)

            version_row["val_auc"] = _round_or_none(val_metrics.get("auc"))
            version_row["val_aucpr"] = _round_or_none(val_metrics.get("aucpr"))
            version_row["val_rows"] = val_metrics.get("rows")
            version_row["test_auc"] = _round_or_none(test_metrics.get("auc"))
            version_row["test_aucpr"] = _round_or_none(test_metrics.get("aucpr"))
            version_row["test_rows"] = test_metrics.get("rows")

            for key, value in bt_metrics.items():
                version_row[key] = _round_or_none(value) if isinstance(value, float) else value

            if version_row.get("report_val_aucpr") is not None and version_row.get("val_aucpr") is not None:
                version_row["delta_val_aucpr_vs_report"] = _round_or_none(
                    version_row["val_aucpr"] - version_row["report_val_aucpr"]
                )
            if version_row.get("report_ann_return") is not None and version_row.get("test_portfolio_annualized_return") is not None:
                version_row["delta_test_ann_return_vs_report"] = _round_or_none(
                    version_row["test_portfolio_annualized_return"] - version_row["report_ann_return"]
                )
            if version_row.get("report_sharpe") is not None and version_row.get("test_portfolio_sharpe") is not None:
                version_row["delta_test_sharpe_vs_report"] = _round_or_none(
                    version_row["test_portfolio_sharpe"] - version_row["report_sharpe"]
                )
            if version_row.get("report_excess_ann") is not None and version_row.get("test_excess_annualized_return") is not None:
                version_row["delta_test_excess_ann_vs_report"] = _round_or_none(
                    version_row["test_excess_annualized_return"] - version_row["report_excess_ann"]
                )

            results.append(version_row)
            _write_summary_files(results, suite_dir)
            continue

        if version_row not in results:
            results.append(version_row)
            _write_summary_files(results, suite_dir)


if __name__ == "__main__":
    main()
