from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.backtest_analysis import (  # noqa: E402
    add_liquidity_bucket,
    compute_trade_group_metrics,
    compute_yearly_backtest_metrics,
    enrich_trades_with_split_context,
)
from src.analysis.plots import plot_grouped_bar, plot_nav_vs_benchmark  # noqa: E402
from src.experiments.versioning import (  # noqa: E402
    DEFAULT_VERSION_REGISTRY,
    load_version_specs,
    materialize_version_configs,
)
from src.modeling.train import load_artifacts  # noqa: E402
from src.utils.paths import ensure_dir, load_yaml, resolve_paths  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"$ {' '.join(cmd)}",
                f"exit_code={proc.returncode}",
                f"elapsed_sec={elapsed:.2f}",
                "",
                "[stdout]",
                proc.stdout or "",
                "",
                "[stderr]",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return proc.returncode, elapsed


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_preds_path(run_dir: Path, split: str) -> Path:
    parquet = run_dir / "preds" / f"{split}_preds.parquet"
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Missing predictions for split={split}: {parquet}")


def _pred_metrics(path: Path) -> dict[str, Any]:
    df = _read_df(path)
    return {
        "rows": int(len(df)),
        "auc": float(roc_auc_score(df["y"], df["prob"])),
        "aucpr": float(average_precision_score(df["y"], df["prob"])),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_report(
    run_dir: Path,
    spec,
    metrics: dict[str, Any],
    yearly_df: pd.DataFrame,
    history_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
) -> None:
    std_dir = run_dir / "standardized"
    charts_dir = std_dir / "charts"
    report_path = std_dir / "report.md"
    lines: list[str] = []
    lines.append(f"# 标准化实验报告 | {spec.version}")
    lines.append("")
    lines.append(f"- 版本: `{spec.version}`")
    lines.append(f"- 标签: `{spec.label}`")
    lines.append(f"- 运行目录: `{run_dir}`")
    lines.append("")

    train_metrics = metrics.get("training", {})
    event_metrics = metrics.get("event_eval", {})
    backtest_metrics = metrics.get("backtest", {})
    three_way = backtest_metrics.get("three_way_comparison", {})
    alpha_capture = three_way.get("alpha_capture", {})

    lines.append("## 1. 训练指标")
    lines.append("")
    lines.append(
        f"- `val_auc={train_metrics.get('val_auc', 0.0):.4f}`, "
        f"`val_aucpr={train_metrics.get('val_aucpr', 0.0):.4f}`, "
        f"`test_auc={train_metrics.get('test_auc', 0.0):.4f}`, "
        f"`test_aucpr={train_metrics.get('test_aucpr', 0.0):.4f}`。"
    )
    if train_metrics.get("best_iteration") is not None:
        lines.append(f"- `best_iteration={train_metrics['best_iteration']}`。")
    lines.append("")

    global_metrics = event_metrics.get("global_metrics", {})
    event_summary = event_metrics.get("event_level_summary", {})
    lines.append("## 2. 事件评估")
    lines.append("")
    lines.append(
        f"- 精确率 `{global_metrics.get('precision', 0.0):.4f}`，召回率 `{global_metrics.get('recall', 0.0):.4f}`，"
        f"F1 `{global_metrics.get('f1', 0.0):.4f}`。"
    )
    lines.append(
        f"- 事件命中率 `{event_summary.get('hit_rate_events', 0.0):.4f}`，"
        f"假警率 `{event_summary.get('false_alert_rate', 0.0):.4f}`。"
    )
    lines.append("")

    portfolio = backtest_metrics.get("portfolio", {})
    excess = backtest_metrics.get("excess_vs_benchmark", {})
    lines.append("## 3. 回测指标")
    lines.append("")
    lines.append(
        f"- 组合年化 `{portfolio.get('annualized_return', 0.0):.2%}`，Sharpe `{portfolio.get('sharpe', 0.0):.4f}`，"
        f"最大回撤 `{portfolio.get('max_drawdown', 0.0):.2%}`。"
    )
    lines.append(
        f"- 超额年化 `{excess.get('annualized_return', 0.0):.2%}`，超额 Sharpe `{excess.get('sharpe', 0.0):.4f}`。"
    )
    if alpha_capture.get("primary_alpha_capture_ratio") is not None:
        lines.append(
            f"- 相对随机下界与 oracle return 上界的 alpha capture 为 "
            f"`{alpha_capture['primary_alpha_capture_ratio']:.4f}`，评级 `{three_way.get('grade', 'N/A')}`。"
        )
    lines.append("")

    lines.append("## 4. 分层分析")
    lines.append("")
    if not yearly_df.empty:
        best_year = yearly_df.sort_values("excess_annualized_return", ascending=False).iloc[0]
        worst_year = yearly_df.sort_values("excess_annualized_return", ascending=True).iloc[0]
        lines.append(
            f"- 年度维度上，超额收益最好的年份是 `{int(best_year['year'])}` ({best_year['excess_annualized_return']:.2%})，"
            f"最弱年份是 `{int(worst_year['year'])}` ({worst_year['excess_annualized_return']:.2%})。"
        )
    if not history_df.empty:
        for _, row in history_df.iterrows():
            lines.append(
                f"- `{row['history_bucket']}`: 平均单笔收益 `{row['mean_trade_return']:.2%}`，命中率 `{row['hit_rate']:.2%}`，交易数 `{int(row['n_trades'])}`。"
            )
    if not liquidity_df.empty:
        best_liq = liquidity_df.sort_values("mean_trade_return", ascending=False).iloc[0]
        worst_liq = liquidity_df.sort_values("mean_trade_return", ascending=True).iloc[0]
        lines.append(
            f"- 流动性最好桶 `{best_liq['liquidity_bucket']}` 的平均单笔收益为 `{best_liq['mean_trade_return']:.2%}`；"
            f"最弱桶为 `{worst_liq['liquidity_bucket']}` ({worst_liq['mean_trade_return']:.2%})。"
        )
    lines.append("")

    lines.append("## 5. 图表与产物")
    lines.append("")
    lines.append(f"- NAV 图: `{charts_dir / 'nav_vs_benchmark.png'}`")
    lines.append(f"- 年度超额收益图: `{charts_dir / 'yearly_excess_return.png'}`")
    lines.append(f"- 流动性收益图: `{charts_dir / 'liquidity_trade_return.png'}`")
    lines.append(f"- 分组命中率图: `{charts_dir / 'history_hit_rate.png'}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standardized end-to-end experiment pipeline for one version.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--version_registry", default=str(DEFAULT_VERSION_REGISTRY))
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--cfg", default="configs/config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_predict", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--force_backtest", action="store_true")
    args = parser.parse_args()

    registry_path = PROJECT_ROOT / args.version_registry
    specs = load_version_specs(registry_path)
    if args.version not in specs:
        raise SystemExit(f"Unknown version: {args.version}")
    spec = specs[args.version]

    base_model_cfg = load_yaml(PROJECT_ROOT / args.model_cfg)
    base_backtest_cfg = load_yaml(PROJECT_ROOT / args.backtest_cfg)
    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)
    run_dir = paths.outputs_dir / args.run_id
    std_dir = run_dir / "standardized"
    logs_dir = std_dir / "logs"
    configs_dir = std_dir / "configs"
    analysis_dir = std_dir / "analysis"
    charts_dir = std_dir / "charts"
    ensure_dir(std_dir)
    ensure_dir(logs_dir)
    ensure_dir(configs_dir)
    ensure_dir(analysis_dir)
    ensure_dir(charts_dir)

    model_cfg_path = configs_dir / "model.yaml"
    backtest_cfg_path = configs_dir / "backtest.yaml"
    materialize_version_configs(
        version=args.version,
        base_model_cfg=base_model_cfg,
        base_backtest_cfg=base_backtest_cfg,
        out_model_path=model_cfg_path,
        out_backtest_path=backtest_cfg_path,
        registry_path=registry_path,
    )

    model_artifact = paths.models_dir / f"xgb_{args.run_id}.joblib"
    train_cmd = ["python", "scripts/run_train.py", "--run_id", args.run_id, "--model_cfg", str(model_cfg_path)]
    if spec.data_suffix:
        train_cmd += ["--data_suffix", spec.data_suffix]
    if spec.tradability_weight:
        train_cmd += ["--tradability_weight"]
    if spec.runner_kind != "random" and (args.force_train or not model_artifact.exists()):
        code, _ = _run_cmd(train_cmd, logs_dir / "train.log")
        if code != 0:
            raise SystemExit("train failed")

    pred_targets = ["val", args.split] if args.split != "val" else ["val"]
    for split in pred_targets:
        pred_path = run_dir / "preds" / f"{split}_preds.parquet"
        pred_csv = pred_path.with_suffix(".csv")
        exists = pred_path.exists() or pred_csv.exists()
        if exists and not args.force_predict:
            continue
        if spec.runner_kind == "random":
            cmd = ["python", "scripts/run_random_predict.py", "--run_id", args.run_id, "--split", split, "--seed", str(spec.random_seed)]
        else:
            cmd = ["python", "scripts/run_predict.py", "--run_id", args.run_id, "--split", split]
            if spec.data_suffix:
                cmd += ["--data_suffix", spec.data_suffix]
        code, _ = _run_cmd(cmd, logs_dir / f"predict_{split}.log")
        if code != 0:
            raise SystemExit(f"predict failed for split={split}")

    eval_summary_path = run_dir / "eval_stratified" / "summary.json"
    if args.force_eval or not eval_summary_path.exists():
        code, _ = _run_cmd(
            ["python", "scripts/run_eval.py", "--cfg", args.cfg, "--paths", args.paths, "--model_cfg", str(model_cfg_path), "--run_id", args.run_id, "--split", args.split],
            logs_dir / f"eval_{args.split}.log",
        )
        if code != 0:
            raise SystemExit("eval failed")

    backtest_summary_path = run_dir / "backtest" / "summary.json"
    if args.force_backtest or not backtest_summary_path.exists():
        code, _ = _run_cmd(
            ["python", "scripts/run_backtest.py", "--paths", args.paths, "--backtest_cfg", str(backtest_cfg_path), "--run_id", args.run_id, "--split", args.split],
            logs_dir / f"backtest_{args.split}.log",
        )
        if code != 0:
            raise SystemExit("backtest failed")

    val_metrics = _pred_metrics(_resolve_preds_path(run_dir, "val"))
    test_metrics = _pred_metrics(_resolve_preds_path(run_dir, args.split))
    event_summary = _load_json(eval_summary_path)
    backtest_summary = _load_json(backtest_summary_path)

    training_summary: dict[str, Any] = {
        "val_auc": val_metrics["auc"],
        "val_aucpr": val_metrics["aucpr"],
        "test_auc": test_metrics["auc"],
        "test_aucpr": test_metrics["aucpr"],
        "val_rows": val_metrics["rows"],
        "test_rows": test_metrics["rows"],
    }
    if spec.runner_kind != "random" and model_artifact.exists():
        art = load_artifacts(model_artifact)
        training_summary["best_iteration"] = getattr(art.booster, "best_iteration", None)

    split_df = pd.read_parquet(paths.processed_dir / f"{args.split}.parquet")
    trades_df = pd.read_csv(run_dir / "backtest" / "trades.csv")
    daily_df = pd.read_csv(run_dir / "backtest" / "daily_portfolio.csv")
    trades_ctx = enrich_trades_with_split_context(trades_df, split_df)
    trades_ctx, _ = add_liquidity_bucket(trades_ctx, value_col="turnover_5d", q=4)

    yearly_df = compute_yearly_backtest_metrics(daily_df, version=spec.version)
    history_df = compute_trade_group_metrics(trades_ctx, "history_bucket", version=spec.version)
    liquidity_df = compute_trade_group_metrics(trades_ctx, "liquidity_bucket", version=spec.version)

    yearly_df.to_csv(analysis_dir / "yearly_backtest_metrics.csv", index=False)
    history_df.to_csv(analysis_dir / "history_trade_metrics.csv", index=False)
    liquidity_df.to_csv(analysis_dir / "liquidity_trade_metrics.csv", index=False)

    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    plot_nav_vs_benchmark(daily_df, charts_dir / "nav_vs_benchmark.png", title=f"{spec.version} NAV vs Benchmark")
    plot_grouped_bar(
        yearly_df,
        x_col="year",
        y_col="excess_annualized_return",
        hue_col="version",
        path=charts_dir / "yearly_excess_return.png",
        title="Yearly Excess Annualized Return",
        ylabel="Excess Annualized Return",
    )
    plot_grouped_bar(
        liquidity_df,
        x_col="liquidity_bucket",
        y_col="mean_trade_return",
        hue_col="version",
        path=charts_dir / "liquidity_trade_return.png",
        title="Trade Return by Liquidity Bucket",
        ylabel="Mean Trade Return",
    )
    plot_grouped_bar(
        history_df,
        x_col="history_bucket",
        y_col="hit_rate",
        hue_col="version",
        path=charts_dir / "history_hit_rate.png",
        title="Hit Rate by Dividend History",
        ylabel="Hit Rate",
    )

    metrics = {
        "version": spec.to_manifest(),
        "training": training_summary,
        "event_eval": event_summary,
        "backtest": backtest_summary,
    }
    (std_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _build_report(run_dir, spec, metrics, yearly_df, history_df, liquidity_df)
    print(std_dir)


if __name__ == "__main__":
    main()
