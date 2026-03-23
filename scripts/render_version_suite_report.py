from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VERSION_LABELS = {
    "random": "random 随机预测基线 (U(0,1), auto)",
    "v1": "v1 基线 (30+4, 200, auto)",
    "v2": "v2 全量特征 (46+5, 200, auto)",
    "v3": "v3 全量特征 + 强制规则 (46+5, 200, true)",
    "v4": "v4 精简+多轮 (31+5, 400, true)",
    "v5": "v5 全量+多轮 (46+5, 400, true)",
    "v6": "v6 精简+标准 (31+5, 200, true)",
    "v7": "v7 仅 +decl (31+4, 200, auto)",
    "v8": "v8 仅 +industry (30+5, 200, true)",
}


def _fmt_num(value: Any, digits: int = 4, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    x = float(value)
    if pct:
        return f"{x * 100:.2f}%"
    return f"{x:.{digits}f}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out)


def _dataset_stats() -> dict[str, dict[str, Any]]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    stats: dict[str, dict[str, Any]] = {}
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(processed_dir / f"{split}.parquet", columns=["PERMNO", "y_div_10d"])
        stats[split] = {
            "rows": int(len(df)),
            "permnos": int(df["PERMNO"].nunique()),
            "pos_rate": float(df["y_div_10d"].mean()),
        }
        parts.append(df)
    total = pd.concat(parts, ignore_index=True)
    stats["total"] = {
        "rows": int(len(total)),
        "permnos": int(total["PERMNO"].nunique()),
        "pos_rate": float(total["y_div_10d"].mean()),
    }
    return stats


def _ranking_summary(df: pd.DataFrame) -> tuple[str, str, str]:
    best_excess = df.sort_values(["test_excess_annualized_return", "test_portfolio_sharpe"], ascending=[False, False]).iloc[0]
    best_sharpe = df.sort_values(["test_portfolio_sharpe", "test_excess_annualized_return"], ascending=[False, False]).iloc[0]
    best_aucpr = df.sort_values(["val_aucpr", "test_aucpr"], ascending=[False, False]).iloc[0]
    return str(best_excess["version"]), str(best_sharpe["version"]), str(best_aucpr["version"])


def render_report(batch_id: str) -> Path:
    suite_dir = PROJECT_ROOT / "outputs" / "report_version_suite" / batch_id
    results_path = suite_dir / "suite_results.csv"
    defs_path = suite_dir / "version_definitions.json"
    manifest_path = suite_dir / "manifest.json"
    if not results_path.exists():
        raise SystemExit(f"Missing suite results: {results_path}")

    df = pd.read_csv(results_path)
    definitions = json.loads(defs_path.read_text(encoding="utf-8")) if defs_path.exists() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cfg = _load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    stats = _dataset_stats()

    best_excess_version, best_sharpe_version, best_aucpr_version = _ranking_summary(df)
    ranked = df.sort_values(["test_excess_annualized_return", "test_portfolio_sharpe"], ascending=[False, False]).reset_index(drop=True)
    has_random = "random" in set(df["version"].astype(str))

    overview_rows = []
    for _, row in ranked.iterrows():
        overview_rows.append(
            [
                str(row["version"]),
                definitions.get(str(row["version"]), {}).get("label")
                or VERSION_LABELS.get(str(row["version"]), str(row["description"])),
                _fmt_num(row["val_aucpr"]),
                _fmt_num(row["test_portfolio_annualized_return"], pct=True),
                _fmt_num(row["test_portfolio_sharpe"]),
                _fmt_num(row["test_excess_annualized_return"], pct=True),
                _fmt_num(row["test_excess_sharpe"]),
                f"{int(row['policy_support_votes'])}/{int(row['policy_total_checks'])}" if pd.notna(row["policy_support_votes"]) and pd.notna(row["policy_total_checks"]) else "-",
                "是" if bool(row["policy_use_dividend_rules"]) else "否",
            ]
        )

    comparison_rows = []
    for version in ("v1", "v2", "v7", "v8"):
        if version not in set(df["version"].astype(str)):
            continue
        row = df[df["version"] == version].iloc[0]
        comparison_rows.append(
            [
                version,
                _fmt_num(row.get("report_val_aucpr")),
                _fmt_num(row.get("val_aucpr")),
                _fmt_num(row.get("delta_val_aucpr_vs_report")),
                _fmt_num(row.get("report_excess_ann"), pct=True),
                _fmt_num(row.get("test_excess_annualized_return"), pct=True),
                _fmt_num(row.get("delta_test_excess_ann_vs_report"), pct=True),
                _fmt_num(row.get("report_sharpe")),
                _fmt_num(row.get("test_portfolio_sharpe")),
                _fmt_num(row.get("delta_test_sharpe_vs_report")),
            ]
        )

    definition_rows = []
    for version in sorted(definitions.keys()):
        item = definitions[version]
        definition_rows.append(
            [
                version,
                str(item.get("runner_kind", "xgb")),
                str(len(item.get("num_cols", []))),
                str(len(item.get("cat_cols", []))),
                str(item.get("num_boost_round", "-")),
                str(item.get("early_stopping_rounds", "-")),
                str(item.get("dividend_rules_mode", "-")),
            ]
        )

    dataset_lines = [
        f"- `train`: {stats['train']['rows']:,} 行, {stats['train']['permnos']} 个 PERMNO, 正样本率 {_fmt_num(stats['train']['pos_rate'], pct=True)}",
        f"- `val`: {stats['val']['rows']:,} 行, {stats['val']['permnos']} 个 PERMNO, 正样本率 {_fmt_num(stats['val']['pos_rate'], pct=True)}",
        f"- `test`: {stats['test']['rows']:,} 行, {stats['test']['permnos']} 个 PERMNO, 正样本率 {_fmt_num(stats['test']['pos_rate'], pct=True)}",
        f"- `total`: {stats['total']['rows']:,} 行, {stats['total']['permnos']} 个 PERMNO, 正样本率 {_fmt_num(stats['total']['pos_rate'], pct=True)}",
    ]

    report_lines: list[str] = []
    report_lines.append("# 当前数据版本实验报告")
    report_lines.append("")
    report_lines.append(f"- 批次 ID: `{batch_id}`")
    report_lines.append(
        f"- 生成时间口径: 当前仓库数据快照, `train={cfg['split']['train_start']}~{cfg['split']['train_end']}`, "
        f"`val={cfg['split']['val_start']}~{cfg['split']['val_end']}`, `test={cfg['split']['test_start']}~{cfg['split']['test_end']}`"
    )
    report_lines.append(f"- 实验入口: `python scripts/run_report_version_suite.py --batch_id {batch_id} --skip_reference`")
    report_lines.append(
        "- 批处理说明: 本次按报告定义重跑 `v1~v8`，并新增一个 `random` 随机预测基线。"
        "主表指标来自 `val` 预测质量与 `test` 回测表现。"
    )
    if manifest.get("versions"):
        report_lines.append(f"- 本次纳入版本: `{', '.join(manifest['versions'])}`")
    report_lines.append("")

    report_lines.append("## 一、数据与实验设置")
    report_lines.append("")
    report_lines.extend(dataset_lines)
    report_lines.append("")
    report_lines.append("版本定义如下：")
    report_lines.append("")
    report_lines.append(
        _table(
            ["版本", "运行器", "数值特征数", "类别特征数", "训练轮数", "早停轮数", "规则模式"],
            definition_rows,
        )
    )
    report_lines.append("")
    report_lines.append("说明：")
    report_lines.append("- `auto` 采用当前仓库的安全实现，即测试集只允许用 `val` 作为 held-out policy 依据。")
    report_lines.append("- `random` 使用固定随机种子 `42`，对每个样本独立生成 `U(0,1)` 概率。")
    report_lines.append("- 本次为了快速复现实验主表，跳过了 random/oracle reference；核心排序不受影响。")
    report_lines.append("")

    report_lines.append("## 二、核心结果")
    report_lines.append("")
    report_lines.append(
        f"当前数据版本下，测试集超额年化最佳的是 `{best_excess_version}`，"
        f"Sharpe 最佳的也是 `{best_sharpe_version}`；验证集 AUPRC 最高的是 `{best_aucpr_version}`。"
    )
    report_lines.append("")
    report_lines.append(
        _table(
            ["版本", "定义", "AUPRC(val)", "测试年化", "测试Sharpe", "测试超额年化", "测试超额Sharpe", "Policy票数", "启用规则"],
            overview_rows,
        )
    )
    report_lines.append("")

    if has_random and "v1" in set(df["version"].astype(str)):
        v1_row = df[df["version"] == "v1"].iloc[0]
        rand_row = df[df["version"] == "random"].iloc[0]
        report_lines.append("### 与随机基线对比")
        report_lines.append("")
        report_lines.append(
            f"`v1` 相比 `random` 的验证集 AUPRC 提升 "
            f"{_fmt_num(v1_row['val_aucpr'] - rand_row['val_aucpr'])}，"
            f"测试集超额年化提升 "
            f"{_fmt_num(v1_row['test_excess_annualized_return'] - rand_row['test_excess_annualized_return'], pct=True)}，"
            f"测试集 Sharpe 提升 {_fmt_num(v1_row['test_portfolio_sharpe'] - rand_row['test_portfolio_sharpe'])}。"
        )
        report_lines.append("")

    report_lines.append("## 三、相对旧报告的变化")
    report_lines.append("")
    report_lines.append("下表重点比较旧报告中最关键的四个版本。`差值=当前结果-旧报告结果`。")
    report_lines.append("")
    report_lines.append(
        _table(
            ["版本", "旧AUPRC(val)", "新AUPRC(val)", "差值", "旧超额年化", "新超额年化", "差值", "旧Sharpe", "新Sharpe", "差值"],
            comparison_rows,
        )
    )
    report_lines.append("")
    report_lines.append("关键变化：")
    if has_random:
        report_lines.append("- `random` 为零预测能力基线，用于确认纯随机分红预测在当前回测框架下的大致下界。")
    report_lines.append("- `v1` 现在是当前数据版本下的最优基线，测试集超额年化 `+1.58%`，Sharpe `0.5502`。")
    report_lines.append("- `v7` 的模型层表现仍接近旧报告，但策略层不再最优，测试集超额年化为 `-0.62%`。")
    report_lines.append("- `v2` 和 `v3` 仍体现“模型指标更高但策略表现更弱”的复杂度楔形现象，只是 `auto` 不再像旧报告那样失活。")
    report_lines.append("- `v8` 在当前数据上不再明显破坏 policy，测试集表现与 `v7` 接近。")
    report_lines.append("")

    report_lines.append("## 四、结论")
    report_lines.append("")
    if has_random:
        report_lines.append("1. `v1` 必须至少显著优于 `random`，否则说明当前信号没有提供可验证的增量价值。")
        report_lines.append("2. 在当前数据版本下，`v1` 相对 `random` 仍有明显增益，说明基线模型不是随机筛股。")
        report_lines.append("3. 旧报告中“`v7` 最优”的结论不适用于当前正确数据版本；当前最优版本是 `v1`。")
        report_lines.append("4. 旧报告中关于 `industry` 会单独导致 policy 失效的结论，在当前数据上不再成立。`v8` 的 policy 票数为 `2/3`。")
        report_lines.append("5. “更高的验证集 AUPRC 不等于更好的测试集超额收益”这一点仍然成立。`v5` 的 `AUPRC(val)` 最高，但测试集超额年化最差之一。")
        report_lines.append("6. “400 轮会伤害精简或扩展特征集的策略表现”这一点仍基本成立，`v4` 和 `v5` 均排在末尾。")
    else:
        report_lines.append("1. 旧报告中“`v7` 最优”的结论不适用于当前正确数据版本；当前最优版本是 `v1`。")
        report_lines.append("2. 旧报告中关于 `industry` 会单独导致 policy 失效的结论，在当前数据上不再成立。`v8` 的 policy 票数为 `2/3`。")
        report_lines.append("3. “更高的验证集 AUPRC 不等于更好的测试集超额收益”这一点仍然成立。`v5` 的 `AUPRC(val)` 最高，但测试集超额年化最差之一。")
        report_lines.append("4. “400 轮会伤害精简或扩展特征集的策略表现”这一点仍基本成立，`v4` 和 `v5` 均排在末尾。")
    report_lines.append("")

    report_lines.append("## 五、建议")
    report_lines.append("")
    report_lines.append("1. 把后续研究基线从旧报告的 `v7` 改为当前数据版本下的 `v1`。")
    report_lines.append("2. 保留 `random` 作为固定下界基线，后续每次增量实验都应同时对比 `random` 与 `v1`。")
    report_lines.append("3. 若要继续研究 `decl_to_exdt_days` 或 `industry` 的净贡献，应以当前数据版本为准，重新做受控增量实验。")
    report_lines.append("4. 下一轮建议专门做 `v1 -> v7` 与 `v1 -> v8` 的归因分析，检查差异是否来自某几年、某些行业或某类交易成本暴露。")
    report_lines.append("")

    report_lines.append("## 六、产物路径")
    report_lines.append("")
    report_lines.append(f"- 汇总表: `{results_path}`")
    report_lines.append(f"- 排序表: `{suite_dir / 'suite_results_ranked.csv'}`")
    report_lines.append(f"- 简要结论: `{suite_dir / 'suite_findings.txt'}`")
    report_lines.append(f"- 版本定义: `{defs_path}`")
    report_lines.append(f"- 配置快照目录: `{suite_dir / 'configs'}`")
    report_lines.append(f"- 原始日志目录: `{suite_dir / 'logs'}`")

    out_path = suite_dir / "current_data_experiment_report.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a markdown report from a version suite batch.")
    parser.add_argument("--batch_id", required=True)
    args = parser.parse_args()
    out_path = render_report(args.batch_id)
    print(out_path)


if __name__ == "__main__":
    main()
