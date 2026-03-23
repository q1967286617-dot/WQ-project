from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.backtest_analysis import (
    add_liquidity_bucket,
    compute_trade_group_metrics,
    compute_yearly_backtest_metrics,
    enrich_trades_with_split_context,
    load_run_backtest_frames,
)
from src.analysis.plots import plot_grouped_bar
from src.utils.paths import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pairwise_gap(df: pd.DataFrame, group_col: str, metric_cols: list[str], base_version: str, other_versions: list[str]) -> pd.DataFrame:
    pivot = df.pivot(index=group_col, columns="version", values=metric_cols)
    pivot.columns = [f"{metric}_{version}" for metric, version in pivot.columns]
    out = pivot.reset_index()
    for other in other_versions:
        for metric in metric_cols:
            base_col = f"{metric}_{base_version}"
            other_col = f"{metric}_{other}"
            if base_col in out.columns and other_col in out.columns:
                out[f"{metric}_{base_version}_minus_{other}"] = out[base_col] - out[other_col]
    return out


def _top_industries(trades_df: pd.DataFrame, top_n: int) -> list[str]:
    counts = trades_df["industry"].astype("string").fillna("unknown").value_counts()
    return counts.head(top_n).index.astype(str).tolist()


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _build_report(
    out_dir: Path,
    versions: list[str],
    by_year_compare: pd.DataFrame,
    by_history_compare: pd.DataFrame,
    by_liquidity_compare: pd.DataFrame,
    by_industry_compare: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# v1 vs v7/v8 分层翻转分析")
    lines.append("")
    lines.append(f"- 版本范围: `{', '.join(versions)}`")
    lines.append(f"- 输出目录: `{out_dir}`")
    lines.append("")

    if not by_year_compare.empty:
        year_rows = by_year_compare.sort_values("year")
        lines.append("## 年份维度")
        lines.append("")
        for other in [v for v in versions if v != "v1"]:
            gap_col = f"excess_annualized_return_v1_minus_{other}"
            if gap_col not in year_rows.columns:
                continue
            win_years = int((year_rows[gap_col] > 0).sum())
            total_years = int(year_rows[gap_col].notna().sum())
            best = year_rows.loc[year_rows[gap_col].idxmax()]
            worst = year_rows.loc[year_rows[gap_col].idxmin()]
            lines.append(
                f"- `v1` 相比 `{other}` 在 {win_years}/{total_years} 个年份里拥有更高的超额年化收益；"
                f"优势最大年份是 `{int(best['year'])}` ({_fmt_pct(best[gap_col])})，"
                f"劣势最大年份是 `{int(worst['year'])}` ({_fmt_pct(worst[gap_col])})。"
            )
        lines.append("")

    if not by_history_compare.empty:
        lines.append("## 历史分组")
        lines.append("")
        for _, row in by_history_compare.sort_values("history_bucket").iterrows():
            parts = [
                f"`{row['history_bucket']}`",
                f"v1 收益={_fmt_pct(row.get('mean_trade_return_v1'))}",
                f"v1 命中率={_fmt_pct(row.get('hit_rate_v1'))}",
            ]
            for other in [v for v in versions if v != "v1"]:
                parts.append(
                    f"v1-{other} 收益差={_fmt_pct(row.get(f'mean_trade_return_v1_minus_{other}'))}"
                )
                parts.append(
                    f"v1-{other} 命中率差={_fmt_pct(row.get(f'hit_rate_v1_minus_{other}'))}"
                )
            lines.append("- " + "，".join(parts) + "。")
        lines.append("")

    if not by_liquidity_compare.empty:
        lines.append("## 流动性分层")
        lines.append("")
        sorted_liq = by_liquidity_compare.sort_values("liquidity_bucket")
        for _, row in sorted_liq.iterrows():
            parts = [f"`{row['liquidity_bucket']}`"]
            for other in [v for v in versions if v != "v1"]:
                parts.append(
                    f"v1-{other} 收益差={_fmt_pct(row.get(f'mean_trade_return_v1_minus_{other}'))}"
                )
                parts.append(
                    f"v1-{other} 命中率差={_fmt_pct(row.get(f'hit_rate_v1_minus_{other}'))}"
                )
            lines.append("- " + "，".join(parts) + "。")
        lines.append("")

    if not by_industry_compare.empty:
        lines.append("## 行业层面")
        lines.append("")
        for other in [v for v in versions if v != "v1"]:
            gap_col = f"mean_trade_return_v1_minus_{other}"
            if gap_col not in by_industry_compare.columns:
                continue
            top = by_industry_compare.sort_values(gap_col, ascending=False).head(3)
            bottom = by_industry_compare.sort_values(gap_col, ascending=True).head(3)
            lines.append(f"- `v1` 相比 `{other}` 的正向贡献行业: " + "，".join(
                f"{r['industry']} ({_fmt_pct(r[gap_col])})" for _, r in top.iterrows()
            ) + "。")
            lines.append(f"- `v1` 相比 `{other}` 的负向贡献行业: " + "，".join(
                f"{r['industry']} ({_fmt_pct(r[gap_col])})" for _, r in bottom.iterrows()
            ) + "。")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze why v1 outperforms v7/v8 on the current data snapshot.")
    parser.add_argument("--batch_id", default="report_version_suite_current_data_20260323")
    parser.add_argument("--versions", nargs="*", default=["v1", "v7", "v8"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_industries", type=int, default=12)
    args = parser.parse_args()

    batch_dir = PROJECT_ROOT / "outputs" / "report_version_suite" / args.batch_id
    suite_results = pd.read_csv(batch_dir / "suite_results.csv")
    split_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / f"{args.split}.parquet")

    out_dir = PROJECT_ROOT / "outputs" / "analysis" / f"{args.batch_id}_v1_flip"
    charts_dir = out_dir / "charts"
    ensure_dir(out_dir)
    ensure_dir(charts_dir)

    yearly_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for version in args.versions:
        row = suite_results[suite_results["version"] == version]
        if row.empty:
            raise SystemExit(f"Missing version in suite results: {version}")
        run_id = str(row.iloc[0]["run_id"])
        daily_df, trades_df, _ = load_run_backtest_frames(PROJECT_ROOT, run_id)
        trades_ctx = enrich_trades_with_split_context(trades_df, split_df)

        yearly_parts.append(compute_yearly_backtest_metrics(daily_df, version=version))
        trades_ctx["version"] = version
        trade_parts.append(trades_ctx)

    yearly_df = pd.concat(yearly_parts, ignore_index=True)
    trades_all = pd.concat(trade_parts, ignore_index=True)
    trades_all, liq_edges = add_liquidity_bucket(trades_all, value_col="turnover_5d", q=4)

    history_df = []
    liquidity_df = []
    by_year_trade_df = []
    for version, g in trades_all.groupby("version", sort=False):
        history_df.append(compute_trade_group_metrics(g, "history_bucket", version=version))
        liquidity_df.append(compute_trade_group_metrics(g, "liquidity_bucket", version=version))
        by_year_trade_df.append(compute_trade_group_metrics(g.rename(columns={"signal_year": "year"}), "year", version=version))

    history_df = pd.concat(history_df, ignore_index=True)
    liquidity_df = pd.concat(liquidity_df, ignore_index=True)
    by_year_trade_df = pd.concat(by_year_trade_df, ignore_index=True)

    top_industries = _top_industries(trades_all, top_n=args.top_industries)
    industry_df = []
    for version, g in trades_all.groupby("version", sort=False):
        filtered = g[g["industry"].astype("string").isin(top_industries)].copy()
        industry_df.append(compute_trade_group_metrics(filtered, "industry", version=version))
    industry_df = pd.concat(industry_df, ignore_index=True)

    by_year_compare = _pairwise_gap(
        yearly_df[["version", "year", "portfolio_annualized_return", "excess_annualized_return", "portfolio_sharpe", "excess_sharpe"]],
        group_col="year",
        metric_cols=["portfolio_annualized_return", "excess_annualized_return", "portfolio_sharpe", "excess_sharpe"],
        base_version="v1",
        other_versions=[v for v in args.versions if v != "v1"],
    )
    by_history_compare = _pairwise_gap(
        history_df[["version", "history_bucket", "mean_trade_return", "hit_rate", "n_trades"]],
        group_col="history_bucket",
        metric_cols=["mean_trade_return", "hit_rate", "n_trades"],
        base_version="v1",
        other_versions=[v for v in args.versions if v != "v1"],
    )
    by_liquidity_compare = _pairwise_gap(
        liquidity_df[["version", "liquidity_bucket", "mean_trade_return", "hit_rate", "n_trades"]],
        group_col="liquidity_bucket",
        metric_cols=["mean_trade_return", "hit_rate", "n_trades"],
        base_version="v1",
        other_versions=[v for v in args.versions if v != "v1"],
    )
    by_industry_compare = _pairwise_gap(
        industry_df[["version", "industry", "mean_trade_return", "hit_rate", "n_trades"]],
        group_col="industry",
        metric_cols=["mean_trade_return", "hit_rate", "n_trades"],
        base_version="v1",
        other_versions=[v for v in args.versions if v != "v1"],
    )

    yearly_df.to_csv(out_dir / "yearly_backtest_metrics.csv", index=False)
    by_year_trade_df.to_csv(out_dir / "yearly_trade_metrics.csv", index=False)
    history_df.to_csv(out_dir / "history_trade_metrics.csv", index=False)
    liquidity_df.to_csv(out_dir / "liquidity_trade_metrics.csv", index=False)
    industry_df.to_csv(out_dir / "industry_trade_metrics.csv", index=False)
    by_year_compare.to_csv(out_dir / "yearly_backtest_compare.csv", index=False)
    by_history_compare.to_csv(out_dir / "history_trade_compare.csv", index=False)
    by_liquidity_compare.to_csv(out_dir / "liquidity_trade_compare.csv", index=False)
    by_industry_compare.to_csv(out_dir / "industry_trade_compare.csv", index=False)
    pd.DataFrame({"edge": liq_edges}).to_csv(out_dir / "liquidity_bucket_edges.csv", index=False)

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
        path=charts_dir / "liquidity_mean_trade_return.png",
        title="Mean Trade Return by Liquidity Bucket",
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
    plot_grouped_bar(
        industry_df,
        x_col="industry",
        y_col="mean_trade_return",
        hue_col="version",
        path=charts_dir / "industry_mean_trade_return.png",
        title="Mean Trade Return by Industry",
        ylabel="Mean Trade Return",
    )

    _build_report(
        out_dir=out_dir,
        versions=args.versions,
        by_year_compare=by_year_compare,
        by_history_compare=by_history_compare,
        by_liquidity_compare=by_liquidity_compare,
        by_industry_compare=by_industry_compare,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
