from __future__ import annotations

import unittest

import pandas as pd

from src.backtest.benchmark import build_equal_weight_benchmark
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import enrich_daily_report, summarize_backtest
from src.backtest.signal import add_forward_returns, build_backtest_panel, build_daily_candidates, compute_stable_gap_cv_threshold


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        rows = []
        probs_a = [0.90, 0.88, 0.20, 0.10, 0.10, 0.10]
        probs_b = [0.80, 0.78, 0.77, 0.20, 0.10, 0.10]
        rets_a = [0.00, 0.10, 0.10, 0.00, 0.00, 0.00]
        rets_b = [0.00, 0.01, 0.01, 0.01, 0.00, 0.00]
        rets_c = [0.00, 0.02, -0.01, 0.01, 0.00, 0.00]
        for i, dt in enumerate(dates):
            rows.extend(
                [
                    {
                        "DlyCalDt": dt,
                        "PERMNO": 1,
                        "DlyRet": rets_a[i],
                        "DlyPrc": 10.0,
                        "turnover_5d": 1.0,
                        "SICCD": 1111,
                        "industry": "11",
                        "has_div_history": 1,
                        "div_count_exp": 5,
                        "gap_cv_exp": 0.2,
                        "gap_med_exp": 90.0,
                        "days_since_last_div": 88.0,
                        "z_to_med_exp": -0.2,
                        "y_div_10d": 1,
                    },
                    {
                        "DlyCalDt": dt,
                        "PERMNO": 2,
                        "DlyRet": rets_b[i],
                        "DlyPrc": 11.0,
                        "turnover_5d": 0.9,
                        "SICCD": 1112,
                        "industry": "11",
                        "has_div_history": 1,
                        "div_count_exp": 3,
                        "gap_cv_exp": 0.5,
                        "gap_med_exp": 92.0,
                        "days_since_last_div": 95.0,
                        "z_to_med_exp": 0.4,
                        "y_div_10d": 0,
                    },
                    {
                        "DlyCalDt": dt,
                        "PERMNO": 3,
                        "DlyRet": rets_c[i],
                        "DlyPrc": 12.0,
                        "turnover_5d": 0.05,
                        "SICCD": 2010,
                        "industry": "20",
                        "has_div_history": 0,
                        "div_count_exp": 1,
                        "gap_cv_exp": 1.0,
                        "gap_med_exp": 100.0,
                        "days_since_last_div": 120.0,
                        "z_to_med_exp": 2.0,
                        "y_div_10d": 0,
                    },
                ]
            )
        self.split_df = pd.DataFrame(rows)
        pred_rows = []
        for i, dt in enumerate(dates):
            pred_rows.extend(
                [
                    {"date": dt, "permno": 1, "prob": probs_a[i]},
                    {"date": dt, "permno": 2, "prob": probs_b[i]},
                    {"date": dt, "permno": 3, "prob": 0.95},
                ]
            )
        self.preds_df = pd.DataFrame(pred_rows)

    def test_build_candidates_filters_and_industry_cap(self) -> None:
        panel = add_forward_returns(build_backtest_panel(self.preds_df, self.split_df), horizons=(1, 2, 3))
        threshold = compute_stable_gap_cv_threshold(self.split_df, stable_div_count_min=4, quantile=0.5)
        candidates = build_daily_candidates(
            panel=panel,
            top_k=2,
            stable_gap_cv_threshold=threshold,
            turnover_quantile_min=0.2,
            exclude_div_count_le=1,
            min_price=3.0,
            stable_div_count_min=4,
            stable_prob_threshold=0.4,
            regular_prob_threshold=0.75,
            max_industry_weight=0.5,
            use_dividend_rules=True,
        )
        day1 = candidates[candidates["date"] == pd.Timestamp("2024-01-01")]
        self.assertEqual(day1["permno"].tolist(), [1])
        self.assertTrue((candidates["permno"] != 3).all())

    def test_portfolio_respects_next_day_entry_and_holding(self) -> None:
        panel = add_forward_returns(build_backtest_panel(self.preds_df, self.split_df), horizons=(1, 2, 3))
        threshold = compute_stable_gap_cv_threshold(self.split_df, stable_div_count_min=4, quantile=0.5)
        candidates = build_daily_candidates(
            panel=panel,
            top_k=1,
            stable_gap_cv_threshold=threshold,
            turnover_quantile_min=0.2,
            exclude_div_count_le=1,
            min_price=3.0,
            stable_div_count_min=4,
            stable_prob_threshold=0.4,
            regular_prob_threshold=0.75,
            max_industry_weight=1.0,
            use_dividend_rules=True,
        )
        daily_df, trades_df, positions_df = simulate_portfolio(
            panel=panel,
            candidates=candidates,
            top_k=1,
            holding_td=2,
            cooldown_td=0,
            cost_bps_one_way=0.0,
            max_industry_weight=1.0,
        )
        self.assertEqual(len(trades_df), 2)
        first_trade = trades_df.iloc[0]
        self.assertEqual(pd.Timestamp(first_trade["signal_date"]), pd.Timestamp("2024-01-01"))
        self.assertEqual(pd.Timestamp(first_trade["entry_date"]), pd.Timestamp("2024-01-02"))
        self.assertEqual(pd.Timestamp(first_trade["exit_date"]), pd.Timestamp("2024-01-03"))
        first_day = daily_df[daily_df["date"] == pd.Timestamp("2024-01-01")].iloc[0]
        self.assertAlmostEqual(float(first_day["portfolio_ret"]), 0.0, places=8)
        second_day = daily_df[daily_df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        self.assertAlmostEqual(float(second_day["portfolio_ret"]), 0.10, places=8)
        self.assertEqual(
            positions_df[positions_df["trade_id"] == first_trade["trade_id"]]["date"].tolist(),
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        )

    def test_benchmark_and_summary_outputs(self) -> None:
        panel = add_forward_returns(build_backtest_panel(self.preds_df, self.split_df), horizons=(1, 2, 3))
        benchmark = build_equal_weight_benchmark(panel, min_price=3.0)
        day2 = benchmark[benchmark["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        self.assertAlmostEqual(float(day2["benchmark_ret"]), (0.10 + 0.01 + 0.02) / 3.0, places=8)

        threshold = compute_stable_gap_cv_threshold(self.split_df, stable_div_count_min=4, quantile=0.5)
        candidates = build_daily_candidates(
            panel=panel,
            top_k=1,
            stable_gap_cv_threshold=threshold,
            turnover_quantile_min=0.2,
            exclude_div_count_le=1,
            min_price=3.0,
            stable_div_count_min=4,
            stable_prob_threshold=0.4,
            regular_prob_threshold=0.75,
            max_industry_weight=1.0,
            use_dividend_rules=True,
        )
        daily_df, trades_df, _ = simulate_portfolio(
            panel=panel,
            candidates=candidates,
            top_k=1,
            holding_td=2,
            cooldown_td=0,
            cost_bps_one_way=10.0,
            max_industry_weight=1.0,
        )
        daily_df = enrich_daily_report(daily_df, benchmark)
        summary = summarize_backtest(
            daily_df,
            trades_df,
            {"mode": "auto", "use_dividend_rules": True, "support_votes": 3, "total_checks": 3, "checks": {}},
        )
        self.assertIn("portfolio", summary)
        self.assertIn("excess_vs_benchmark", summary)
        self.assertEqual(summary["trades"]["n_trades"], len(trades_df))


if __name__ == "__main__":
    unittest.main()
