from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_backtest import policy_split_names, threshold_reference_split_names
from src.backtest.benchmark import (
    build_equal_weight_benchmark,
    build_non_prob_candidates,
    build_oracle_candidates,
    build_random_prob_candidates,
)
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import enrich_daily_report, summarize_backtest
from src.backtest.signal import add_execution_returns, add_forward_returns, build_backtest_panel, build_daily_candidates, compute_stable_gap_cv_threshold, infer_execution_basis, merge_execution_price_data


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        rows = []
        probs_a = [0.90, 0.88, 0.20, 0.10, 0.10, 0.10]
        probs_b = [0.80, 0.78, 0.77, 0.20, 0.10, 0.10]
        rets_a = [0.00, 0.10, 0.10, 0.00, 0.00, 0.00]
        rets_b = [0.00, 0.01, 0.01, 0.01, 0.00, 0.00]
        rets_c = [0.00, 0.02, -0.01, 0.01, 0.00, 0.00]
        opens_a = [10.0, 10.0, 11.0, 12.1, 12.1, 12.1]
        opens_b = [10.0, 10.0, 10.1, 10.201, 10.30301, 10.30301]
        opens_c = [10.0, 10.2, 10.1, 10.0, 10.0, 10.0]
        price_rows = []
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
            price_rows.extend(
                [
                    {"date": dt, "permno": 1, "DlyOpen": opens_a[i], "DlyClose": opens_a[i], "DlyHigh": opens_a[i], "DlyLow": opens_a[i], "DlyBid": opens_a[i], "DlyAsk": opens_a[i]},
                    {"date": dt, "permno": 2, "DlyOpen": opens_b[i], "DlyClose": opens_b[i], "DlyHigh": opens_b[i], "DlyLow": opens_b[i], "DlyBid": opens_b[i], "DlyAsk": opens_b[i]},
                    {"date": dt, "permno": 3, "DlyOpen": opens_c[i], "DlyClose": opens_c[i], "DlyHigh": opens_c[i], "DlyLow": opens_c[i], "DlyBid": opens_c[i], "DlyAsk": opens_c[i]},
                ]
            )
        self.split_df = pd.DataFrame(rows)
        self.price_df = pd.DataFrame(price_rows)
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

    def _panel(self) -> pd.DataFrame:
        panel = build_backtest_panel(self.preds_df, self.split_df)
        panel = merge_execution_price_data(panel, self.price_df)
        panel = add_execution_returns(panel)
        panel = add_forward_returns(panel, horizons=(1, 2, 3))
        return panel

    def test_split_reference_rules_are_time_safe(self) -> None:
        self.assertEqual(policy_split_names("train"), [])
        self.assertEqual(policy_split_names("val"), [])
        self.assertEqual(policy_split_names("test"), ["val"])
        self.assertEqual(threshold_reference_split_names("train"), [])
        self.assertEqual(threshold_reference_split_names("val"), ["train"])
        self.assertEqual(threshold_reference_split_names("test"), ["train", "val"])

    def test_open_prices_switch_execution_basis(self) -> None:
        panel = self._panel()
        self.assertEqual(infer_execution_basis(panel), "open_to_open")
        day2 = panel[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
        self.assertAlmostEqual(float(day2["exec_ret_1d"]), 0.0, places=8)
        day3 = panel[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2024-01-03"))].iloc[0]
        self.assertAlmostEqual(float(day3["exec_ret_1d"]), 0.10, places=8)

    def test_split_side_raw_fields_override_scaled_prediction_extras(self) -> None:
        preds = self.preds_df.copy()
        preds["turnover_5d"] = -999.0
        panel = build_backtest_panel(preds, self.split_df)
        merged = panel[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
        raw_value = float(
            self.split_df[(self.split_df["PERMNO"] == 1) & (self.split_df["DlyCalDt"] == pd.Timestamp("2024-01-01"))]["turnover_5d"].iloc[0]
        )
        self.assertAlmostEqual(float(merged["turnover_5d"]), raw_value, places=8)

    def test_build_candidates_filters_and_industry_cap(self) -> None:
        panel = self._panel()
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

    def test_portfolio_uses_next_open_to_open_returns(self) -> None:
        panel = self._panel()
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
            ret_col="exec_ret_1d",
        )
        self.assertEqual(len(trades_df), 1)
        first_trade = trades_df.iloc[0]
        self.assertEqual(pd.Timestamp(first_trade["signal_date"]), pd.Timestamp("2024-01-01"))
        self.assertEqual(pd.Timestamp(first_trade["entry_date"]), pd.Timestamp("2024-01-02"))
        self.assertEqual(pd.Timestamp(first_trade["exit_date"]), pd.Timestamp("2024-01-04"))
        self.assertAlmostEqual(float(daily_df[daily_df["date"] == pd.Timestamp("2024-01-02")]["portfolio_ret"].iloc[0]), 0.0, places=8)
        self.assertAlmostEqual(float(daily_df[daily_df["date"] == pd.Timestamp("2024-01-03")]["portfolio_ret"].iloc[0]), 0.10, places=8)
        self.assertAlmostEqual(float(daily_df[daily_df["date"] == pd.Timestamp("2024-01-04")]["portfolio_ret"].iloc[0]), 0.10, places=8)
        self.assertEqual(
            positions_df[positions_df["trade_id"] == first_trade["trade_id"]]["in_return_window"].tolist(),
            [0, 1, 1],
        )
        weighted = positions_df[positions_df["trade_id"] == first_trade["trade_id"]]["weighted_ret"].tolist()
        self.assertAlmostEqual(weighted[0], 0.0, places=8)
        self.assertAlmostEqual(weighted[1], 0.1, places=8)
        self.assertAlmostEqual(weighted[2], 0.1, places=8)

    def test_bid_ask_spread_costs_reduce_returns(self) -> None:
        panel = self._panel()
        panel.loc[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2024-01-02")), ["DlyBid", "DlyAsk"]] = [9.9, 10.1]
        panel.loc[(panel["permno"] == 1) & (panel["date"] == pd.Timestamp("2024-01-04")), ["DlyBid", "DlyAsk"]] = [11.88, 12.12]
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
            cost_bps_one_way=0.0,
            max_industry_weight=1.0,
            ret_col="exec_ret_1d",
            use_bid_ask_spread=True,
            spread_cost_cap_bps_one_way=1000.0,
        )
        self.assertAlmostEqual(float(daily_df[daily_df["date"] == pd.Timestamp("2024-01-02")]["portfolio_ret"].iloc[0]), -0.01, places=8)
        self.assertAlmostEqual(float(daily_df[daily_df["date"] == pd.Timestamp("2024-01-04")]["portfolio_ret"].iloc[0]), 0.09, places=8)
        first_trade = trades_df.iloc[0]
        self.assertAlmostEqual(float(first_trade["entry_spread_rate"]), 0.01, places=8)
        self.assertAlmostEqual(float(first_trade["exit_spread_rate"]), 0.01, places=8)
        self.assertAlmostEqual(float(first_trade["realized_holding_return"]), 0.19, places=8)

    def test_benchmark_and_summary_outputs(self) -> None:
        panel = self._panel()
        benchmark = build_equal_weight_benchmark(panel, min_price=3.0, ret_col="exec_ret_1d")
        day3 = benchmark[benchmark["date"] == pd.Timestamp("2024-01-03")].iloc[0]
        self.assertAlmostEqual(float(day3["benchmark_ret"]), (0.10 + 0.01 - 0.009803921568627416) / 3.0, places=8)

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
            ret_col="exec_ret_1d",
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

    def test_oracle_return_mode_uses_future_return_not_event_hit(self) -> None:
        panel = self._panel()
        oracle = build_oracle_candidates(panel, top_k=1, mode="return", holding_td=2, min_price=3.0)
        day = oracle[oracle["date"] == pd.Timestamp("2024-01-03")].iloc[0]
        self.assertEqual(int(day["permno"]), 2)
        self.assertEqual(int(day["y_div_10d"]), 0)
        self.assertEqual(str(day["signal_group"]), "oracle_return")

    def test_oracle_event_mode_does_not_use_future_return_tiebreak(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 1,
                    "DlyPrc": 10.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "11",
                    "y_div_10d": 1,
                    "fwd_ret_10d": 0.01,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 2,
                    "DlyPrc": 10.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "11",
                    "y_div_10d": 1,
                    "fwd_ret_10d": 0.50,
                },
            ]
        )
        oracle = build_oracle_candidates(panel, top_k=1, mode="event", min_price=3.0)
        day = oracle.iloc[0]
        self.assertEqual(int(day["permno"]), 1)
        self.assertEqual(int(day["y_div_10d"]), 1)
        self.assertEqual(str(day["signal_group"]), "oracle_hit")

    def test_oracle_event_mode_can_share_strategy_filters(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 1,
                    "prob": 0.95,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "11",
                    "div_count_exp": 6,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.0,
                    "y_div_10d": 0,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 2,
                    "prob": 0.80,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "20",
                    "div_count_exp": 3,
                    "gap_cv_exp": 0.60,
                    "z_to_med_exp": 0.0,
                    "y_div_10d": 1,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 3,
                    "prob": 0.99,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "30",
                    "div_count_exp": 1,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.0,
                    "y_div_10d": 1,
                },
            ]
        )
        oracle = build_oracle_candidates(
            panel,
            top_k=2,
            mode="event",
            min_price=3.0,
            exclude_div_count_le=1,
            stable_gap_cv_threshold=0.20,
            stable_div_count_min=4,
            stable_prob_threshold=0.45,
            regular_prob_threshold=0.75,
            max_industry_weight=1.0,
            use_dividend_rules=True,
        )
        self.assertEqual(oracle["permno"].tolist(), [2, 1])
        self.assertEqual(oracle["signal_group"].tolist(), ["regular", "stable"])
        self.assertEqual(oracle["oracle_event_hit"].tolist(), [1, 0])
        self.assertAlmostEqual(float(oracle.iloc[0]["prob"]), 0.80, places=8)
        self.assertAlmostEqual(float(oracle.iloc[1]["prob"]), 0.95, places=8)

    def test_non_prob_candidates_ignore_probability_thresholds(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 1,
                    "prob": 0.10,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "11",
                    "div_count_exp": 6,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.10,
                    "y_div_10d": 0,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 2,
                    "prob": 0.20,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 0.9,
                    "industry": "20",
                    "div_count_exp": 3,
                    "gap_cv_exp": 0.50,
                    "z_to_med_exp": 0.20,
                    "y_div_10d": 1,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 3,
                    "prob": 0.99,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "30",
                    "div_count_exp": 1,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.0,
                    "y_div_10d": 1,
                },
            ]
        )
        candidates = build_non_prob_candidates(
            panel=panel,
            top_k=2,
            min_price=3.0,
            exclude_div_count_le=1,
            stable_gap_cv_threshold=0.20,
            stable_div_count_min=4,
            stable_prob_threshold=0.45,
            regular_prob_threshold=0.75,
            max_industry_weight=1.0,
            use_dividend_rules=True,
        )
        self.assertEqual(candidates["permno"].tolist(), [1, 2])
        self.assertEqual(candidates["signal_group"].tolist(), ["stable", "regular"])

    def test_random_prob_candidates_share_prob_filtered_pool(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 1,
                    "prob": 0.90,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "11",
                    "div_count_exp": 6,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.10,
                    "y_div_10d": 0,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 2,
                    "prob": 0.80,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 0.9,
                    "industry": "20",
                    "div_count_exp": 3,
                    "gap_cv_exp": 0.50,
                    "z_to_med_exp": 0.20,
                    "y_div_10d": 1,
                },
                {
                    "date": pd.Timestamp("2024-01-03"),
                    "permno": 3,
                    "prob": 0.20,
                    "DlyPrc": 10.0,
                    "DlyRet": 0.0,
                    "exec_ret_1d": 0.0,
                    "turnover_5d": 1.0,
                    "industry": "30",
                    "div_count_exp": 6,
                    "gap_cv_exp": 0.10,
                    "z_to_med_exp": 0.0,
                    "y_div_10d": 1,
                },
            ]
        )
        candidates = build_random_prob_candidates(
            panel=panel,
            top_k=2,
            seed=7,
            min_price=3.0,
            exclude_div_count_le=1,
            stable_gap_cv_threshold=0.20,
            stable_div_count_min=4,
            stable_prob_threshold=0.45,
            regular_prob_threshold=0.75,
            max_industry_weight=1.0,
            use_dividend_rules=True,
        )
        self.assertEqual(set(candidates["permno"].tolist()), {1, 2})
        self.assertTrue((candidates["permno"] != 3).all())

if __name__ == "__main__":
    unittest.main()
