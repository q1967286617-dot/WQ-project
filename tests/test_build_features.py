from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.build_features import build_causal_features_full, build_div_event_features


class BuildFeaturesTests(unittest.TestCase):
    def test_merge_asof_normalizes_mixed_datetime_units(self) -> None:
        df_full = pd.DataFrame(
            {
                "PERMNO": [1, 1],
                "DlyCalDt": pd.Series(
                    np.array(["2024-01-03", "2024-01-04"], dtype="datetime64[ns]")
                ),
                "DlyPrc": [10.0, 10.5],
                "ShrOut": [100.0, 100.0],
                "DlyRet": [0.01, 0.02],
                "DlyVol": [1000.0, 1200.0],
                "DlyHigh": [10.2, 10.7],
                "DlyLow": [9.8, 10.3],
                "DlyOpen": [10.0, 10.4],
                "DlyClose": [10.1, 10.5],
                "DlyBid": [10.0, 10.4],
                "DlyAsk": [10.2, 10.6],
                "DlyNumTrd": [20, 22],
                "SICCD": [1311, 1311],
            }
        )
        div_ev = pd.DataFrame(
            {
                "PERMNO": [1, 1],
                "DCLRDT": pd.Series(
                    np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[us]")
                ),
                "EXDT": pd.Series(
                    np.array(["2024-01-06", "2024-01-07"], dtype="datetime64[us]")
                ),
                "DIVAMT": [0.5, 0.6],
                "FACSHR": [1.0, 1.0],
            }
        )

        div_event_feats = build_div_event_features(div_ev, recent_n=2)
        out = build_causal_features_full(df_full, div_ev, div_event_feats)

        self.assertEqual(str(out["DlyCalDt"].dtype), "datetime64[ns]")
        self.assertEqual(str(out["last_div_dclrdt"].dtype), "datetime64[ns]")
        self.assertEqual(len(out), 2)
        self.assertEqual(out["last_div_dclrdt"].tolist(), [pd.Timestamp("2024-01-02")] * 2)
        self.assertEqual(out["days_since_last_div"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
