from __future__ import annotations

import pandas as pd


def build_equal_weight_benchmark(panel: pd.DataFrame, min_price: float) -> pd.DataFrame:
    x = panel.copy()
    tradable = (
        pd.to_numeric(x["DlyPrc"], errors="coerce").abs().ge(float(min_price))
        & pd.to_numeric(x["DlyRet"], errors="coerce").notna()
    )
    x = x[tradable].copy()
    out = (
        x.groupby("date", sort=True)
        .agg(
            benchmark_ret=("DlyRet", "mean"),
            benchmark_n=("permno", "size"),
        )
        .reset_index()
    )
    out["benchmark_ret"] = out["benchmark_ret"].fillna(0.0)
    out["benchmark_nav"] = (1.0 + out["benchmark_ret"]).cumprod()
    return out
