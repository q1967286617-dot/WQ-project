from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import CONFIG_DIR, DATA_DIR
from .load import DATE_COL, PERMNO_COL


def build_event_dict(events: pd.DataFrame, date_col: str):
    out = {}
    for permno, group in events.groupby(PERMNO_COL):
        out[int(permno)] = np.sort(group[date_col].values.astype("datetime64[ns]"))
    return out


def label_within_h_trading_days(df: pd.DataFrame, event_dict: dict, h: int, label_name: str):
    x = df.copy().sort_values([PERMNO_COL, DATE_COL])
    x["t_end"] = x.groupby(PERMNO_COL)[DATE_COL].shift(-h)
    x = x.dropna(subset=["t_end"])

    t0 = x[DATE_COL].values.astype("datetime64[ns]")
    t1 = x["t_end"].values.astype("datetime64[ns]")
    perm = x[PERMNO_COL].values.astype(int)

    y = np.zeros(len(x), dtype=np.int8)
    for permno in np.unique(perm):
        idx = np.where(perm == permno)[0]
        arr = event_dict.get(int(permno))
        if arr is None or len(arr) == 0:
            continue
        left = np.searchsorted(arr, t0[idx], side="right")
        right = np.searchsorted(arr, t1[idx], side="right")
        y[idx] = (right > left).astype(np.int8)

    x[label_name] = y
    return x.drop(columns=["t_end"])


def label_forward_return(
    df: pd.DataFrame,
    h: int,
    price_col: str = "DlyOpen",
    label_name: str = "fwd_ret_10d",
) -> pd.DataFrame:
    """
    Forward return aligned with the backtest execution convention:
      - enter at t+1 open
      - exit at t+1+h open
      - label_t = p_exit / p_entry - 1

    Rows near the tail (where t+1+h is unavailable) get NaN.
    """
    x = df.sort_values([PERMNO_COL, DATE_COL]).copy()
    g = x.groupby(PERMNO_COL, observed=True)[price_col]
    p_entry = g.shift(-1)
    p_exit = g.shift(-(1 + h))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (p_exit.astype("float64") / p_entry.astype("float64")) - 1.0
    # Mask non-positive or missing entry prices
    ret = ret.where(p_entry.astype("float64") > 0.0)
    x[label_name] = ret.astype("float32")
    return x


def main():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    h_div = data["H_label"]
    div_ev = pd.read_parquet(DATA_DIR / "interim" / "stage1" / "div_ev.parquet")
    df_full_feat = pd.read_parquet(DATA_DIR / "interim" / "stage2" / "df_full_feat.parquet")

    div_dict = build_event_dict(div_ev, "DCLRDT")
    df_full_labeled = label_within_h_trading_days(df_full_feat, div_dict, h=h_div, label_name="y_div_10d")

    output_dir = DATA_DIR / "interim" / "stage3"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_full_labeled.to_parquet(output_dir / "df_full_labeled.parquet", index=False)

    print(f"Data Labelled: {output_dir}")


if __name__ == "__main__":
    main()
