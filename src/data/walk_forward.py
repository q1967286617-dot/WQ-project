from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. All dates are pandas-compatible date strings (YYYY-MM-DD)."""
    fold_id: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def as_split_dict(self) -> dict:
        return {
            "train_start": self.train_start,
            "train_end":   self.train_end,
            "val_start":   self.val_start,
            "val_end":     self.val_end,
            "test_start":  self.test_start,
            "test_end":    self.test_end,
        }


def _year_bounds(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _shift_days(date_str: str, days: int) -> str:
    return (pd.to_datetime(date_str) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def build_annual_folds(
    test_start_year: int,
    test_end_year: int,
    train_years: int = 5,
    val_years: int = 1,
    mode: str = "rolling",
    embargo_calendar_days: int = 14,
) -> List[Fold]:
    """
    Generate yearly walk-forward folds.

    For each test year Y in [test_start_year, test_end_year]:
      val window  = Y-val_years .. Y-1
      train window = ends val_years before val_start
        - rolling:   train_years long
        - expanding: starts at min(2010-01-01, ...)

    embargo_calendar_days is applied as a gap between the end of one window
    and the start of the next (train_end ... val_start, val_end ... test_start).
    Defaults to ~10 trading days = 14 calendar days, matching configs/config.yaml
    embargo_td=10.
    """
    folds: List[Fold] = []
    for test_year in range(test_start_year, test_end_year + 1):
        test_start, test_end = _year_bounds(test_year)

        # val window ends embargo before test_start
        val_end_anchor   = _shift_days(test_start, -embargo_calendar_days)
        val_end_year     = pd.to_datetime(val_end_anchor).year
        val_start_year   = val_end_year - val_years + 1
        val_start, _     = _year_bounds(val_start_year)
        _, val_end       = _year_bounds(val_end_year)
        # Clip val_end so it stays embargo-clear of test_start
        val_end          = min(val_end, val_end_anchor)

        # train window ends embargo before val_start
        train_end_anchor = _shift_days(val_start, -embargo_calendar_days)
        train_end_year   = pd.to_datetime(train_end_anchor).year
        if mode == "rolling":
            train_start_year = train_end_year - train_years + 1
        elif mode == "expanding":
            train_start_year = 2010  # earliest available
        else:
            raise ValueError(f"Unknown mode: {mode}")
        train_start, _ = _year_bounds(train_start_year)
        _, train_end   = _year_bounds(train_end_year)
        train_end      = min(train_end, train_end_anchor)

        folds.append(Fold(
            fold_id=f"wf_{test_year}",
            train_start=train_start, train_end=train_end,
            val_start=val_start,     val_end=val_end,
            test_start=test_start,   test_end=test_end,
        ))
    return folds


def slice_panel_for_fold(full_panel: pd.DataFrame, fold: Fold, date_col: str = "DlyCalDt") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, val_df, test_df) sliced by the fold's date windows. Inclusive bounds.
    """
    d = pd.to_datetime(full_panel[date_col], errors="coerce")
    def _slice(start: str, end: str) -> pd.DataFrame:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        return full_panel[(d >= s) & (d <= e)]
    return _slice(fold.train_start, fold.train_end), _slice(fold.val_start, fold.val_end), _slice(fold.test_start, fold.test_end)


def iter_folds(folds: List[Fold]) -> Iterator[Fold]:
    for f in folds:
        yield f
