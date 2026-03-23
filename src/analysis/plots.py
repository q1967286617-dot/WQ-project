from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_nav_vs_benchmark(daily_df: pd.DataFrame, path: str | Path, title: str) -> None:
    out = Path(path)
    _ensure_parent(out)
    x = daily_df.copy()
    if x.empty:
        return
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x["date"], x["portfolio_nav"], label="Portfolio", linewidth=2.0)
    if "benchmark_nav" in x.columns:
        ax.plot(x["date"], x["benchmark_nav"], label="Benchmark", linewidth=1.8)
    if "excess_nav" in x.columns:
        ax.plot(x["date"], x["excess_nav"], label="Excess NAV", linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    out = Path(path)
    _ensure_parent(out)
    data = df.copy()
    if data.empty:
        return
    pivot = data.pivot(index=x_col, columns=hue_col, values=y_col)
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
