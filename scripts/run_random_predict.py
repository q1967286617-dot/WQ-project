from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, load_yaml, resolve_paths


logger = get_logger("run_random_predict")

KEEP_EXTRA_COLS = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry", "has_div_history"]


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read parquet: {path}. Install pyarrow (recommended) or convert to CSV.\n"
                f"Original error: {e}"
            )
    return pd.read_csv(path)


def random_predict_to_eval_df(
    df: pd.DataFrame,
    seed: int = 42,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    target_col: str = "y_div_10d",
    keep_extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = pd.to_numeric(x[permno_col], errors="coerce").astype("Int64")
    x = x.loc[x[permno_col].notna()].copy()
    x[permno_col] = x[permno_col].astype(int)
    x = x.sort_values([permno_col, date_col]).reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    prob = rng.random(len(x), dtype=float)
    y = x[target_col].astype(int).values if target_col in x.columns else np.full(len(x), -1, dtype=int)

    eval_df = pd.DataFrame(
        {
            "date": x[date_col].values,
            "permno": x[permno_col].values,
            "y": y,
            "prob": prob,
        }
    )

    if keep_extra_cols:
        for c in keep_extra_cols:
            if c in x.columns and c not in eval_df.columns:
                eval_df[c] = x[c].values

    return eval_df.sort_values(["permno", "date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    ensure_dir(run_dir / "preds")

    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    df = _read_df(split_path)

    eval_df = random_predict_to_eval_df(df, seed=args.seed, keep_extra_cols=KEEP_EXTRA_COLS)

    out_path = run_dir / "preds" / f"{args.split}_preds.parquet"
    try:
        eval_df.to_parquet(out_path, index=False)
        logger.info(f"saved: {out_path}")
    except Exception as e:
        out_csv = out_path.with_suffix(".csv")
        eval_df.to_csv(out_csv, index=False)
        logger.info(f"parquet unavailable ({e}); saved CSV instead: {out_csv}")


if __name__ == "__main__":
    main()
