from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.predict import predict_to_eval_df


logger = get_logger("run_predict")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--model_artifacts", default=None, help="Override artifacts path; otherwise inferred from models/xgb_<run_id>.joblib")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    ensure_dir(run_dir / "preds")

    # Load split data
    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    df = _read_df(split_path)

    art_path = Path(args.model_artifacts) if args.model_artifacts else (paths.models_dir / f"xgb_{args.run_id}.joblib")
    if not art_path.exists():
        raise FileNotFoundError(f"Missing artifacts: {art_path}")

    # Keep extra columns for cohorting if present
    keep = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry"]
    eval_df = predict_to_eval_df(df, art_path, keep_extra_cols=keep)

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
