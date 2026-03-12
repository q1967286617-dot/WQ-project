from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger

from src.data import load, build_features, label, split


logger = get_logger("run_build_dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--recent_n", type=int, default=3)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.cfg))
    paths = resolve_paths(cfg, project_root=Path.cwd())

    start_all = cfg["start_all"]
    end_all = cfg["end_all"]
    div_distcd = cfg["div_distcd"]
    h_label = int(cfg.get("H_label", 10))
    split_cfg = cfg["split"]

    logger.info("loading raw data")
    permno_set = load.build_fixed_universe(paths.tableB_path, start_all, end_all)
    df_full_raw = load.load_market_data_full(paths.tableB_path, permno_set, start_all, end_all)
    div_ev = load.load_div_events(paths.tableA_path, div_distcd)

    logger.info("building features")
    div_event_feats = build_features.build_div_event_features(div_ev, recent_n=args.recent_n)
    df_full_feat = build_features.build_causal_features_full(df_full_raw, div_ev, div_event_feats)

    logger.info("labeling")
    div_dict = label.build_event_dict(div_ev, "DCLRDT")
    df_full_labeled = label.label_within_h_trading_days(
        df_full_feat,
        div_dict,
        h=h_label,
        label_name="y_div_10d",
    )

    logger.info("splitting")
    train_df, val_df, test_df = split.split_by_date(df_full_labeled, split_cfg)

    ensure_dir(paths.processed_dir)
    train_path = paths.processed_dir / "train.parquet"
    val_path = paths.processed_dir / "val.parquet"
    test_path = paths.processed_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"saved splits: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()