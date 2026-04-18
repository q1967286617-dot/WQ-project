"""Train a Logistic Regression binary classifier (L1/L2, sklearn).

Usage:
    python scripts/run_train_lr.py --run_id lr_v1
    python scripts/run_train_lr.py --run_id lr_v1 --C 0.05 --penalty l2

Model artifact is saved to models/lr_<run_id>.joblib.

Notes:
  - Numeric features are StandardScaler-scaled (do_scale=True).
  - Categorical features are one-hot encoded; unseen categories are silently ignored.
  - Default penalty=l1 with saga solver provides automatic feature selection.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.preprocess import (
    fit_train_imputer_and_scaler,
    apply_imputer_and_scaler,
)
from src.modeling.train_lr import (
    LRArtifacts,
    train_lr_binary,
    save_artifacts,
)

logger = get_logger("run_train_lr")


def _read_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--C", type=float, default=0.1, help="Inverse regularization strength")
    ap.add_argument("--penalty", default="l2", choices=["l1", "l2", "elasticnet"], help="Regularization type")
    ap.add_argument("--solver", default="lbfgs", choices=["lbfgs", "saga", "liblinear"], help="Optimization solver")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--poly_degree", type=int, default=1,
                    help="Degree for PolynomialFeatures on numeric columns. 1=no expansion, 2=quadratic interactions.")
    ap.add_argument("--interaction_only", action="store_true",
                    help="If set with poly_degree>=2, only generate cross terms (no x^2 terms).")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    poly_degree = int(args.poly_degree)
    paths_cfg = load_yaml(Path(args.paths))
    model_cfg = load_yaml(Path(args.model_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / run_id
    ensure_dir(run_dir)
    ensure_dir(paths.models_dir)
    logger.info(f"run_id={run_id}, C={args.C}, penalty={args.penalty}, poly_degree={poly_degree}")

    train_df = _read_df(paths.processed_dir / "train.parquet")
    val_df = _read_df(paths.processed_dir / "val.parquet")

    target_col = model_cfg.get("target_col", "y_div_10d")
    num_cols = list(model_cfg["num_cols"])
    cat_cols = list(model_cfg["cat_cols"])

    # LR needs scaling
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=True)
    val_df = apply_imputer_and_scaler(val_df, num_cols, stats, do_scale=True)

    # One-hot encode categoricals; fit on train, handle_unknown='ignore' for val/test
    cat_data_train = train_df[cat_cols].astype(str).fillna("<<MISSING>>")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(cat_data_train)

    X_train_num = train_df[num_cols].values
    X_train_cat = ohe.transform(cat_data_train)
    y_train = train_df[target_col].astype(int).values

    # --- Polynomial feature expansion (fitted on train numerics only) ---
    poly = None
    if poly_degree >= 2:
        poly = PolynomialFeatures(
            degree=poly_degree,
            interaction_only=bool(args.interaction_only),
            include_bias=False,
        )
        poly.fit(X_train_num)
        n_poly_feats = poly.transform(X_train_num[:1]).shape[1]
        logger.info(
            f"PolynomialFeatures degree={poly_degree}, interaction_only={args.interaction_only}: "
            f"{len(num_cols)} numeric -> {n_poly_feats} poly features"
        )
    else:
        n_poly_feats = len(num_cols)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    logger.info(f"train pos={pos}, neg={neg}, ratio={pos/(pos+neg):.3f}")
    logger.info(
        f"feature dim: num_poly={n_poly_feats}, cat_ohe={X_train_cat.shape[1]}, "
        f"total={n_poly_feats + X_train_cat.shape[1]}"
    )

    model = train_lr_binary(
        X_train_num=X_train_num,
        X_train_cat=X_train_cat,
        y_train=y_train,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        poly=poly,
    )

    # Log val AUCPR
    from sklearn.metrics import average_precision_score
    val_cat = ohe.transform(val_df[cat_cols].astype(str).fillna("<<MISSING>>"))
    val_num = poly.transform(val_df[num_cols].values) if poly is not None else val_df[num_cols].values
    X_val = np.hstack([val_num, val_cat])
    y_val = val_df[target_col].astype(int).values
    val_prob = model.predict_proba(X_val)[:, 1]
    val_aucpr = average_precision_score(y_val, val_prob)
    logger.info(f"val AUCPR={val_aucpr:.4f}")

    art = LRArtifacts(
        model=model,
        ohe=ohe,
        impute_stats=stats,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_col=target_col,
        do_scale=True,
        poly=poly,
    )

    art_path = paths.models_dir / f"lr_{run_id}.joblib"
    save_artifacts(art, art_path)
    logger.info(f"saved artifacts: {art_path}")
    (run_dir / "meta.txt").write_text(f"model_artifacts={art_path}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
