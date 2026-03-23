"""
scripts/run_ablation.py

特征消融实验：逐组剔除 v2 新增特征，在验证集上观察 AUC-PR 和回测超额 Sharpe 的变化。

用法：
    python scripts/run_ablation.py

结果输出：
    outputs/ablation/ablation_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import shutil
import subprocess
import yaml
import pandas as pd
from copy import deepcopy

ABLATION_EXPERIMENTS = {
    "v2_full":    [],
    "no_DIVAMT":  ["div_amt_last", "div_amt_mean_exp", "div_amt_chg",
                   "div_amt_dir_mean_r3", "div_no_cut_streak"],
    "no_EXDT":    ["decl_to_exdt_days", "decl_to_exdt_mean_exp"],
    "no_OHLC":    ["daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct"],
    "no_LIQ":     ["bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike"],
    "no_ALL_NEW": [
        "div_amt_last", "div_amt_mean_exp", "div_amt_chg",
        "div_amt_dir_mean_r3", "div_no_cut_streak",
        "decl_to_exdt_days", "decl_to_exdt_mean_exp",
        "daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct",
        "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike",
        "industry",
    ],
}

V2_BASE_NUM_COLS = [
    "log_mkt_cap", "ret_5d", "ret_21d", "vol_5d", "vol_21d",
    "turnover_5d", "turnover_21d", "price_to_high", "volume_spike",
    "vol_ratio", "turnover_ratio", "ret_rel_to_ind",
    "days_since_last_div", "gap_mean_exp", "gap_med_exp", "gap_std_exp",
    "gap_mean_rN", "gap_std_rN", "gap_cv_exp", "time_to_med_exp",
    "z_to_med_exp", "time_to_mean_rN", "z_to_mean_rN", "div_count_exp",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "div_amt_last", "div_amt_mean_exp", "div_amt_chg",
    "div_amt_dir_mean_r3", "div_no_cut_streak",
    "decl_to_exdt_days", "decl_to_exdt_mean_exp",
    "daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct",
    "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike",
]

V2_BASE_CAT_COLS = ["quarter", "weekday", "is_month_start", "is_month_end", "industry"]


def run_cmd(cmd):
    print(f"\n  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def read_json_field(path, *keys):
    if not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    for k in keys:
        if isinstance(data, dict):
            data = data.get(k)
        else:
            return None
    return data


def main():
    MODEL_CFG    = "configs/model.yaml"
    BACKTEST_CFG = "configs/backtest.yaml"
    PATHS_CFG    = "configs/paths.yaml"
    CFG          = "configs/config.yaml"

    with open(MODEL_CFG) as f:
        base_model_cfg = yaml.safe_load(f)
    with open(BACKTEST_CFG) as f:
        base_bt_cfg = yaml.safe_load(f)

    bt_ablation = deepcopy(base_bt_cfg)
    bt_ablation["dividend_rules_mode"] = "true"

    out_dir = Path("outputs/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(MODEL_CFG,    out_dir / "model_yaml_backup.yaml")
    shutil.copy(BACKTEST_CFG, out_dir / "backtest_yaml_backup.yaml")

    results = []

    for exp_name, feats_to_remove in ABLATION_EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"实验: {exp_name}  剔除 {len(feats_to_remove)} 个特征")
        print(f"{'='*60}")

        run_id = f"ablation_{exp_name}"

        exp_cfg = deepcopy(base_model_cfg)
        exp_cfg["num_cols"] = [f for f in V2_BASE_NUM_COLS if f not in feats_to_remove]
        exp_cfg["cat_cols"] = [f for f in V2_BASE_CAT_COLS if f not in feats_to_remove]

        with open(MODEL_CFG, "w") as f:
            yaml.dump(exp_cfg, f, allow_unicode=True, sort_keys=False)
        with open(BACKTEST_CFG, "w") as f:
            yaml.dump(bt_ablation, f, allow_unicode=True, sort_keys=False)

        print(f"  特征数: num={len(exp_cfg['num_cols'])}, cat={len(exp_cfg['cat_cols'])}")

        # 训练
        if run_cmd(["python", "scripts/run_train.py",
                    "--paths", PATHS_CFG, "--model_cfg", MODEL_CFG,
                    "--run_id", run_id]) != 0:
            results.append({"experiment": exp_name, "status": "train_failed"})
            continue

        # 预测 val
        if run_cmd(["python", "scripts/run_predict.py",
                    "--paths", PATHS_CFG, "--run_id", run_id,
                    "--split", "val"]) != 0:
            results.append({"experiment": exp_name, "status": "predict_failed"})
            continue

        # 评估 val
        run_cmd(["python", "scripts/run_eval.py",
                 "--cfg", CFG, "--paths", PATHS_CFG,
                 "--run_id", run_id, "--split", "val"])

        # 回测 val
        run_cmd(["python", "scripts/run_backtest.py",
                 "--paths", PATHS_CFG, "--backtest_cfg", BACKTEST_CFG,
                 "--run_id", run_id, "--split", "val", "--skip_reference"])

        # 读取结果路径
        from src.utils.paths import load_yaml, resolve_paths
        paths = resolve_paths(load_yaml(Path(PATHS_CFG)), project_root=Path.cwd())
        run_dir = paths.outputs_dir / run_id

        aucpr  = read_json_field(run_dir/"eval_stratified"/"summary.json",
                                 "global_metrics", "aucpr")
        sharpe = read_json_field(run_dir/"backtest"/"summary.json",
                                 "excess_vs_benchmark", "sharpe")

        results.append({
            "experiment":        exp_name,
            "n_num":             len(exp_cfg["num_cols"]),
            "n_cat":             len(exp_cfg["cat_cols"]),
            "removed":           len(feats_to_remove),
            "val_aucpr":         round(aucpr, 6)  if aucpr  else None,
            "val_excess_sharpe": round(sharpe, 4) if sharpe else None,
            "status":            "ok",
        })
        print(f"  AUC-PR={aucpr:.4f}  超额Sharpe={sharpe:.4f}")

    # 恢复配置
    shutil.copy(out_dir/"model_yaml_backup.yaml",    MODEL_CFG)
    shutil.copy(out_dir/"backtest_yaml_backup.yaml", BACKTEST_CFG)
    print("\n✓ 已恢复原始配置文件")

    df = pd.DataFrame(results)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("消融实验汇总")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    ok = df[df["status"] == "ok"]
    if len(ok) > 0 and "val_excess_sharpe" in ok.columns:
        best = ok.loc[ok["val_excess_sharpe"].idxmax()]
        print(f"\n最佳配置: {best['experiment']}")
        print(f"  AUC-PR={best['val_aucpr']}  超额Sharpe={best['val_excess_sharpe']}")

    print(f"\n结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
