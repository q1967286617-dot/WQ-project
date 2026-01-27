# financial-event-prediction（金融分红事件预测）

## 项目简介
本项目将原始 notebook 重构为可复用的 Python 包与脚本流程，用于基于日频行情与分红事件数据，预测未来 H 个交易日内是否会发生分红宣告事件（DCLRDT）。
核心模型为 XGBoost（二分类），支持数值与类别特征混合输入，并提供事件级评估与运营风格的告警指标输出。

## 快速开始
1) 安装依赖（建议使用虚拟环境）：
```bash
python -m pip install pandas numpy pyyaml pyarrow xgboost scikit-learn joblib
```

2) 准备处理后的数据（`data/processed` 目录下的 `train/val/test.parquet`）：
```bash
# 训练
python scripts/run_build_dataset.py --cfg configs/config.yaml

python scripts/run_train.py --run_id <run_id>

# 预测
python scripts/run_predict.py --run_id <run_id> --split test

# 评估
python scripts/run_eval.py --run_id <run_id> --split test --mode threshold
```

> `run_id` 为空时训练脚本会自动生成时间戳。预测/评估需要与训练一致的 `run_id`。

## 目录结构
```
code2/
  configs/           # 路径、模型与评估配置
  data/              # 原始数据与中间产物
    raw/             # tableA.csv、tableB.csv
    interim/         # 分阶段中间产物（stage1/2/3）
    processed/       # 训练/验证/测试集 parquet
  models/            # 训练好的模型工件
  outputs/           # 运行输出（按 run_id 分目录）
  scripts/           # 训练/预测/评估入口脚本
  src/               # 核心逻辑（数据、建模、评估、工具）
  pyproject.toml
  README.md
```

## 数据格式与要求
原始数据默认放在 `data/raw`：
- `tableA.csv`（分红事件表）：必须包含列 `PERMNO`、`DCLRDT`
- `tableB.csv`（日频行情表）：必须包含列
  `PERMNO, DlyCalDt, Ticker, SICCD, DlyClose, DlyPrc, DlyVol, DlyRet, ShrOut`

配置中的路径可在 `configs/paths.yaml` / `configs/config.yaml` 中修改。

## 数据处理流程（可选，生成 processed 数据）
以下脚本位于 `src/data`，用于从 raw 构建特征、打标并切分：

1) 载入并筛选全量行情与事件：
```bash
# 建议在 src/data 目录执行（这些脚本采用本地相对导入）
cd src/data
python load.py
```
生成：
- `data/interim/stage1/df_full_raw.parquet`
- `data/interim/stage1/div_ev.parquet`

2) 构建特征：
```bash
python build_features.py
```
生成：
- `data/interim/stage2/df_full_feat.parquet`
- `data/interim/stage2/div_event_feats.parquet`

3) 生成标签（默认 `y_div_10d`，H 来自 `configs/config.yaml`）：
```bash
python label.py
```
生成：
- `data/interim/stage3/df_full_labeled.parquet`

4) 按时间切分：
```bash
python split.py
```
生成：
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

## 训练与预测
训练脚本会读取 `data/processed` 下的 parquet 文件，并在 `models/` 输出模型工件。
```bash
python scripts/run_train.py --run_id <run_id>
```

预测脚本会将结果写入 `outputs/runs/<run_id>/preds/`：
```bash
python scripts/run_predict.py --run_id <run_id> --split val
```

## 评估方式
评估脚本依赖预测产物与事件表（`tableA.csv`），支持两种模式：
- `threshold`：阈值触发告警（支持 cooldown）
- `topk`：每日 Top-K 告警（`--k` 控制）

```bash
python scripts/run_eval.py --run_id <run_id> --split test --mode threshold
# 或
python scripts/run_eval.py --run_id <run_id> --split test --mode topk --k 50
```

评估包含：
- 全局指标：AUC / AUC-PR / LogLoss / Precision / Recall / F1 等
- 事件级指标：命中率、误报率、lead time 等
- 诊断报告：Top-K 日报、队列分组指标、右侧删失诊断、运营日度统计等

## 运行产物说明
训练：
- `models/xgb_<run_id>.joblib`
- `outputs/runs/<run_id>/meta.txt`

预测：
- `outputs/runs/<run_id>/preds/<split>_preds.parquet`
- 若缺少 parquet 引擎则输出 `*.csv`

评估（写入 `outputs/runs/<run_id>/eval/`）：
- `summary.json`：全局与事件级汇总
- `analysis_summary.json`：结构化分析摘要（包含 best/worst 股票 AUC-PR 及 full 统计对比）
- `events_out.csv`：事件级命中表
- `alerts_out.csv`：告警表
- `daily_topk.csv`：每日 Top-K 指标
- `cohorts.csv`：分组表现
- `censoring_diag.csv`：删失诊断
- `phase_table.csv`：lead time 分布
- `ops_daily.csv`：运营日度模拟
- `event_recall_by_date.csv`：按事件日期的召回
- `stock_cohorts.csv`：股票分组标签
- `stock_cohorts_cutoff.csv`：截断版（仅使用 eval 起始日前的事件统计）
- `stock_cohorts_full.csv`：全历史版（使用全历史事件统计）
- `stock_aucpr_best_worst.csv`：AUC-PR 最好/最差各 10 支股票（包含 cutoff/full 两套字段）

同时会输出：
- `outputs/runs/<run_id>/preds/eval_df.parquet`

## 配置说明
- `configs/paths.yaml`
  - 原始数据、processed、models、outputs 的路径
- `configs/config.yaml`
  - 原始数据路径、起止日期、标签窗口 H、冷却期、阈值、分割日期、embargo 等
- `configs/model.yaml`
  - 特征列（数值/类别）、目标列、是否缩放、XGBoost 超参数
- `configs/eval.yaml`
  - 评估窗口 H、阈值、Top-K 列表、分割日期、embargo

脚本参数可覆盖配置文件路径：
- 训练：`--paths`、`--model_cfg`
- 评估：`--paths`、`--eval_cfg`

## 备注与常见问题
- 缺少 parquet 引擎时，读取 parquet 会报错，写入会自动回退到 CSV。
- XGBoost 会在可用且安装了 CUDA/CuPy 时自动使用 GPU，否则使用 CPU。
- 若在 Kaggle 环境运行，`outputs/data/models` 会自动重定向到可写目录。

## 推荐工作流（端到端）
```bash
# 数据处理（一次性）
cd src/data
python load.py
python build_features.py
python label.py
python split.py

# 训练/预测/评估
cd ../../
python scripts/run_train.py --run_id 2026-01-24_165844
python scripts/run_predict.py --run_id 2026-01-24_165844 --split test
python scripts/run_eval.py --run_id 2026-01-24_165844 --split test --mode threshold
```

如需调整特征、标签窗口或评估策略，请先修改配置文件，然后重新训练与评估。
