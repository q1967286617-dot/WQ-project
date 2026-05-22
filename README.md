# 分红事件预测与回测系统

本仓库用于构建股票日频数据集、训练分红公告事件分类模型、评估事件预警质量，并把预测概率转换为可交易的日频选股信号进行组合回测。代码对时序泄漏、规则选择偏差、回测口径、面板隔离等问题做过审计与修正，适合做日频研究型回测。

仓库以**版本化实验**为核心组织方式：同一份代码通过不同的 `run_id` + `configs/versions/<vN>_*.yaml` 组合可以并行跑多组实验，从基线模型（v1）到加入 Compustat 基本面 + HPO 调优 + 滚动重训（v12 / walk-forward）均可复现。

## 项目在做什么

回答两个相互独立的问题：

1. **事件预警**：能否预测某只股票在未来 `H=10` 个交易日内是否会发生分红公告事件？
2. **可交易性**：能否把事件预测转成一套日频长仓 Top-K 策略，并在考虑成本、执行滞后、停牌、价差、行业约束等真实摩擦下评估其表现？

两个层次的评估完全分离：
- 事件预警质量由 `run_eval.py` 度量（精度、召回、误报、冷却期、按历史分红次数分层等）。
- 组合表现由 `run_backtest.py` 度量（日度净值、成本、换手、相对基准超额、池/排名分解、年度归因等）。

## 仓库结构

```text
configs/
  config.yaml                   数据集构建：输入路径、标签 H、分红代码、切分日期、embargo
  model.yaml                    XGBoost 默认特征列表与超参数
  backtest.yaml                 默认回测参数：top_k、持有期、阈值、成本、基准
  backtest_v9_optimized.yaml    HPO 之后的优化回测配置（top_k=15, holding_td=7 等）
  paths.yaml                    输出目录
  version_registry.yaml         版本登记：feature_groups + v1..v12 实验定义 + 报告目标
  versions/                     materialize_version_configs.py 物化出的逐版本 model/backtest yaml
                                  含 v12 成本敏感性套件（cost_2/5/10 × spread off/25/50/100）

data/
  raw/                          tableA（分红事件）、tableB（日频行情与基本面）
  processed/                    train/val/test 的 parquet 切分
                                  *_with_fundamentals.parquet：合并 Compustat 后的版本（v9+）

models/                         训练产物：
                                  xgb_<run_id>.joblib       分类器（含 TrainArtifacts）
                                  xgb_reg_<run_id>.joblib   双 booster 的回归头（run_train_dual）
                                  xgb_<combined>__<fold>.joblib   walk-forward 每折模型

outputs/runs/<run_id>/
  preds/                        {train,val,test}_preds.parquet
  eval/                         事件预警评估结果
  backtest/                     daily_portfolio.csv / trades.csv / positions.csv / summary.json
                                  + 成本、换手、池/排名分解、top-k 收益诊断
outputs/analysis/               跨 run 的分析产物（年度归因、SHAP drift、tradability、reliability）

scripts/                        所有可执行入口（见下节）
  analysis/                       calib_shap_50v200.py 、shap_deep_and_tradability.py
src/
  data/           load、build_features、label、split、walk_forward
  modeling/       preprocess、train（含 binary / regressor / ranker）、predict（xgb/cat/lgbm/lr）
  backtest/      signal、portfolio、benchmark、report
  eval/          eval_tools、report、plots
  analysis/      backtest_analysis、plots（score bucket、池/排名分解、年度归因等）
  experiments/   versioning（version_registry 加载与物化）
  utils/         paths、logging

tests/           test_backtest.py、test_build_features.py
notebooks/       探索性 notebook
reports/         静态报告
```

## 数据输入

- `tableA`：分红事件表，使用 `DCLRDT` 作为公告日。通过 `config.yaml` 中 `div_distcd` 过滤有效分红类型（1202/1212/…/1272）。
- `tableB`：日频行情与基本面，含 OHLC、成交量、买卖盘等。
- `start_all`~`end_all`（默认 2010-01-01 ~ 2024-12-31）定义全样本范围。
- **Compustat 基本面（可选）**：`scripts/fetch_compustat.py` 通过 WRDS 拉取季度财报，生成 `roe / payout_ratio / div_coverage / dvpsxq / leverage / profit_margin` 六个字段并 PIT 合并进 panel，输出 `data/processed/{train,val,test}_with_fundamentals.parquet`，供 v9+ 使用。

切分与 embargo 定义在 `config.yaml` 的 `split` 字段与 `embargo_td`（默认 10 个交易日）。embargo 防止训练特征跨入标签窗口，避免未来信息泄漏。

## 完整流程

### A. 单期端到端（默认 split）

```bash
# 1. 读取原始表、构建特征面板与标签、切分
python scripts/run_build_dataset.py --cfg configs/config.yaml

# 2.（可选）拉取 Compustat 并合并基本面
python scripts/fetch_compustat.py

# 3. 训练 XGBoost 二分类
python scripts/run_train.py --run_id v12_run1 \
    --model_cfg configs/versions/v12_model.yaml \
    --data_suffix _with_fundamentals

# 4. 对每个 split 生成预测概率
python scripts/run_predict.py --run_id v12_run1 --split val --data_suffix _with_fundamentals
python scripts/run_predict.py --run_id v12_run1 --split test --data_suffix _with_fundamentals

# 5. 事件预警质量评估
python scripts/run_eval.py --run_id v12_run1 --split test

# 6. 组合回测
python scripts/run_backtest.py --run_id v12_run1 --split test \
    --cfg configs/versions/v12_backtest.yaml
```

每个脚本以 `run_id` 为组织单位，产物一律落在 `outputs/runs/<run_id>/` 下，方便并行多版本实验。

### B. Walk-forward 滚动重训（年度）

`scripts/run_walk_forward.py` 编排「按年滚动训练 → 拼接预测 → 一次回测」的完整流程：

```bash
python scripts/run_walk_forward.py \
    --combined_id wf_v12_5y \
    --model_cfg configs/versions/v12_model.yaml \
    --backtest_cfg configs/versions/v12_backtest.yaml \
    --data_suffix _with_fundamentals \
    --test_start 2023 --test_end 2024 --train_years 5
```

每年作为一个 test fold，前 `train_years` 年 + 一年 val 作为该折的训练区间；每折产物落在 `models/xgb_<combined>__<fold>.joblib`，预测拼接后写入 `outputs/runs/<combined_id>/preds/`，最后调用一次 `run_backtest.py` 在拼接的预测上跑回测。

为了让单期（ST）回测与 walk-forward（WF）回测在同一时间窗口上可比，`run_backtest.py` 新增 `--reset_active_at <YYYY-MM-DD>`：在该日期之前清空所有 active 持仓，避免 ST 模型从 val 末尾继承的持仓污染对比。

### C. 双 booster（分类 + 回归）

`run_train_dual.py` 同时训练 `y_div_10d` 的二分类和 `fwd_ret_Hd` 的回归器，两者共享同一套预处理。`run_predict_dual.py` 把两个分数**按交易日横截面 rank 归一**后线性融合：

```
prob = alpha * rank(score_cls) + (1 - alpha) * rank(score_reg)
```

下游的 `run_eval.py` / `run_backtest.py` 直接消费 `prob` 列，无需任何改动；`score_cls` / `score_reg` 作为诊断列保留。

```bash
python scripts/run_train_dual.py --run_id dual_v12 --model_cfg configs/versions/v12_model.yaml
python scripts/run_predict_dual.py --run_id dual_v12 --split test --alpha 0.6
```

## 脚本清单

### 核心管线

| 脚本 | 作用 |
| --- | --- |
| `run_build_dataset.py` | 从 tableA/B 构建因果特征面板，生成 `y_div_10d` 标签，切分 train/val/test |
| `fetch_compustat.py` | 通过 WRDS 拉取 Compustat 季报，PIT 合并基本面字段（v9+） |
| `run_train.py` | 训练 XGBoost；在 train 上拟合 impute/scale，val 上早停；保存 booster 与预处理统计量 |
| `run_predict.py` | 加载模型对指定 split 生成预测 parquet |
| `run_eval.py` | 按阈值/冷却期规则评估事件预警（精度、召回、误报、按分红历史分层） |
| `run_backtest.py` | 组合回测，输出日度净值、交易、持仓、成本/换手归因与诊断表 |

### 替代模型与基线

| 脚本 | 作用 |
| --- | --- |
| `run_train_catboost.py` | CatBoost 训练 |
| `run_train_lgbm.py` | LightGBM 训练 |
| `run_train_lr.py` | Logistic Regression（含多项式特征选项） |
| `run_train_dual.py` | 分类 + 回归双 booster，共享预处理 |
| `run_predict_dual.py` | 双 booster 横截面 rank 融合 |
| `run_random_predict.py` | 随机分数零假设基线 |

### 滚动重训与超参搜索

| 脚本 | 作用 |
| --- | --- |
| `run_walk_forward.py` | 年度滚动 train → predict → 拼接 → 回测 |
| `sweep_hpo.py` | `max_depth × learning_rate × subsample` 27 组合搜索，按 val AUCPR 选优，top-K 进一步跑回测 |
| `sweep_backtest_params.py` | 回测参数（top_k / holding_td / 阈值）基础扫描 |
| `sweep_backtest_v2.py` | 精细化扫描 + tradability 过滤（如 `bid_ask_spread_5d` 上限） |
| `run_rounds_scan.py` | 训练轮数扫描 |

### 分析与诊断

| 脚本 | 作用 |
| --- | --- |
| `run_standardized_pipeline.py` | 固定配置的端到端一键流水线 |
| `run_ablation.py` | 特征消融 |
| `run_pool_rank_decomposition.py` | 把回测收益分解为「候选池效应」vs「排名效应」 |
| `run_version_flip_analysis.py` | 跨版本预测翻转分析 |
| `analyze_feature_importance.py` | 特征重要性导出 |
| `analyze_yearly_attribution.py` | 多版本年度 Sharpe / 超额 / 命中率归因 |
| `analysis/calib_shap_50v200.py` | 概率校准（Brier/ECE/可靠性）+ SHAP 漂移：50 轮 vs 200 轮 |
| `analysis/shap_deep_and_tradability.py` | 深度 SHAP 对比 + 选股名单的 tradability 检验（验证「多训轮数学到的是 alpha 还是不可交易微结构」） |

### 报告生成

| 脚本 | 作用 |
| --- | --- |
| `materialize_version_configs.py` | 从 `version_registry.yaml` 为每个版本物化独立 `configs/versions/<v>_*.yaml` |
| `run_report_version_suite.py` / `render_version_suite_report.py` | 版本套件对比报告 |
| `generate_today_weekly_report_v3.py` | 周报 docx 生成 |

## 特征与版本登记

特征构建在 `src/data/build_features.py`，按组织在 `configs/version_registry.yaml` 的 `feature_groups` 中：

- `base_num` / `base_cat`：日历 sin/cos、价格/动量/波动率、分红历史间隔统计、行业相对收益等。
- `divamt_num`：历次分红金额及变动方向。
- `exdt_num`：宣告至除权日的时距。
- `ohlc_num`：日内 range、ATR、open gap。
- `liq_num`：买卖价差、成交笔数等流动性。
- `industry_cat`：行业类别。
- `fundamentals_num`：Compustat 六字段（v9+）。

`versions:` 块按版本组合特征组与超参，目前覆盖：

| 版本 | 说明 |
| --- | --- |
| `random` | 随机基线 |
| `v1` | 30+4 基线 |
| `v2` | 全量特征 46+5（auto 阈值） |
| `v3` | 全量 + 强制 dividend_rules_mode |
| `v4` / `v5` | 多轮训练（400 轮）变体 |
| `v6` / `v7` / `v8` | 单组特征消融 |
| `v9` | 全量 + Compustat 基本面（52+5） |
| `v10` | v9 + 仅 50 轮（欠拟合对照） |
| `v11` | v9 + tradability weighted 训练样本 |
| `v12` | v9 + HPO 结果（depth=6, lr=0.08, subsample=0.8） |

预处理（均值填充、标准化、类别编码）的统计量只在 train 拟合，保存在 `TrainArtifacts` 中随模型一起加载。

## 回测口径

默认配置在 `configs/backtest.yaml`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `top_k` | 20 | 每日持仓数 |
| `holding_td` | 10 | 持有交易日数 |
| `cooldown_td` | 0 | 同一标的再次进入的冷却 |
| `stable_div_count_min` | 4 | 稳定分红组的历史分红次数门槛 |
| `stable_gap_cv_quantile` | 0.5 | 间隔 CV 的稳定分组分位 |
| `stable_prob_threshold` | 0.45 | 稳定组概率阈值 |
| `regular_prob_threshold` | 0.55 | 常规组概率阈值 |
| `dividend_rules_mode` | true | 使用分红历史规则分组阈值 |
| `min_price` | 3.0 | 价格下限 |
| `max_industry_weight` | 0.25 | 单行业权重上限 |
| `turnover_quantile_min` | 0.2 | 换手率分位下限 |
| `exclude_div_count_le` | 1 | 历史分红次数 ≤1 的剔除 |
| `cost_bps_one_way` | 10 bps | 单边固定成本 |
| `use_bid_ask_spread` | true | 使用真实买卖价差作为成本 |
| `spread_cost_cap_bps_one_way` | 100 bps | 价差成本封顶 |
| `benchmark` | equal_weight_universe | 基准 |

`configs/backtest_v9_optimized.yaml` 是 HPO 后的紧凑版本（`top_k=15, holding_td=7`）。`configs/versions/v12_cost_*_spr*_backtest.yaml` 是 v12 的成本敏感性套件，覆盖单边成本 2/5/10 bps × 价差封顶 off/25/50/100 的组合。

**执行时序**（核心不变量）：
1. 在 t 日收盘后用 ≤t 的信息生成概率。
2. 在 t+1 日开盘执行买入候选。
3. 持有 `holding_td` 个交易日。
4. 若 `DlyOpen` 可用则用 open-to-open 收益，否则回退到 close-to-close。

这保证信号、执行、收益在时间轴上无未来信息泄漏。`run_backtest.py` 的 `--reset_active_at` 用于在 walk-forward 和单期回测对比时把起点对齐。基准包括等权全域、随机打分、无概率、oracle 等，方便判断 alpha 来源。

## 输出产物

`outputs/runs/<run_id>/backtest/` 下典型文件：

- `summary.json`：汇总指标（年化、夏普、最大回撤、换手、平均持仓数、与基准对比等）
- `daily_portfolio.csv`：日度净值、基准、收益归因
- `trades.csv` / `positions.csv`：逐笔与逐日持仓
- 成本/换手归因、池/排名分解、top-k 收益诊断

`outputs/runs/<run_id>/eval/` 下包含事件预警指标表与按历史分红次数分层的诊断。

`outputs/analysis/` 下放跨 run 的对比产物，例如：

- `calibration_shap_50v200/`：reliability 曲线、ECE、Brier、SHAP 排名漂移、tradability cohort。
- 年度归因表与图（`analyze_yearly_attribution.py`）。

## 依赖

见 `pyproject.toml` 与 `requirements.txt`：

- 核心：`pandas`、`numpy`、`pyarrow`、`pyyaml`、`scikit-learn`、`xgboost`、`joblib`
- 模型扩展：`lightgbm`、`catboost`
- 分析/报告：`matplotlib`、`python-docx`
- 可选：`wrds`（仅 `fetch_compustat.py` 用），`shap`（分析脚本用）

```bash
pip install -r requirements.txt
```

## 测试

```bash
python -m unittest discover tests
```

`tests/test_backtest.py` 覆盖自动阈值选择、开盘价执行、候选过滤、行业约束、次日执行收益时序、买卖价差成本、基准一致性、walk-forward 面板隔离等关键不变量。`tests/test_build_features.py` 覆盖特征构建。

## 已知假设与局限

- 只做长仓，不做做空，不考虑融资融券限制。
- 涨跌停、停牌仅通过可交易性字段近似处理，未建模真实委托队列与冲击成本。
- 成本模型为「固定 bps + 价差 bps（封顶）」，未含冲击成本与流动性折扣。
- 标签为「10 交易日内是否有分红公告」，与实际投资目标之间存在转换损失（双 booster 是部分缓解）。
- 分红类型通过 `div_distcd` 白名单过滤，调整该名单会改变样本与标签。
- Compustat 字段为季报 PIT 合并，存在最长 90 天的信息陈旧度。

## 版本管理与可复现

1. 在 `configs/version_registry.yaml` 中登记新版本（特征组 + 超参 + 阈值策略）。
2. `python scripts/materialize_version_configs.py` 物化出 `configs/versions/<v>_model.yaml` 与 `<v>_backtest.yaml`。
3. 用 `run_train.py / run_predict.py / run_backtest.py` 配合 `--model_cfg` / `--cfg` 跑该版本，或用 `run_walk_forward.py` 做滚动重训。
4. `run_report_version_suite.py` + `render_version_suite_report.py` 输出横向对比报告，`report_targets` 中的目标值用于回归检查。
