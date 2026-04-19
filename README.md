# 分红事件预测与回测系统

本仓库用于构建股票日频数据集、训练分红公告事件分类模型、评估事件预警质量，并把预测概率转换为可交易的日频选股信号进行组合回测。代码针对时序泄漏、规则选择偏差、回测口径等问题做过审计与修正，适合做日频研究型回测。

## 项目在做什么

项目回答两个问题：

1. 能否预测某只股票在未来 `H=10` 个交易日内是否会发生分红公告事件？
2. 能否把这个事件预测转成一套可交易的日频长仓 Top-K 策略，并在考虑成本、执行滞后、停牌等真实摩擦下评估其表现？

两个评估层次分离：事件预警质量由 `run_eval.py` 度量（精度、召回、误报、冷却期等）；组合表现由 `run_backtest.py` 度量（日度净值、成本、换手、相对基准超额等）。

## 仓库结构

```text
configs/                      配置文件
  config.yaml                 数据集构建：输入路径、标签 H、分红代码、切分日期、embargo
  model.yaml                  XGBoost 特征列表与超参数
  backtest.yaml               回测参数：top_k、持有期、阈值、成本、基准
  paths.yaml                  输出目录
  version_registry.yaml       实验/版本元数据

data/
  raw/                        tableA（分红事件）、tableB（日频行情与基本面）
  processed/                  train/val/test 的 parquet 切分

models/                       训练产物：xgb_<run_id>.joblib 等
outputs/runs/<run_id>/
  preds/                      {train,val,test}_preds.parquet
  eval/                       事件预警评估结果
  backtest/                   daily_portfolio.csv、trades.csv、positions.csv、summary.json 等

scripts/                      所有可执行入口（见下节）
src/
  data/          load、build_features、label、split
  modeling/      preprocess、train、predict（xgb/catboost/lgbm/lr）
  backtest/      signal、portfolio、benchmark、report
  eval/          eval_tools、report、plots
  analysis/      backtest_analysis、plots（score bucket、池/排名分解等）
  experiments/   versioning
  utils/         paths、logging

tests/           test_backtest.py、test_build_features.py
notebooks/       探索性 notebook
reports/         静态报告
```

## 数据输入

- `tableA`：分红事件表，使用 `DCLRDT` 作为公告日。通过 `config.yaml` 中 `div_distcd` 过滤有效分红类型（1202/1212/…/1272）。
- `tableB`：日频行情与基本面，字段包括 OHLC、成交量、买卖盘、基本面等。
- `start_all`~`end_all`（默认 2010-01-01 ~ 2024-12-31）定义全样本范围。

切分与 embargo 定义在 `config.yaml` 的 `split` 字段与 `embargo_td`（默认 10 交易日）。embargo 防止训练特征跨入标签窗口，避免未来信息泄漏。

## 完整流程

最小端到端流程：

```bash
# 1. 读取原始表、构建特征面板与标签、切分
python scripts/run_build_dataset.py --cfg configs/config.yaml

# 2. 训练 XGBoost 二分类
python scripts/run_train.py --run_id my_run

# 3. 对每个 split 生成预测概率
python scripts/run_predict.py --run_id my_run --split val
python scripts/run_predict.py --run_id my_run --split test

# 4. 事件预警质量评估
python scripts/run_eval.py --run_id my_run --split test

# 5. 组合回测
python scripts/run_backtest.py --run_id my_run --split test
```

每个脚本都以 `run_id` 为组织单位，产物一律落在 `outputs/runs/<run_id>/` 下，方便并行多版本实验。

## 脚本清单

核心管线：

| 脚本 | 作用 |
| --- | --- |
| `run_build_dataset.py` | 从 tableA/B 构建因果特征面板，生成 `y_div_10d` 标签，切分 train/val/test |
| `run_train.py` | 训练 XGBoost；在 train 上拟合 impute/scale，val 上早停；保存 booster 与预处理统计量 |
| `run_predict.py` | 加载模型对指定 split 生成预测 parquet |
| `run_eval.py` | 基于阈值/冷却期规则评估事件预警（精度、召回、误报、按分红历史分层等） |
| `run_backtest.py` | 组合回测，输出日度净值、交易、持仓、成本/换手归因与诊断表 |

替代模型与基线：

| 脚本 | 作用 |
| --- | --- |
| `run_train_catboost.py` | CatBoost 训练 |
| `run_train_lgbm.py` | LightGBM 训练 |
| `run_train_lr.py` | Logistic Regression（含多项式特征选项） |
| `run_random_predict.py` | 随机分数零假设基线 |

分析与诊断：

| 脚本 | 作用 |
| --- | --- |
| `run_standardized_pipeline.py` | 固定配置的端到端一键流水线 |
| `run_ablation.py` | 特征消融 |
| `run_pool_rank_decomposition.py` | 把回测收益分解为「候选池效应」vs「排名效应」 |
| `run_rounds_scan.py` | 训练轮数扫描 |
| `run_version_flip_analysis.py` | 跨版本预测翻转分析 |
| `analyze_feature_importance.py` | 特征重要性导出 |

报告生成：

| 脚本 | 作用 |
| --- | --- |
| `materialize_version_configs.py` | 为版本套件物化配置文件 |
| `run_report_version_suite.py` / `render_version_suite_report.py` | 版本套件对比报告 |
| `generate_today_weekly_report_v3.py` | 周报 docx 生成 |

## 特征与模型

特征构建在 `src/data/build_features.py`，涵盖：

- **日历特征**：月、周、星期（含 sin/cos 编码）。
- **分红历史**：历次间隔统计（均值、波动、z-score）、累计次数、距上次分红天数等。
- **价格/动量/波动率**：多窗口收益、ATR、波动率、相对收益。
- **市场微结构**：买卖价差、成交量、换手率、open gap 等。
- **事件窗口信息**：`decl_to_exdt`、`div_amt` 等。

特征列表与模型超参数集中在 `configs/model.yaml`。XGBoost 为主模型，`modeling/` 下也提供 CatBoost、LightGBM、LR 的对照实现。预处理（均值填充、标准化、类别编码）的统计量只在 train 拟合，保存在 `TrainArtifacts` 中随模型一起加载。

## 回测口径

配置在 `configs/backtest.yaml`：

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
| `cost_bps_one_way` | 10 bps | 单边固定成本 |
| `use_bid_ask_spread` | true | 用真实买卖价差作为成本（封顶 100 bps） |
| `benchmark` | equal_weight_universe | 基准 |

**执行时序**（关键）：
1. 在 t 日收盘后用 ≤t 的信息生成概率。
2. 在 t+1 日开盘执行买入候选。
3. 持有 `holding_td` 个交易日。
4. 若 `DlyOpen` 可用用 open-to-open 收益，否则回退到 close-to-close。

这保证信号、执行、收益在时间轴上无未来信息泄漏。基准包括等权全域、随机打分、无概率、oracle 等，方便判断 alpha 来源。

## 输出产物

`outputs/runs/<run_id>/backtest/` 下典型文件：

- `summary.json`：汇总指标（年化、夏普、最大回撤、换手、平均持仓数、与基准对比等）
- `daily_portfolio.csv`：日度净值、基准、收益归因
- `trades.csv` / `positions.csv`：逐笔与逐日持仓
- 成本/换手归因表、池/排名分解表、top-k 收益诊断表

`outputs/runs/<run_id>/eval/` 下包含事件预警指标表与按历史分红次数分层的诊断。

## 依赖

见 `pyproject.toml` 与 `requirements.txt`：

- 核心：`pandas`、`numpy`、`pyarrow`、`pyyaml`、`scikit-learn`、`xgboost`、`joblib`
- 可选：`lightgbm`、`catboost`、`matplotlib`、`python-docx`（替代模型与报告生成）

```bash
pip install -r requirements.txt
```

## 测试

```bash
python -m unittest discover tests
```

`tests/test_backtest.py` 覆盖自动阈值选择、开盘价执行、候选过滤、行业约束、次日执行收益时序、买卖价差成本、基准一致性等关键不变量。`tests/test_build_features.py` 覆盖特征构建。

## 已知假设与局限

- 只做长仓，不做做空，不考虑融资融券限制。
- 涨跌停、停牌仅通过可交易性字段近似处理，未建模真实委托队列。
- 成本模型为「固定 bps + 价差 bps（封顶）」，未包含冲击成本。
- 标签为「10 交易日内是否有分红公告」，与实际投资目标之间存在转换损失。
- 分红类型通过 `div_distcd` 白名单过滤，调整该名单会改变样本与标签。

## 版本管理

同一份代码可以通过不同 `run_id`、不同 `configs/*.yaml` 组合生成多个实验。`src/experiments/versioning.py` 与 `configs/version_registry.yaml` 负责登记；`materialize_version_configs.py` 与 `render_version_suite_report.py` 支持批量物化配置与横向对比报告。
