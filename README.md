# 分红事件预测与回测系统

本仓库用于构建股票日频数据集、训练分红公告事件分类模型、评估事件预警质量，并把预测概率转换为可交易的日频选股信号进行组合回测。

当前代码已经针对主要的时序错误、规则选择泄漏和回测口径问题做过审计与修正。按照本文档的假设条件，这套系统适合做日频研究型回测。

## 项目在做什么

这个项目主要回答两个问题：

1. 能否预测某只股票在未来 `H=10` 个交易日内是否会发生分红公告事件？
2. 能否把这个事件预测结果转成一套可交易的选股策略，并用相对可靠的回测方式评估它？

完整流程如下：

1. 从原始事件数据和市场数据构建因果特征面板。
2. 生成标签 `y_div_10d`。
3. 按日期切分为 `train`、`val`、`test`。
4. 训练 XGBoost 二分类模型。
5. 对指定 split 生成预测概率。
6. 用阈值规则做事件预警评估。
7. 用概率信号驱动长仓 Top-K 组合回测。

## 仓库结构

```text
configs/              配置文件
  config.yaml         数据集构建配置
  model.yaml          特征列表与模型超参数
  paths.yaml          数据、模型、输出路径
  backtest.yaml       回测配置

data/
  raw/                原始数据，包括 tableA 和 tableB
  processed/          最终的 train/val/test parquet 数据

models/               模型产物：xgb_<run_id>.joblib
outputs/runs/         按 run_id 保存预测、评估、回测输出

scripts/
  run_build_dataset.py
  run_train.py
  run_predict.py
  run_eval.py
  run_backtest.py

src/
  data/               数据读取、特征工程、打标、切分
  modeling/           预处理、训练、预测
  eval/               事件预警评估与报表
  backtest/           信号构建、组合模拟、基准、报表
```

## 数据输入

路径配置定义在 [configs/paths.yaml](/D:/Users/Workspace/WQ-Project/code/configs/paths.yaml)：

```yaml
tableA_path: data/raw/tableA.csv
tableB_path: data/raw/tableB.csv
processed_dir: data/processed
models_dir: models
outputs_dir: outputs/runs
```

### `tableA`

`tableA` 是分红事件数据。当前数据集构建流程使用的关键事件日期是公告日 `DCLRDT`。

预期核心字段至少包括：

- `PERMNO`
- `DCLRDT`
- 配置中 `div_distcd` 需要用到的分红分类字段

### `tableB`

`tableB` 是股票日频行情数据。当前代码会直接或条件性使用以下字段：

- `PERMNO`
- `DlyCalDt`
- `Ticker`
- `SICCD`
- `DlyOpen`
- `DlyClose`
- `DlyHigh`
- `DlyLow`
- `DlyBid`
- `DlyAsk`
- `DlyPrc`
- `DlyVol`
- `DlyRet`
- `ShrOut`

其中 `DlyOpen`、`DlyBid`、`DlyAsk` 已经接入回测执行层。

## 运行环境

推荐安装以下 Python 依赖：

```bash
python -m pip install pandas numpy pyyaml pyarrow xgboost scikit-learn joblib
```

建议使用 parquet。部分脚本在无法写 parquet 时会退回 CSV，但 parquet 是默认输出格式。

## 端到端使用流程

### 1. 构建处理后的数据集

```bash
python scripts/run_build_dataset.py --cfg configs/config.yaml
```

这个脚本会完成：

1. 读取原始市场数据和分红事件数据。
2. 构建严格因果的历史特征。
3. 生成未来 `H_label` 个交易日内是否发生事件的标签。
4. 按日期切分成 `train.parquet`、`val.parquet`、`test.parquet` 并写入 `data/processed/`。

主要配置文件： [configs/config.yaml](/D:/Users/Workspace/WQ-Project/code/configs/config.yaml)

### 2. 训练模型

```bash
python scripts/run_train.py --run_id <run_id>
```

这个脚本会：

- 从 `data/processed/` 读取 `train/val/test`
- 只用 `train` 拟合缺失值填补和标准化
- 只用 `train` 对齐类别特征取值空间
- 训练基于 `y_div_10d` 的 XGBoost 二分类模型
- 把模型写入 `models/xgb_<run_id>.joblib`
- 创建 `outputs/runs/<run_id>/`

主要配置文件： [configs/model.yaml](/D:/Users/Workspace/WQ-Project/code/configs/model.yaml)

### 3. 生成指定 split 的预测结果

```bash
python scripts/run_predict.py --run_id <run_id> --split test
```

输出路径为：

- `outputs/runs/<run_id>/preds/test_preds.parquet`

预测表至少包含：

- `date`
- `permno`
- `prob`
- `y`
- 若存在则保留部分额外分组字段，例如 `turnover_5d`、`industry`、`has_div_history`

### 4. 做事件预警评估

```bash
python scripts/run_eval.py --run_id <run_id> --split test
```

这一层是事件评估，不是组合回测。它会根据“有无分红历史”使用不同阈值，评估预警命中率、误报、冷却期后的 alert 效果等。

这个脚本回答的是：模型是不是一个合格的“事件预警器”。

### 5. 运行组合回测

```bash
python scripts/run_backtest.py --run_id <run_id> --split test
```

这一层才是组合回测。它会把模型预测概率转换为每日候选股票清单，再模拟一个长仓 Top-K 组合。

主要配置文件： [configs/backtest.yaml](/D:/Users/Workspace/WQ-Project/code/configs/backtest.yaml)

## 当前特征集

训练特征定义在 [configs/model.yaml](/D:/Users/Workspace/WQ-Project/code/configs/model.yaml)。

当前数值特征主要包括：

- 市场和流动性特征：`ret_5d`、`ret_21d`、`vol_5d`、`vol_21d`、`turnover_5d`、`turnover_21d`
- 量价异常特征：`price_to_high`、`volume_spike`、`vol_ratio`、`turnover_ratio`
- 行业相对收益特征：`ret_rel_to_ind`
- 分红节奏与稳定性特征：`days_since_last_div`、`gap_mean_exp`、`gap_med_exp`、`gap_std_exp`、`gap_cv_exp`、`z_to_med_exp`、`div_count_exp`
- 日历编码特征：`weekday_sin`、`weekday_cos`、`month_sin`、`month_cos`、`doy_sin`、`doy_cos`

当前类别特征包括：

- `quarter`
- `weekday`
- `is_month_start`
- `is_month_end`

## 回测设计

当前回测是一套日频、长仓、Top-K、等权目标权重的选股框架。

### 策略定义

默认参数来自 [configs/backtest.yaml](/D:/Users/Workspace/WQ-Project/code/configs/backtest.yaml)：

- `top_k: 20`
- `holding_td: 10`
- `cooldown_td: 0`
- `turnover_quantile_min: 0.2`
- `exclude_div_count_le: 1`
- `cost_bps_one_way: 10.0`
- `benchmark: equal_weight_universe`
- `weighting: equal_weight`
- `min_price: 3.0`
- `max_industry_weight: 0.25`
- `stable_div_count_min: 4`
- `stable_gap_cv_quantile: 0.5`
- `stable_prob_threshold: 0.45`
- `regular_prob_threshold: 0.55`
- `dividend_rules_mode: auto`
- `use_bid_ask_spread: true`
- `spread_cost_cap_bps_one_way: 100.0`

### 信号构建逻辑

当前回测把分红信息分成三层使用：

1. 事件概率层
   - `prob` 是主排序分数。
2. 分红历史可信度层
   - 使用 `has_div_history`、`div_count_exp`、`gap_cv_exp`、`gap_med_exp`、`days_since_last_div`、`z_to_med_exp` 等字段。
3. 交易决策层
   - 候选过滤、分组阈值、排名、容量约束。

### 候选过滤与排序

当分红规则开启时，当前候选规则包括：

- 剔除低流动性股票，门槛为当日 `turnover_5d` 分位数过滤
- 剔除分红历史次数 `<= exclude_div_count_le` 的股票
- 按分红稳定性分成稳定组和一般组
- 对不同组使用不同 `prob` 入选阈值
- 按 `prob` 从高到低排序
- 每日最多保留 `top_k` 只股票
- 同时受 `max_industry_weight` 约束

如果分红规则关闭，则退化为更接近纯概率排序的框架，但仍然保留交易性过滤、组合容量和行业约束。

## 回测时序与反穿越规则

这一部分最重要。

### 信号与执行时序

当前实现采用以下时序：

1. 在 `t` 日收盘后，用 `t` 及以前可见的信息生成信号。
2. 在 `t+1` 日开盘执行。
3. 执行完成后才开始计入持仓收益。
4. 固定持有 `holding_td` 个交易日，除非样本结束而被截尾。

### 收益口径

当前回测优先使用 `open_to_open` 收益口径，只要 `tableB` 中存在 `DlyOpen`。

- 若有 `DlyOpen`，执行口径就是 `open_to_open`
- 若没有开盘价，则退回到更保守的 delayed close-to-close 近似

实际使用的执行口径会写入 `summary.json`。

### `auto` 规则选择的安全约束

`dividend_rules_mode: auto` 不会用当前正在回测的 split 来决定是否启用分红规则。

当前策略选择逻辑是：

- 回测 `test` 时，只允许使用更早的 held-out `val` 来决定 policy
- 回测 `val` 时，没有更早的 held-out split，所以 `auto` 会关闭分红规则
- 回测 `train` 时，同样没有更早的 held-out split，所以 `auto` 会关闭分红规则

这样可以避免“用当前测试集未来结果挑策略规则”的数据窥探。

### 稳定阈值参考口径

稳定分红阈值可以参考更早 split 的统计量，但“是否启用分红规则”这件事与当前 split 的结果是隔离的。

## 成本模型

回测成本模型会写入 `summary.json` 的 `cost_model` 字段。

当前成本模型包括：

- 固定单边交易成本 `cost_bps_one_way`
- 可选的日频 `Bid/Ask` 半价差成本
- 价差成本受 `spread_cost_cap_bps_one_way` 上限约束

当 `use_bid_ask_spread: true` 时，系统会把当日 `DlyBid` 和 `DlyAsk` 计算出的半价差作为单边成本近似。这比纯固定 bps 更真实，但仍然只是日频近似，不是逐笔成交模拟。

## 回测输出

运行 `scripts/run_backtest.py` 后，结果会写到：

- `outputs/runs/<run_id>/backtest/`

常见输出文件包括：

- `summary.json`
- `daily_portfolio.csv`
- `trades.csv`
- `positions.csv`
- `benchmark_daily.csv`
- `candidates.csv`
- `reports/prob_buckets.csv`
- `reports/dividend_groups.csv`
- `reports/industry.csv`
- 以及若干 research 诊断表

### 核心输出定义

`summary.json` 包含高层指标，例如：

- 回测天数
- 平均持仓数
- 平均换手率
- 入场时点的事件标签命中率
- 组合总收益、年化收益、波动、Sharpe、最大回撤、Calmar
- 相对基准的超额收益指标
- 基准最终净值
- 交易笔数、胜率、平均单笔收益、截尾比例
- `research_decision`
- `cost_model`

`daily_portfolio.csv` 是组合层面的逐日结果表。

`trades.csv` 是交易级别明细，包含入场、出场、实现收益和成本拆分。

`positions.csv` 是持仓级别逐日明细，包含 `weighted_ret`，因此可以直接用底层持仓表复算组合日收益。

## 当前可信度说明

当前回测逻辑已经修复过对研究有效性影响最大的几类问题。

已经落实到代码中的关键修复包括：

- 修复预测表与 split 合并时的重名列覆盖问题，避免 `turnover_5d` 等关键字段丢失
- 移除 `auto` 策略选择中的当前 split 泄漏
- 将 `auto` 的 policy 选择限制为历史 held-out 验证段
- 当 `DlyOpen` 可用时改为 `t+1` 开盘执行
- 修复执行日收益提前计入的问题
- 加入 `Bid/Ask` 半价差成本模型
- 增加底层审计字段，使日报表和交易表能由底层持仓记录复算

## 当前限制与假设

文档写的是“现在系统真实做到的部分”，同时也明确列出它还没有做到什么。

### 研究层假设

- 这是日频研究框架，不是实盘成交系统
- 当前只做长仓
- 当前按等权目标权重构建组合
- 当前不建模市场冲击和容量约束
- 当前不建模开盘集合竞价失败、涨跌停、停牌无法成交等细节
- 当前不做市场中性或因子中性优化

### 数据层假设

- `DlyOpen`、`DlyBid`、`DlyAsk` 被视为可用于日频执行近似的参考价
- 日频 `Bid/Ask` 只能近似价差成本，不能代表真实开盘瞬时盘口
- 样本末尾的交易可能因为未来观测不足而被截尾

### 建模层限制

当前模型标签仍然是 `y_div_10d`，这是“事件是否发生”的标签，不是“未来收益是否更高”的标签。

因此，一个事件分类器即使命中率很高，也不一定能产生正超额收益。这是研究结论问题，不一定是代码 bug。

## 建议的研究工作流

比较实用的一套研究流程是：

1. 构建数据集。
2. 训练模型。
3. 在 `val` 和 `test` 上生成预测结果。
4. 先用 `run_eval.py` 检查事件预警质量。
5. 再用 `run_backtest.py` 检查组合是否可交易、是否有 alpha。
6. 同时对比全市场等权基准和更细的分组归因报表。
7. 如果事件预测很好但超额收益很弱，应把它视为上游事件信号，再叠加收益模型或后处理过滤。

## 最小命令清单

```bash
python scripts/run_build_dataset.py --cfg configs/config.yaml
python scripts/run_train.py --run_id <run_id>
python scripts/run_predict.py --run_id <run_id> --split val
python scripts/run_predict.py --run_id <run_id> --split test
python scripts/run_eval.py --run_id <run_id> --split test
python scripts/run_backtest.py --run_id <run_id> --split test
```

## 测试

回测测试位于 [tests/test_backtest.py](/D:/Users/Workspace/WQ-Project/code/tests/test_backtest.py)。

运行方式：

```bash
python -m unittest tests.test_backtest
```

当前测试覆盖：

- policy 选择的时间顺序安全性
- 开盘价执行口径
- 候选过滤和行业约束
- 次日执行后的收益起算时点
- 价差成本会降低净收益
- 基准与 summary 输出一致性

## 关键文件

主入口脚本：

- [scripts/run_build_dataset.py](/D:/Users/Workspace/WQ-Project/code/scripts/run_build_dataset.py)
- [scripts/run_train.py](/D:/Users/Workspace/WQ-Project/code/scripts/run_train.py)
- [scripts/run_predict.py](/D:/Users/Workspace/WQ-Project/code/scripts/run_predict.py)
- [scripts/run_eval.py](/D:/Users/Workspace/WQ-Project/code/scripts/run_eval.py)
- [scripts/run_backtest.py](/D:/Users/Workspace/WQ-Project/code/scripts/run_backtest.py)

回测模块：

- [src/backtest/signal.py](/D:/Users/Workspace/WQ-Project/code/src/backtest/signal.py)
- [src/backtest/portfolio.py](/D:/Users/Workspace/WQ-Project/code/src/backtest/portfolio.py)
- [src/backtest/benchmark.py](/D:/Users/Workspace/WQ-Project/code/src/backtest/benchmark.py)
- [src/backtest/report.py](/D:/Users/Workspace/WQ-Project/code/src/backtest/report.py)

## 研究结论模板

下面这段模板可以直接用于每次回测后的研究结论记录。建议基于 `summary.json`、`trades.csv`、`daily_portfolio.csv` 和各类归因表来填写。

### 模板一：简版结论

```text
本次回测基于 run_id=<run_id>，测试区间为 <start_date> 至 <end_date>。

策略定义：日频、长仓、Top-K=<top_k>、持有期=<holding_td>、执行口径=<execution_basis>、成本模型=<cost_model>。

结果上，组合绝对收益为 <portfolio_total_return>，年化收益为 <portfolio_annualized_return>，Sharpe 为 <portfolio_sharpe>，最大回撤为 <portfolio_max_drawdown>。

相对基准，超额总收益为 <excess_total_return>，超额年化为 <excess_annualized_return>，超额 Sharpe 为 <excess_sharpe>。

交易层面，共发生 <n_trades> 笔交易，胜率为 <win_rate>，平均单笔收益为 <mean_trade_return>。

研究判断：
1. 该模型对分红事件的识别能力 <强/一般/较弱>。
2. 该策略相对基准的选股 alpha <存在/不明显/为负>。
3. 当前拖累收益的主要来源是 <行业暴露/高换手成本/弱分组/市场风格偏移/其他>。
4. 下一步应优先验证 <某个分组/某个过滤条件/收益标签重构/行业中性化/成本模型>。
```

### 模板二：标准研究结论

```text
一、回测设定
- run_id：<run_id>
- split：<split>
- 执行口径：<execution_basis>
- 成本模型：<cost_model>
- Top-K：<top_k>
- 持有期：<holding_td>
- 是否启用分红规则：<use_dividend_rules>
- 规则来源：<policy_source>

二、核心结果
- 组合总收益：<portfolio_total_return>
- 组合年化收益：<portfolio_annualized_return>
- 组合波动：<portfolio_annualized_vol>
- 组合 Sharpe：<portfolio_sharpe>
- 组合最大回撤：<portfolio_max_drawdown>
- 超额总收益：<excess_total_return>
- 超额年化收益：<excess_annualized_return>
- 超额 Sharpe：<excess_sharpe>
- 基准最终净值：<benchmark_final_nav>

三、交易特征
- 平均持仓数：<avg_positions>
- 平均换手率：<avg_turnover>
- 交易笔数：<n_trades>
- 胜率：<win_rate>
- 平均单笔收益：<mean_trade_return>
- 截尾比例：<truncated_rate>
- 入场标签命中率：<signal_hit_rate_on_entries>

四、规则诊断
- auto 模式结论：<research_recommended_dividend_rules>
- 支持票数：<support_votes>/<total_checks>
- history_filter：<true_or_false>
- gap_cv_filter：<true_or_false>
- z_phase_filter：<true_or_false>

五、结论判断
- 如果“事件命中高，但超额收益差”，应判断为：模型更像事件识别器，而不是收益预测器。
- 如果“绝对收益正，但超额收益负”，应判断为：策略赚钱，但跑输基准，不具备稳定 alpha。
- 如果“超额收益为正，但换手和回撤偏高”，应判断为：信号可能有效，但执行成本和风险约束仍需优化。
- 如果“分红规则在 held-out 验证段不成立”，应判断为：当前分红过滤规则不够稳健，只能作为切片分析，不应强行作为交易规则。

六、下一步建议
- 优先查看 `reports/prob_buckets.csv`，确认高概率组是否真的对应更高 forward return。
- 查看 `reports/dividend_groups.csv`，确认分红历史过滤是在提升收益还是只提升命中率。
- 查看 `reports/industry.csv`，识别是否是行业暴露拖累了超额收益。
- 若事件识别优于收益表现，优先新增收益目标，例如 `fwd_ret_10d` 或 `fwd_excess_ret_10d`。
```

### 模板三：一句话摘要

```text
这次模型对分红事件的识别能力 <较强/一般/较弱>，但策略相对基准的超额收益 <为正/不明显/为负>，因此当前更适合把它作为事件信号上游模块，而不是直接作为最终选股模型。
```

## 如何理解两个入口脚本

最后再强调一次：

- `run_eval.py` 衡量的是事件预警质量
- `run_backtest.py` 衡量的是组合投资表现

一个模型可以非常擅长识别“会不会分红”，但依然不能战胜市场基准。这是研究结论，不一定是实现错误。
