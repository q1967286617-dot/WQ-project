"""
四因子回归：Fama-French 3 因子 + 动量因子
用途：检验 v12_run1 和 wf_v12_cd3 的日度超额收益中是否存在因子调整后的 alpha
运行：python scripts/run_factor_regression.py
输出：outputs/factor_regression/
"""

import io
import zipfile
import urllib.request
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "factor_data"
OUT_DIR = ROOT / "outputs" / "factor_regression"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = {
    "v12_run1": {
        "path": ROOT / "outputs/runs/v12_run1/backtest/daily_portfolio.csv",
        "label": "v12 单训（2015-2024）",
    },
    "wf_v12_cd3": {
        "path": ROOT / "outputs/runs/wf_v12_cd3/backtest/daily_portfolio.csv",
        "label": "WF+cd3（2022-2024）",
    },
}

# ─────────────────────────────────────────────
# 1. 下载 / 读取 Ken French 因子数据
# ─────────────────────────────────────────────
FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def fetch_ff_csv(url: str, cache_path: Path) -> pd.DataFrame:
    """下载并解析 Ken French 因子 CSV（自动处理描述行）"""
    if cache_path.exists():
        print(f"  [cache] {cache_path.name}")
        raw = cache_path.read_bytes()
    else:
        print(f"  [download] {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
        cache_path.write_bytes(raw)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        text = zf.read(csv_name).decode("utf-8", errors="ignore")

    # Ken French CSV：前 N 行是描述文字，找到第一个全数字开头的行作为数据起点
    lines = text.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) >= 2 and parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
            data_start = i
            break

    # 找 header 行（数据起点的前一行有列名）
    header_line = lines[data_start - 1]
    col_names = ["date"] + [c.strip() for c in header_line.split(",")[1:] if c.strip()]

    # 找数据结束行（空行或非数字起头）
    data_lines = []
    for line in lines[data_start:]:
        parts = line.split(",")
        if parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
            data_lines.append(line)
        elif data_lines:
            break  # 遇到第一个非数据行就停（尾部是年度数据，不要）

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=col_names[:len(data_lines[0].split(","))],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    # 因子单位：% → 小数
    df = df.astype(float) / 100
    return df


print("=== 获取因子数据 ===")
ff3 = fetch_ff_csv(FF3_URL, DATA_DIR / "ff3_daily.zip")
mom = fetch_ff_csv(MOM_URL, DATA_DIR / "mom_daily.zip")
print(f"  FF3 因子：{ff3.index[0].date()} ~ {ff3.index[-1].date()}，{len(ff3)} 天")
print(f"  动量因子：{mom.index[0].date()} ~ {mom.index[-1].date()}，{len(mom)} 天")

# 合并四因子
factors = ff3.join(mom, how="inner")
# 统一列名
factors.columns = [c.replace("Mkt-RF", "MKT").replace("Mom", "MOM") for c in factors.columns]
print(f"  合并后列名：{list(factors.columns)}")


# ─────────────────────────────────────────────
# 2. 回归函数
# ─────────────────────────────────────────────
def run_regression(excess_ret: pd.Series, factor_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    对 excess_ret（策略净日收益 - RF）在若干因子模型下做 OLS 回归。
    使用 Newey-West HAC 标准误（lag=10），控制日度序列自相关。
    返回各模型的回归摘要 DataFrame。
    """
    models = {
        "CAPM":   ["MKT"],
        "FF3":    ["MKT", "SMB", "HML"],
        "FF4":    ["MKT", "SMB", "HML", "MOM"],
    }

    records = []
    for model_name, factors_used in models.items():
        X = factor_df[factors_used].copy()
        X.insert(0, "const", 1.0)
        y = excess_ret

        # 对齐
        data = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(data) < 30:
            print(f"  [{label}] {model_name}: 样本不足，跳过")
            continue

        res = sm.OLS(data["y"], data[X.columns]).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10}
        )

        alpha_daily = res.params["const"]
        alpha_annual = (1 + alpha_daily) ** 252 - 1
        t_alpha = res.tvalues["const"]
        p_alpha = res.pvalues["const"]
        r2 = res.rsquared

        row = {
            "策略": label,
            "模型": model_name,
            "因子数": len(factors_used),
            "观测天数": len(data),
            "α日度": f"{alpha_daily*100:.4f}%",
            "α年化": f"{alpha_annual*100:.2f}%",
            "α t值": f"{t_alpha:.2f}",
            "α p值": f"{p_alpha:.3f}",
            "R²": f"{r2:.3f}",
        }
        # 各因子系数
        for f in factors_used:
            row[f"β_{f}"] = f"{res.params[f]:.3f}"
            row[f"t_{f}"] = f"{res.tvalues[f]:.2f}"

        records.append(row)

        sig = "***" if p_alpha < 0.01 else ("**" if p_alpha < 0.05 else ("*" if p_alpha < 0.1 else ""))
        print(
            f"  [{label}] {model_name:4s}  "
            f"α年化={alpha_annual*100:+.2f}%  t={t_alpha:.2f}  p={p_alpha:.3f}{sig}  R²={r2:.3f}"
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 3. 对每个策略运行回归
# ─────────────────────────────────────────────
all_results = []

for run_name, cfg in STRATEGIES.items():
    print(f"\n=== {cfg['label']} ===")
    daily = pd.read_csv(cfg["path"], parse_dates=["date"]).set_index("date")

    # 策略净日收益（已扣成本）
    port_ret = daily["portfolio_ret"]

    # 对齐因子数据
    merged = factors.join(port_ret.rename("port"), how="inner")
    merged = merged.loc[merged.index.isin(port_ret.index)]

    # 超额收益 = 策略净收益 - 无风险利率
    merged["excess"] = merged["port"] - merged["RF"]

    df_result = run_regression(merged["excess"], merged[["MKT", "SMB", "HML", "MOM"]], cfg["label"])
    all_results.append(df_result)

# ─────────────────────────────────────────────
# 4. 汇总输出
# ─────────────────────────────────────────────
final = pd.concat(all_results, ignore_index=True)

# 保存 CSV
csv_path = OUT_DIR / "factor_regression_results.csv"
final.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n✓ 结果已保存至：{csv_path}")

# 打印汇总表
print("\n" + "=" * 80)
print("【四因子回归汇总】Newey-West HAC 标准误（lag=10），显著性：* p<0.1  ** p<0.05  *** p<0.01")
print("=" * 80)
display_cols = ["策略", "模型", "观测天数", "α年化", "α t值", "α p值", "R²",
                "β_MKT", "β_SMB", "β_HML", "β_MOM"]
display_cols = [c for c in display_cols if c in final.columns]
print(final[display_cols].to_string(index=False))

# 保存解读说明
note_path = OUT_DIR / "interpretation_note.txt"
note_path.write_text(
    """四因子回归解读说明
===================
回归模型：(portfolio_ret - RF) = α + β_MKT*(Mkt-RF) + β_SMB*SMB + β_HML*HML + β_MOM*MOM + ε

数据来源：Ken French Data Library（日度，自动下载）
标准误：Newey-West HAC（lag=10），控制日度收益序列自相关，比 OLS 标准误更保守
RF：Ken French 提供的每日无风险利率（近似 1 个月国债）

结果解读：
- α年化 > 0 且 p < 0.05：控制四类系统风险后仍有统计显著超额，是最强的"独立 alpha"证据
- α年化 > 0 但 p > 0.1：超额存在但统计意义不足（样本偏短时常见）
- β_SMB > 0：策略偏向小盘股
- β_HML > 0：策略偏向价值股（高账面市值比）
- β_MOM > 0：策略具有动量特征
- R² 越高：策略收益越多被系统因子解释（纯 alpha 成分越少）

注意：
- v12_run1 测试期 2015-2024（约 2500 天），统计功效较强
- wf_v12_cd3 测试期 2022-2024（约 760 天），样本较短，α 置信度有限
- 该回归使用等权候选池基准的超额，而非市场超额，两者在 β_MKT 上会有差异
""",
    encoding="utf-8",
)
print(f"✓ 解读说明已保存至：{note_path}")
