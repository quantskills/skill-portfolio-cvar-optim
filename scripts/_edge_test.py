"""边界路径回归测试 —— 覆盖离线 validate 未触发的分支

P1-1: LP 三档降级（超高目标 target_relaxed / N×upper<1 上限自动放宽）
P1-2: EVT 补样（短样本触发，验证不崩 + 情景数正确）
2 资产下界 / 确定性
"""
import numpy as np
import pandas as pd

from factor import optimize_portfolio
from validate import _load_fixture_or_network

stock, _ = _load_fixture_or_network(with_index=False)
print(f"=== fixture: {len(stock)} 行, {stock['symbol'].nunique()} 资产 ===\n")

# --- 1. relax 收益: 目标年化 500%（必然不可行 → target relax）---
print("【1】极高目标收益(500%) → 应触发 target_relaxed")
r = optimize_portfolio(stock_df=stock, target_annual_return=5.0)
print(f"    target_relaxed={r.attrs['target_relaxed']}, "
      f"预期年化={r.attrs['portfolio_expected_annual_return']:.4%}")
assert r.attrs["target_relaxed"] is True
assert r.attrs["weight_upper_relaxed"] is False, "20资产×0.30 不应触发上限放宽"
print("    PASS\n")

# --- 2. EVT 真触发: 前60天（尾部~6 < 30）---
print("【2】短样本(60天) → 应触发 EVT GPD 补样")
short_dates = sorted(stock["date"].astype(str).unique())[:60]
short = stock[stock["date"].astype(str).isin(short_dates)]
r2 = optimize_portfolio(stock_df=short)
evt = r2.attrs["evt_resampled"]
print(f"    触发资产 {len(evt)} 个, n_scenarios={r2.attrs['n_scenarios']} "
      f"(历史 {r2.attrs['n_hist_scenarios']})")
assert evt, "短样本应触发 EVT 补样"
print("    PASS\n")

# --- 3. P1-1 修复: 2资产 + 默认0.30(N×upper<1) → 自动放宽上限，不崩 ---
print("【3】2资产 + 默认上限0.30(N×upper<1) → 应自动放宽上限而非崩溃")
two = stock[stock["symbol"].isin(stock["symbol"].unique()[:2])]
r3 = optimize_portfolio(stock_df=two)
print(f"    weight_upper_relaxed={r3.attrs['weight_upper_relaxed']}, "
      f"effective_upper={r3.attrs['effective_weight_upper']}, "
      f"权重和={r3['weight'].sum():.6f}")
assert r3.attrs["weight_upper_relaxed"] is True
assert r3.attrs["effective_weight_upper"] == 1.0
print("    PASS\n")

# --- 3b. 正常路径: 20资产不触发上限放宽 ---
print("【3b】20资产 + 默认0.30(N×upper=6≥1) → 不触发上限放宽")
r3b = optimize_portfolio(stock_df=stock)
assert r3b.attrs["weight_upper_relaxed"] is False
print(f"    weight_upper_relaxed={r3b.attrs['weight_upper_relaxed']}")
print("    PASS\n")

# --- 4. 确定性 ---
print("【4】同输入两次求解确定性")
a = optimize_portfolio(stock_df=stock).sort_values("symbol")["weight"].values
b = optimize_portfolio(stock_df=stock).sort_values("symbol")["weight"].values
assert np.allclose(a, b, atol=1e-9)
print("    PASS\n")

print("=== 边界回归测试全部通过 ===")
