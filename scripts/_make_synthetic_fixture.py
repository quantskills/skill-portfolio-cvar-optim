"""合成 fixture（无需联网，供离线自举与边缘测试）

⚠️⚠️⚠️  覆盖警告  ⚠️⚠️⚠️
本脚本写入 sample_stocks.parquet / sample_index.parquet，与 save_fixture.py（联网拉真实数据）
【同名】。cvar 入 git 的这两个 fixture 是【真实数据】（4860 行），运行本生成器会覆盖它们。

用途：无凭证自举 / 构造可控边缘数据（低均值使 500% 目标必然触发 target_relaxed、
      短样本切片触发 EVT GPD 补样）。用完务必还原真实数据，切勿 commit 合成版：
    git checkout scripts/fixtures/sample_stocks.parquet scripts/fixtures/sample_index.parquet

数据模型：单因子市场模型（20 只无关股票适合，区别于 risk-parity 的多资产类同标的配对）
    r_i = μ + β_i·σ_m·z_m + σ_idio_i·ε_i          （z_m 公共市场因子，β_i∈[0.7,1.2] 分散）
    指数 000300.SH ≈ 市场因子本身（β≈1，低特质波动）
固定 RandomState(42) 保证可复现。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# cvar DEFAULT_STOCK_POOL（与 data_loader.py 一致，hardcode 保持脚本自包含、不触发 SDK import）
STOCKS = [
    "600519.SH", "601318.SH", "600036.SH", "000858.SZ", "601166.SH",
    "600276.SH", "000333.SZ", "600900.SH", "601888.SH", "002594.SZ",
    "600030.SH", "000651.SZ", "601012.SH", "600887.SH", "600309.SH",
    "000001.SZ", "601899.SH", "600028.SH", "601088.SH", "600585.SH",
]
BENCHMARK = "000300.SH"   # 沪深300 指数（cvar DEFAULT_BENCHMARK）

T = 260                   # 交易日数（够 CVaR 尾部估计 + split_train_test）
MU_DAILY = 0.0003         # 日均值（年化 ~7.5%，使 _edge_test【1】的 500% 目标必然触发 target_relaxed）
SIGMA_M = 0.012           # 公共市场因子日波动


def _trade_dates(start: date, n: int) -> list[str]:
    """从 start 起的 n 个交易日（跳过周末），返回 YYYYMMDD 字符串列表"""
    out, d, cnt = [], start, 0
    while cnt < n:
        if d.weekday() < 5:                # 周一~周五
            out.append(d.strftime("%Y%m%d"))
            cnt += 1
        d += timedelta(days=1)
    return out


def main() -> None:
    rs = np.random.RandomState(42)
    n_stock = len(STOCKS)

    # 每只股票的市场敏感度 β 与特质波动（RandomState 可复现）
    betas = rs.uniform(0.7, 1.2, n_stock)             # 市场敏感度分散
    sigmas_idio = rs.uniform(0.010, 0.016, n_stock)   # 特质波动

    # 公共市场因子序列（股票间相关性来源于此，约 0.3~0.5 正相关）
    z_m = rs.randn(T)

    def _gen_returns(beta: float, sigma_idio: float) -> np.ndarray:
        """单因子模型：r = μ + β·σ_m·z_m + σ_idio·ε"""
        eps = rs.randn(T)
        return MU_DAILY + beta * SIGMA_M * z_m + sigma_idio * eps

    dates = _trade_dates(date(2025, 7, 21), T)

    # 20 只股票：收益 → 累积价格（起点 100）→ 长表
    rows = []
    for i, sym in enumerate(STOCKS):
        ret = _gen_returns(betas[i], sigmas_idio[i])
        price = np.empty(T)
        price[0] = 100.0
        for t in range(1, T):
            price[t] = price[t - 1] * (1.0 + ret[t])
        for t in range(T):
            rows.append({"date": dates[t], "symbol": sym, "close": round(float(price[t]), 4)})
    stock_df = pd.DataFrame(rows, columns=["date", "symbol", "close"])

    # 指数 000300.SH：β≈1 跟随市场、低特质波动
    idx_ret = MU_DAILY + 1.0 * SIGMA_M * z_m + 0.003 * rs.randn(T)
    idx_price = np.empty(T)
    idx_price[0] = 100.0
    for t in range(1, T):
        idx_price[t] = idx_price[t - 1] * (1.0 + idx_ret[t])
    index_df = pd.DataFrame({
        "date": dates,
        "symbol": BENCHMARK,
        "close": np.round(idx_price, 4),
    })

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    stock_df.to_parquet(FIXTURE_DIR / "sample_stocks.parquet", index=False)
    index_df.to_parquet(FIXTURE_DIR / "sample_index.parquet", index=False)

    print(f"[OK] 合成 fixture 已生成：{len(stock_df)} 行股票, {n_stock} 资产, {T} 交易日")
    print(f"     股票 β: {[round(float(b), 2) for b in betas]}")
    print(f"     → {FIXTURE_DIR / 'sample_stocks.parquet'}")
    print(f"     → {FIXTURE_DIR / 'sample_index.parquet'}")
    print("⚠️  这是合成数据，已覆盖真实 fixture！用完请还原：")
    print("    git checkout scripts/fixtures/sample_stocks.parquet scripts/fixtures/sample_index.parquet")


if __name__ == "__main__":
    main()
