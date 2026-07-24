"""CVaR 组合样本外回测 —— 样本内优化 / 样本外验证

口径（避免未来函数）：
    1. 把交易日按时间中点切成 train / test 两段
    2. 只用 train 段收益矩阵求 CVaR 最优权重 w*
    3. 用 w* 在 test 段计算组合日收益，评估真实表现
       （权重完全不含 test 段信息，无未来函数）

对比基准：
    - 等权组合（1/N）：验证 CVaR 优化是否真的降低了尾部损失
    - benchmark 指数：验证组合相对宽基的超额收益 / 信息比

关键指标：样本外实现 CVaR@95% / VaR@95% / 年化收益 / 年化波动 /
        最大回撤 / Calmar / 相对 benchmark 超额年化 / 信息比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor import (
    BETA,
    TARGET_ANNUAL_RETURN,
    TRADING_DAYS,
    WEIGHT_UPPER_BOUND,
    build_return_matrix,
    build_scenario_matrix,
    solve_cvar_with_relax,
)
from data_loader import load_index_data, load_stock_data


# === 基础指标函数（纯函数，供 validate / report 复用） ==========================

def empirical_cvar(returns: np.ndarray, beta: float = BETA) -> float:
    """经验 CVaR（损失口径，正值代表损失）：最差 (1-β) 比例收益的平均损失

    CVaR_β = -E[ r | r ≤ VaR_β ]，其中 VaR_β 是收益的 (1-β) 分位。
    """
    if len(returns) == 0:
        return 0.0
    var_quantile = np.quantile(returns, 1.0 - beta)      # 收益的 5% 分位（负值）
    tail = returns[returns <= var_quantile]
    if len(tail) == 0:
        return float(-var_quantile)
    return float(-tail.mean())


def empirical_var(returns: np.ndarray, beta: float = BETA) -> float:
    """经验 VaR（损失口径）：收益 (1-β) 分位的相反数"""
    if len(returns) == 0:
        return 0.0
    return float(-np.quantile(returns, 1.0 - beta))


def annualized_return(returns: np.ndarray) -> float:
    """年化收益率（几何口径）"""
    if len(returns) == 0:
        return 0.0
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return cumulative
    return float((1.0 + cumulative) ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray) -> float:
    """年化波动率"""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    """最大回撤（负值，如 -0.15 = -15%）"""
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = curve / running_max - 1.0
    return float(dd.min())


def sharpe_ratio(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """夏普比率（年化）"""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_daily
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def information_ratio(port_returns: np.ndarray, bench_returns: np.ndarray) -> float:
    """信息比率（年化）：超额收益均值 / 跟踪误差"""
    n = min(len(port_returns), len(bench_returns))
    if n < 2:
        return 0.0
    active = port_returns[:n] - bench_returns[:n]
    te = np.std(active, ddof=1)
    if te == 0:
        return 0.0
    return float(np.mean(active) / te * np.sqrt(TRADING_DAYS))


def portfolio_daily_returns(ret: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """给定权重字典，计算组合日收益序列（index=date）"""
    w = np.array([weights.get(sym, 0.0) for sym in ret.columns])
    return pd.Series(ret.values @ w, index=ret.index, name="portfolio_return")


def split_train_test(ret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间中点切分 train / test（保序，避免未来函数）"""
    dates = ret.index.tolist()
    split = len(dates) // 2
    return ret.iloc[:split], ret.iloc[split:]


def run_backtest(
    stock_df: pd.DataFrame | None = None,
    index_df: pd.DataFrame | None = None,
    target_annual_return: float = TARGET_ANNUAL_RETURN,
    beta: float = BETA,
    weight_upper: float = WEIGHT_UPPER_BOUND,
) -> dict:
    """样本内优化 / 样本外验证回测

    Returns:
        指标 dict：CVaR/VaR/年化/波动/MDD/Calmar/超额/IR + 等权与 benchmark 对比。
    """
    if stock_df is None:
        stock_df = load_stock_data()

    ret = build_return_matrix(stock_df)
    train_ret, test_ret = split_train_test(ret)
    if len(train_ret) < 20 or len(test_ret) < 10:
        raise ValueError(f"样本不足：train={len(train_ret)}, test={len(test_ret)}")

    symbols = ret.columns.tolist()
    n_assets = len(symbols)

    # === 样本内（train）求 CVaR 最优权重 ===
    mu_train = train_ret.mean().values
    target_daily = (1.0 + target_annual_return) ** (1.0 / TRADING_DAYS) - 1.0
    scenario_train, evt_info = build_scenario_matrix(train_ret)

    opt, _ = solve_cvar_with_relax(scenario_train, mu_train, n_assets=n_assets,
                                   beta=beta, target_daily_return=target_daily,
                                   weight_upper=weight_upper)

    cvar_weights = dict(zip(symbols, opt["weights"]))
    equal_weights = {sym: 1.0 / n_assets for sym in symbols}   # 等权基准

    # === 样本外（test）验证 ===
    cvar_test_ret = portfolio_daily_returns(test_ret, cvar_weights).values
    eq_test_ret = portfolio_daily_returns(test_ret, equal_weights).values

    # benchmark 指数在 test 段的日收益（对齐 test 交易日）
    bench_test_ret = None
    if index_df is not None and not index_df.empty:
        bench_ret = _index_daily_returns(index_df)
        bench_test_ret = bench_ret.reindex(test_ret.index).dropna().values

    def _metrics(r: np.ndarray, tag: str) -> dict:
        return {
            f"{tag}_CVaR95": round(empirical_cvar(r, beta), 6),
            f"{tag}_VaR95": round(empirical_var(r, beta), 6),
            f"{tag}_年化收益": round(annualized_return(r), 6),
            f"{tag}_年化波动": round(annualized_vol(r), 6),
            f"{tag}_最大回撤": round(max_drawdown(r), 6),
            f"{tag}_Calmar": round(annualized_return(r) / abs(max_drawdown(r)), 6) if max_drawdown(r) != 0 else 0.0,
            f"{tag}_夏普": round(sharpe_ratio(r), 6),
        }

    metrics: dict = {}
    metrics.update(_metrics(cvar_test_ret, "CVaR组合"))
    metrics.update(_metrics(eq_test_ret, "等权组合"))

    # 相对 benchmark 的超额与信息比
    if bench_test_ret is not None and len(bench_test_ret) > 1:
        metrics.update(_metrics(bench_test_ret, "Benchmark"))
        metrics["超额年化(CVaR-Bench)"] = round(
            annualized_return(cvar_test_ret) - annualized_return(bench_test_ret[:len(cvar_test_ret)]), 6
        )
        metrics["信息比(CVaR vs Bench)"] = round(
            information_ratio(cvar_test_ret, bench_test_ret), 6
        )

    # CVaR 优化相对等权的尾部改善（负值 = CVaR 组合尾部损失更小，符合预期）
    metrics["尾部改善(CVaR - 等权 的 CVaR95)"] = round(
        empirical_cvar(cvar_test_ret, beta) - empirical_cvar(eq_test_ret, beta), 6
    )

    metrics.update({
        "样本内 CVaR95(优化目标值)": round(opt["cvar_95"], 6),
        "样本内 VaR95": round(opt["var_95"], 6),
        "train 交易日数": len(train_ret),
        "test 交易日数": len(test_ret),
        "资产数": n_assets,
        "非零权重资产数": int((opt["weights"] > 1e-6).sum()),
        "最大单资产权重": round(float(opt["weights"].max()), 6),
        "目标年化收益": target_annual_return,
        "beta": beta,
        "EVT补样资产": evt_info,
        "评估口径": (
            f"train 段（前半样本）求 CVaR@{beta:.0%} 最优权重，test 段（后半样本）用固定权重验证，"
            f"权重不含 test 信息，无未来函数。CVaR/VaR 为损失口径正值。对比等权(1/N)与 benchmark 指数。"
        ),
    })
    return metrics


def _index_daily_returns(index_df: pd.DataFrame) -> pd.Series:
    """指数长表 → 日收益 Series（index=date）"""
    s = index_df.sort_values("date").set_index("date")["close"]
    ret = s.pct_change().dropna()
    ret.name = "benchmark_return"
    return ret


if __name__ == "__main__":
    # 联网模式：同时加载股票和指数
    stock = load_stock_data()
    symbols_dates = stock["date"].astype(str)
    start_8, end_8 = symbols_dates.min(), symbols_dates.max()
    start = f"{start_8[:4]}-{start_8[4:6]}-{start_8[6:8]}"
    end = f"{end_8[:4]}-{end_8[4:6]}-{end_8[6:8]}"
    try:
        index = load_index_data(start_date=start, end_date=end)
    except Exception as e:
        print(f"[WARN] 指数数据加载失败，跳过 benchmark 对比: {e}")
        index = None

    metrics = run_backtest(stock_df=stock, index_df=index)
    print("\n=== CVaR 组合样本外回测结果 ===")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {value if value else '(空)'}")
        else:
            print(f"{key}: {value}")
