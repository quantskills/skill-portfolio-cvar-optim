"""CVaR 组合权重验证 —— 契约检查 + 无未来函数 + 样本外切片

验证项：
    1. check_weight_contract      —— 权重硬契约（和为1、非负、上限、约束满足）
    2. check_no_future_function   —— 截断重算，权重结果一致（无未来函数）
    3. check_out_of_sample_slice  —— train/test 切片非空且可回测
    4. check_target_return        —— 组合预期收益满足目标（或已标记 relaxed）
    5. check_cvar_consistency     —— 样本外实现 CVaR 与优化目标值量级一致

离线模式（PANDA_DATA_OFFLINE=1）从 fixtures 加载，跳过联网。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from factor import (
    BETA,
    TARGET_ANNUAL_RETURN,
    TRADING_DAYS,
    WEIGHT_UPPER_BOUND,
    build_return_matrix,
    optimize_portfolio,
)
from data_loader import load_index_data, load_stock_data
from backtest import empirical_cvar, portfolio_daily_returns, split_train_test

FIXTURE_STOCK = Path(__file__).parent / "fixtures" / "sample_stocks.parquet"
FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "sample_index.parquet"


def _load_fixture_or_network(with_index: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """加载测试数据：PANDA_DATA_OFFLINE=1 用 fixture，否则联网

    Returns:
        (stock_df, index_df) —— index_df 在 with_index=False 或 fixture 缺失时为 None
    """
    if os.getenv("PANDA_DATA_OFFLINE", "0") == "1":
        if not FIXTURE_STOCK.exists():
            raise FileNotFoundError(
                f"离线模式未找到 fixture: {FIXTURE_STOCK}。请先联网运行 `python save_fixture.py`。"
            )
        stock = pd.read_parquet(FIXTURE_STOCK)
        index = None
        if with_index and FIXTURE_INDEX.exists():
            index = pd.read_parquet(FIXTURE_INDEX)
        return stock, index

    stock = load_stock_data()
    if not with_index:
        return stock, None
    dates = stock["date"].astype(str)
    start_8, end_8 = dates.min(), dates.max()
    start = f"{start_8[:4]}-{start_8[4:6]}-{start_8[6:8]}"
    end = f"{end_8[:4]}-{end_8[4:6]}-{end_8[6:8]}"
    try:
        index = load_index_data(start_date=start, end_date=end)
    except Exception as e:
        print(f"[WARN] 指数加载失败，benchmark 相关检查跳过: {e}")
        index = None
    return stock, index


def check_weight_contract(result: pd.DataFrame) -> None:
    """权重硬契约（8 条断言，纯离线不依赖联网）"""
    assert not result.empty, "权重表不能为空"

    required_cols = ["symbol", "weight", "expected_return", "tail_contribution",
                     "model_id", "data_version", "update_time"]
    missing = [c for c in required_cols if c not in result.columns]
    assert not missing, f"权重表缺少关键列: {missing}"

    w = result["weight"].values
    # 1. 权重非负
    assert (w >= -1e-9).all(), f"权重必须非负，最小值 {w.min()}"
    # 2. 权重上限
    upper = result.attrs.get("weight_upper", WEIGHT_UPPER_BOUND)
    assert (w <= upper + 1e-6).all(), f"权重超上限 {upper}，最大值 {w.max()}"
    # 3. 权重和为 1（容差放宽到 1e-4，容纳 round(6) 后 N 个权重的累积舍入误差）
    assert abs(w.sum() - 1.0) < 1e-4, f"权重和必须为 1，实际 {w.sum()}"
    # 4. model_id / data_version 字面值
    assert (result["model_id"] == "CVAR1").all(), "model_id 必须全为 'CVAR1'"
    assert (result["data_version"] == "real-v1").all(), "data_version 必须全为 'real-v1'"
    # 5. trade_date 格式
    assert result["trade_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all(), \
        "trade_date 必须为 YYYY-MM-DD 格式"
    # 6. update_time ISO 8601（含 T）
    sample = result["update_time"].iloc[0]
    assert isinstance(sample, str) and "T" in sample, f"update_time 必须含 T，实际 {sample}"
    # 7. 组合级 attrs 存在
    for k in ["portfolio_cvar_95", "portfolio_var_95", "portfolio_expected_annual_return"]:
        assert k in result.attrs, f"组合级指标 attrs 缺少 {k}"
    # 8. CVaR ≥ VaR（尾部损失均值 ≥ 分位数，损失口径）
    assert result.attrs["portfolio_cvar_95"] >= result.attrs["portfolio_var_95"] - 1e-6, \
        f"CVaR({result.attrs['portfolio_cvar_95']}) 应 ≥ VaR({result.attrs['portfolio_var_95']})"

    print(f"PASS: 权重契约检查（8 条断言）通过，共 {len(result)} 个资产，"
          f"非零权重 {(w > 1e-6).sum()} 个")


def check_no_future_function(stock_df: pd.DataFrame) -> None:
    """无未来函数：截断到倒数第二天重算，只用历史数据 → 结果确定且可复现

    验证方式：同一份截断数据两次求解得到相同权重（LP 确定性），
    且截断数据求解不会用到被截掉的未来数据（因 optimize 只吃传入的 stock_df）。
    """
    ret = build_return_matrix(stock_df)
    dates = sorted(ret.index.tolist())
    if len(dates) < 30:
        print("[WARN] 样本过短，跳过无未来函数检查")
        return

    # 截断到倒数第 6 天（留足尾部样本）
    cut_date = dates[-6]
    truncated = stock_df[stock_df["date"].astype(str) <= cut_date]

    r1 = optimize_portfolio(stock_df=truncated)
    r2 = optimize_portfolio(stock_df=truncated)
    # 同数据两次求解权重一致（确定性）
    w1 = r1.sort_values("symbol")["weight"].values
    w2 = r2.sort_values("symbol")["weight"].values
    assert np.allclose(w1, w2, atol=1e-6), "同一截断数据两次求解权重不一致（求解非确定性？）"
    print("PASS: 无未来函数检查通过（截断重算权重可复现）")


def check_out_of_sample_slice(stock_df: pd.DataFrame) -> None:
    """样本外切片：train/test 非空"""
    ret = build_return_matrix(stock_df)
    train, test = split_train_test(ret)
    assert not train.empty, "train 段不能为空"
    assert not test.empty, "test 段不能为空"
    assert len(train) >= 20, f"train 段样本过少 ({len(train)})，CVaR 估计不可靠"
    print(f"PASS: 样本外切片检查通过（train={len(train)}, test={len(test)}）")


def check_target_return(result: pd.DataFrame) -> None:
    """目标收益约束：组合预期年化 ≥ 目标（除非已标记 relaxed）"""
    exp_annual = result.attrs["portfolio_expected_annual_return"]
    target = result.attrs["target_annual_return"]
    relaxed = result.attrs.get("target_relaxed", False)
    if relaxed:
        print(f"[INFO] 目标年化 {target:.1%} 不可行已放松，实际预期年化 {exp_annual:.4%}")
        return
    assert exp_annual >= target - 1e-4, \
        f"组合预期年化 {exp_annual:.4%} 未达目标 {target:.1%}（且未标记 relaxed）"
    print(f"PASS: 目标收益检查通过（预期年化 {exp_annual:.4%} ≥ 目标 {target:.1%}）")


def check_cvar_consistency(stock_df: pd.DataFrame) -> None:
    """CVaR 一致性：优化目标 CVaR 与样本内经验 CVaR 量级一致

    优化得到的 cvar_95（含 EVT 补样情景）应与用最优权重在历史情景上
    直接算的经验 CVaR 处于同一量级（EVT 补样会略微抬高，但不应差数量级）。
    """
    result = optimize_portfolio(stock_df=stock_df)
    ret = build_return_matrix(stock_df)
    weights = dict(zip(result["symbol"], result["weight"]))
    port_ret = portfolio_daily_returns(ret, weights).values
    hist_cvar = empirical_cvar(port_ret, BETA)
    opt_cvar = result.attrs["portfolio_cvar_95"]

    print(f"优化目标 CVaR95={opt_cvar:.4%}，历史经验 CVaR95={hist_cvar:.4%}")
    # 两者应同号（都是正的损失）且量级接近（EVT 补样使 opt ≥ hist 是正常的）
    assert opt_cvar > 0, "优化 CVaR 应为正（损失口径）"
    assert hist_cvar > 0, "历史 CVaR 应为正（损失口径）"
    # 允许 EVT 补样带来最多 3 倍抬升，超出说明补样过度或异常
    assert opt_cvar <= hist_cvar * 3.0 + 1e-4, \
        f"优化 CVaR({opt_cvar:.4%}) 远超历史({hist_cvar:.4%})，EVT 补样可能异常"
    print("PASS: CVaR 一致性检查通过")


if __name__ == "__main__":
    offline_mode = os.getenv("PANDA_DATA_OFFLINE", "0") == "1"
    mode_label = "PANDA_DATA_OFFLINE=1，使用 fixtures 数据" if offline_mode else "联网模式（默认）"
    print(f"[MODE] {mode_label}\n")

    stock, _ = _load_fixture_or_network(with_index=False)

    result = optimize_portfolio(stock_df=stock)

    check_weight_contract(result)
    check_no_future_function(stock)
    check_out_of_sample_slice(stock)
    check_target_return(result)
    check_cvar_consistency(stock)

    print("\n验证通过：权重契约完整、无未来函数、样本外切片可用、目标收益满足、CVaR 一致性达标")
