"""边界路径回归测试 —— 覆盖离线 validate 未触发的分支（pytest 版）

P1-1: LP 三档降级（超高目标 target_relaxed / N×upper<1 上限自动放宽）
P1-2: EVT 补样（短样本触发，验证不崩 + 情景数正确）
2 资产下界 / 确定性

运行：cd scripts && pytest test_edge.py -v（conftest 默认 PANDA_DATA_OFFLINE=1 读 fixtures）
"""
import numpy as np
import pytest

from factor import optimize_portfolio
from validate import _load_fixture_or_network


@pytest.fixture(scope="module")
def stock():
    """离线 fixture 股票数据（conftest 已默认 PANDA_DATA_OFFLINE=1，读 fixtures 不联网）"""
    df, _ = _load_fixture_or_network(with_index=False)
    return df


def test_target_relaxed_high_target(stock):
    """【1】极高目标收益(500%) → 应触发 target_relaxed；20资产×0.30 不应触发上限放宽"""
    r = optimize_portfolio(stock_df=stock, target_annual_return=5.0)
    assert r.attrs["target_relaxed"] is True
    assert r.attrs["weight_upper_relaxed"] is False, "20资产×0.30 不应触发上限放宽"


def test_evt_short_sample(stock):
    """【2】短样本(60天) → 应触发 EVT GPD 补样"""
    short_dates = sorted(stock["date"].astype(str).unique())[:60]
    short = stock[stock["date"].astype(str).isin(short_dates)]
    r = optimize_portfolio(stock_df=short)
    evt = r.attrs["evt_resampled"]
    assert evt, "短样本应触发 EVT 补样"


def test_upper_relaxed_two_assets(stock):
    """【3】2资产 + 默认上限0.30(N×upper<1) → 应自动放宽上限而非崩溃"""
    two = stock[stock["symbol"].isin(stock["symbol"].unique()[:2])]
    r = optimize_portfolio(stock_df=two)
    assert r.attrs["weight_upper_relaxed"] is True
    assert r.attrs["effective_weight_upper"] == 1.0


def test_no_upper_relax_full_pool(stock):
    """【3b】20资产 + 默认0.30(N×upper=6≥1) → 不触发上限放宽"""
    r = optimize_portfolio(stock_df=stock)
    assert r.attrs["weight_upper_relaxed"] is False


def test_determinism(stock):
    """【4】同输入两次求解确定性"""
    a = optimize_portfolio(stock_df=stock).sort_values("symbol")["weight"].values
    b = optimize_portfolio(stock_df=stock).sort_values("symbol")["weight"].values
    assert np.allclose(a, b, atol=1e-9)
