"""回测数据包装层 —— 在 backtest.run_backtest() 基础上保留时序数据，供 HTML 报告使用

为什么需要：run_backtest() 只返回聚合指标 dict，丢掉了净值曲线、回撤序列、
          损失分布等时间维度数据。HTML 报告需要画图，必须保留这些中间产物。

设计原则：
    - 不修改 backtest.py / factor.py（保持原契约）
    - 复用其纯函数，避免重复实现
    - 离线模式注入 panda_data stub + pd.read_parquet fastparquet fallback
    - 输出 JSON 友好（numpy/pandas 类型显式转原生）
"""
from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

# 兼容 Python embed 版本：把脚本所在目录注入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


def _is_offline(offline: bool | None) -> bool:
    """判断离线模式：显式参数优先，其次看环境变量"""
    if offline is not None:
        return offline
    return os.getenv("PANDA_DATA_OFFLINE", "0") == "1"


def _inject_panda_data_stub_for_offline() -> None:
    """panda_data SDK 缺失时注入 stub，绕过 data_loader.py 顶层 import

    离线流程走 fixtures 不调 SDK API，注入 stub 让 import 链通过即可。
    """
    if "panda_data" in sys.modules:
        return
    try:
        import panda_data  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("panda_data")
    stub.init_token = lambda **kw: None  # type: ignore[attr-defined]
    stub.get_stock_daily = lambda **kw: pd.DataFrame()  # type: ignore[attr-defined]
    stub.get_index_daily = lambda **kw: pd.DataFrame()  # type: ignore[attr-defined]
    sys.modules["panda_data"] = stub
    print("[INFO] panda_data SDK 未安装，已注入 stub 模块（离线模式可用；联网会报错）")


_inject_panda_data_stub_for_offline()


def _patch_pd_read_parquet_fallback() -> None:
    """patch pd.read_parquet，pyarrow 失败时 fallback fastparquet"""
    try:
        import fastparquet  # noqa: F401
    except ImportError:
        return
    if getattr(pd.read_parquet, "_is_patched", False):
        return
    orig = pd.read_parquet

    def patched(path, *args, **kwargs):
        if kwargs.get("engine") is None:
            try:
                return orig(path, *args, **kwargs)
            except Exception:
                kwargs["engine"] = "fastparquet"
                return orig(path, *args, **kwargs)
        return orig(path, *args, **kwargs)

    patched._is_patched = True  # type: ignore[attr-defined]
    pd.read_parquet = patched  # type: ignore[assignment]
    print("[INFO] 已为 pd.read_parquet 打补丁：pyarrow 失败时自动 fallback fastparquet")


_patch_pd_read_parquet_fallback()

from factor import (  # noqa: E402
    BETA,
    TARGET_ANNUAL_RETURN,
    TRADING_DAYS,
    WEIGHT_UPPER_BOUND,
    build_return_matrix,
    build_scenario_matrix,
    optimize_portfolio,
    solve_cvar_lp,
)
from backtest import (  # noqa: E402
    _index_daily_returns,
    annualized_return,
    annualized_vol,
    empirical_cvar,
    empirical_var,
    max_drawdown,
    portfolio_daily_returns,
    sharpe_ratio,
    split_train_test,
)
from validate import _load_fixture_or_network  # noqa: E402


def _fmt_date(d: Any) -> str:
    if isinstance(d, str):
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if not np.isnan(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return str(obj)


def _curve_series(returns: np.ndarray, dates: list) -> list[dict]:
    """收益序列 → 累计收益率曲线（从 0 起）"""
    cum = np.cumprod(1.0 + returns) - 1.0
    return [{"date": _fmt_date(d), "cum_return": float(v)} for d, v in zip(dates, cum)]


def _drawdown_series(returns: np.ndarray, dates: list) -> list[dict]:
    """收益序列 → 回撤曲线"""
    curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = curve / running_max - 1.0
    return [{"date": _fmt_date(d), "drawdown": float(v)} for d, v in zip(dates, dd)]


def run_backtest_with_series(offline: bool | None = None) -> dict:
    """复刻 backtest 主流程，保留时序数据供 HTML 报告

    Returns:
        含 meta / timeseries / weights / metrics 的完整 dict
    """
    is_offline = _is_offline(offline)
    data_version = "offline-fixture" if is_offline else "real-v1"

    stock_df, index_df = _load_fixture_or_network(with_index=True)

    ret = build_return_matrix(stock_df)
    train_ret, test_ret = split_train_test(ret)
    symbols = ret.columns.tolist()
    n_assets = len(symbols)

    # === 样本内求权重 ===
    mu_train = train_ret.mean().values
    target_daily = (1.0 + TARGET_ANNUAL_RETURN) ** (1.0 / TRADING_DAYS) - 1.0
    scenario_train, evt_info = build_scenario_matrix(train_ret)
    opt = solve_cvar_lp(scenario_train, mu_train, beta=BETA,
                        target_daily_return=target_daily, weight_upper=WEIGHT_UPPER_BOUND)
    relaxed = False
    if not opt["success"]:
        opt = solve_cvar_lp(scenario_train, mu_train, beta=BETA,
                            target_daily_return=0.0, weight_upper=WEIGHT_UPPER_BOUND)
        relaxed = True
        if not opt["success"]:
            raise ValueError(f"CVaR-LP 求解失败: {opt['message']}")

    cvar_weights = dict(zip(symbols, opt["weights"]))
    equal_weights = {s: 1.0 / n_assets for s in symbols}

    # === 样本外验证 ===
    test_dates = test_ret.index.tolist()
    cvar_test = portfolio_daily_returns(test_ret, cvar_weights).values
    eq_test = portfolio_daily_returns(test_ret, equal_weights).values

    bench_test = None
    bench_dates = None
    if index_df is not None and not index_df.empty:
        bench_ret = _index_daily_returns(index_df).reindex(test_ret.index).dropna()
        if not bench_ret.empty:
            bench_test = bench_ret.values
            bench_dates = bench_ret.index.tolist()

    def _scalar_metrics(r: np.ndarray) -> dict:
        return {
            "CVaR95": round(empirical_cvar(r, BETA), 6),
            "VaR95": round(empirical_var(r, BETA), 6),
            "年化收益": round(annualized_return(r), 6),
            "年化波动": round(annualized_vol(r), 6),
            "最大回撤": round(max_drawdown(r), 6),
            "Calmar": round(annualized_return(r) / abs(max_drawdown(r)), 6) if max_drawdown(r) != 0 else 0.0,
            "夏普": round(sharpe_ratio(r), 6),
        }

    cvar_metrics = _scalar_metrics(cvar_test)
    eq_metrics = _scalar_metrics(eq_test)
    bench_metrics = _scalar_metrics(bench_test) if bench_test is not None else None

    # === 权重明细（非零优先，按权重降序） ===
    tail_contrib = _tail_contribution(scenario_train, opt["weights"], BETA)
    weight_rows = []
    for i, sym in enumerate(symbols):
        weight_rows.append({
            "symbol": sym,
            "weight": round(float(opt["weights"][i]), 6),
            "expected_annual_return": round(float((1.0 + mu_train[i]) ** TRADING_DAYS - 1.0), 6),
            "tail_contribution": round(float(tail_contrib[i]), 6),
        })
    weight_rows.sort(key=lambda x: x["weight"], reverse=True)

    # === 损失分布直方图数据（test 段组合收益，用于画尾部分布） ===
    loss_hist = _loss_histogram(cvar_test, eq_test, BETA)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_version": data_version,
        "sample_start": _fmt_date(test_dates[0]) if test_dates else None,
        "sample_end": _fmt_date(test_dates[-1]) if test_dates else None,
        "train_days": len(train_ret),
        "test_days": len(test_ret),
        "asset_count": n_assets,
        "nonzero_weights": int((opt["weights"] > 1e-6).sum()),
        "max_weight": round(float(opt["weights"].max()), 6),
        "target_annual_return": TARGET_ANNUAL_RETURN,
        "beta": BETA,
        "weight_upper": WEIGHT_UPPER_BOUND,
        "target_relaxed": relaxed,
        "evt_resampled": evt_info,
    }

    return {
        "meta": meta,
        "metrics": {
            "CVaR组合": cvar_metrics,
            "等权组合": eq_metrics,
            "Benchmark": bench_metrics,
            "样本内CVaR95": round(opt["cvar_95"], 6),
            "样本内VaR95": round(opt["var_95"], 6),
            "尾部改善(CVaR-等权)": round(cvar_metrics["CVaR95"] - eq_metrics["CVaR95"], 6),
        },
        "timeseries": {
            "cvar_curve": _curve_series(cvar_test, test_dates),
            "equal_curve": _curve_series(eq_test, test_dates),
            "bench_curve": _curve_series(bench_test, bench_dates) if bench_test is not None else [],
            "cvar_drawdown": _drawdown_series(cvar_test, test_dates),
        },
        "weights": weight_rows,
        "loss_histogram": loss_hist,
        "评估口径": (
            f"train 段（前半样本）求 CVaR@{BETA:.0%} 最优权重，test 段用固定权重验证，"
            f"权重不含 test 信息，无未来函数。CVaR/VaR 为损失口径正值，越小越好。"
        ),
    }


def _tail_contribution(scenario: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    """各资产对组合 CVaR 的边际尾部贡献 = w_i · E[-r_i | 组合处于最差尾部]"""
    port_loss = -(scenario @ w)
    n_tail = max(1, int(np.ceil((1.0 - beta) * len(port_loss))))
    tail_idx = np.argsort(port_loss)[-n_tail:]
    tail_asset_loss = -(scenario[tail_idx].mean(axis=0))
    return w * tail_asset_loss


def _loss_histogram(cvar_ret: np.ndarray, eq_ret: np.ndarray, beta: float, bins: int = 30) -> dict:
    """构建 CVaR 组合与等权组合的收益分布直方图 + VaR/CVaR 标注"""
    all_ret = np.concatenate([cvar_ret, eq_ret])
    lo, hi = float(all_ret.min()), float(all_ret.max())
    edges = np.linspace(lo, hi, bins + 1).tolist()
    cvar_counts, _ = np.histogram(cvar_ret, bins=edges)
    eq_counts, _ = np.histogram(eq_ret, bins=edges)
    return {
        "bin_edges": edges,
        "cvar_counts": cvar_counts.tolist(),
        "equal_counts": eq_counts.tolist(),
        "cvar_var95": round(-float(np.quantile(cvar_ret, 1.0 - beta)), 6),
        "equal_var95": round(-float(np.quantile(eq_ret, 1.0 - beta)), 6),
    }


def save_backtest_result(data: dict, json_path: Path) -> None:
    """保存回测结果到 JSON（自动创建父目录）"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


if __name__ == "__main__":
    result = run_backtest_with_series()
    out_path = Path(__file__).parent.parent / "reports" / "backtest_result.json"
    save_backtest_result(result, out_path)
    print(f"[OK] 回测结果已保存到 {out_path}")
    print(f"     test 区间: {result['meta']['sample_start']} ~ {result['meta']['sample_end']}"
          f" ({result['meta']['test_days']} 交易日, {result['meta']['asset_count']} 资产)")
