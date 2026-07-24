from __future__ import annotations

from dataclasses import dataclass
from math import comb
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence


Number = float


@dataclass(frozen=True)
class PortfolioComparison:
    cvar_optimized: Number
    equal_weight: Number
    benchmark: Number


def _validate_matrix(returns: Sequence[Sequence[Number]]) -> None:
    if not returns:
        raise ValueError("returns must not be empty")
    width = len(returns[0])
    if width == 0:
        raise ValueError("returns must contain at least one asset")
    if any(len(row) != width for row in returns):
        raise ValueError("all return rows must have the same length")


def _mean_by_asset(returns: Sequence[Sequence[Number]]) -> List[Number]:
    _validate_matrix(returns)
    asset_count = len(returns[0])
    return [mean(row[i] for row in returns) for i in range(asset_count)]


def portfolio_returns(
    returns: Sequence[Sequence[Number]],
    weights: Sequence[Number],
) -> List[Number]:
    _validate_matrix(returns)
    if len(weights) != len(returns[0]):
        raise ValueError("weights length must match number of assets")
    return [sum(w * r for w, r in zip(weights, row)) for row in returns]


def cvar(losses: Sequence[Number], alpha: Number = 0.95) -> Number:
    if not losses:
        raise ValueError("losses must not be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ordered = sorted(losses, reverse=True)
    tail_size = max(1, int((1 - alpha) * len(ordered) + 0.999999))
    return mean(ordered[:tail_size])


def _compositions(total_units: int, parts: int) -> Iterable[List[int]]:
    if parts == 1:
        yield [total_units]
        return
    for first in range(total_units + 1):
        for rest in _compositions(total_units - first, parts - 1):
            yield [first] + rest


def solve_cvar_weights(
    returns: Sequence[Sequence[Number]],
    target_return: Number,
    alpha: Number = 0.95,
    expected_returns: Optional[Sequence[Number]] = None,
    resolution: Number = 0.01,
) -> List[Number]:
    _validate_matrix(returns)
    if resolution <= 0 or resolution > 1:
        raise ValueError("resolution must be in (0, 1]")

    asset_count = len(returns[0])
    exp = list(expected_returns) if expected_returns is not None else _mean_by_asset(returns)
    if len(exp) != asset_count:
        raise ValueError("expected_returns length must match asset count")

    units = int(round(1.0 / resolution))
    if abs(units * resolution - 1.0) > 1e-9:
        raise ValueError("resolution must evenly partition 1.0 (for example 0.1, 0.05, 0.01)")

    total_candidates = comb(units + asset_count - 1, asset_count - 1)
    if total_candidates > 300000:
        raise ValueError(
            "search space too large for grid solver; increase resolution or reduce asset count"
        )

    best_weights: Optional[List[Number]] = None
    best_cvar: Optional[Number] = None
    best_return: Optional[Number] = None

    for candidate_units in _compositions(units, asset_count):
        weights = [u / units for u in candidate_units]
        port_return = sum(w * r for w, r in zip(weights, exp))
        if port_return + 1e-12 < target_return:
            continue
        losses = [-r for r in portfolio_returns(returns, weights)]
        score = cvar(losses, alpha=alpha)
        if (
            best_cvar is None
            or score < best_cvar - 1e-12
            or (abs(score - best_cvar) <= 1e-12 and (best_return is None or port_return > best_return))
        ):
            best_weights = weights
            best_cvar = score
            best_return = port_return

    if best_weights is None:
        raise ValueError("no feasible weights satisfy the target return")
    return best_weights


def compare_tail_risk(
    returns: Sequence[Sequence[Number]],
    target_return: Number,
    alpha: Number = 0.95,
    benchmark_weights: Optional[Sequence[Number]] = None,
) -> PortfolioComparison:
    _validate_matrix(returns)
    asset_count = len(returns[0])

    cvar_weights = solve_cvar_weights(returns=returns, target_return=target_return, alpha=alpha)
    equal_weights = [1 / asset_count] * asset_count
    benchmark = list(benchmark_weights) if benchmark_weights is not None else equal_weights
    if len(benchmark) != asset_count:
        raise ValueError("benchmark_weights length must match asset count")

    cvar_port = cvar([-r for r in portfolio_returns(returns, cvar_weights)], alpha=alpha)
    eq_port = cvar([-r for r in portfolio_returns(returns, equal_weights)], alpha=alpha)
    bench_port = cvar([-r for r in portfolio_returns(returns, benchmark)], alpha=alpha)

    return PortfolioComparison(
        cvar_optimized=cvar_port,
        equal_weight=eq_port,
        benchmark=bench_port,
    )


def validate_solution(
    returns: Sequence[Sequence[Number]],
    weights: Sequence[Number],
    target_return: Number,
    expected_returns: Optional[Sequence[Number]] = None,
) -> Dict[str, Number | bool]:
    _validate_matrix(returns)
    exp = list(expected_returns) if expected_returns is not None else _mean_by_asset(returns)
    if len(exp) != len(weights):
        raise ValueError("expected_returns length must match weight length")

    weight_sum = sum(weights)
    min_weight = min(weights)
    achieved_return = sum(w * r for w, r in zip(weights, exp))
    return {
        "is_fully_invested": abs(weight_sum - 1.0) <= 1e-8,
        "is_long_only": min_weight >= -1e-12,
        "meets_target_return": achieved_return + 1e-12 >= target_return,
        "weight_sum": weight_sum,
        "min_weight": min_weight,
        "achieved_return": achieved_return,
    }


def backtest_workflow(
    returns: Sequence[Sequence[Number]],
    target_return: Number,
    alpha: Number = 0.95,
    window: int = 12,
    benchmark_weights: Optional[Sequence[Number]] = None,
) -> Dict[str, Number]:
    _validate_matrix(returns)
    if window <= 1 or window >= len(returns):
        raise ValueError("window must be >1 and < number of return rows")

    asset_count = len(returns[0])
    equal_weights = [1 / asset_count] * asset_count
    benchmark = list(benchmark_weights) if benchmark_weights is not None else equal_weights

    realized_cvar: List[Number] = []
    realized_equal: List[Number] = []
    realized_benchmark: List[Number] = []

    for idx in range(window, len(returns)):
        train = returns[idx - window : idx]
        live_row = returns[idx]
        train_mean = _mean_by_asset(train)
        effective_target = min(target_return, max(train_mean))
        w = solve_cvar_weights(
            returns=train,
            target_return=effective_target,
            alpha=alpha,
            expected_returns=train_mean,
        )
        realized_cvar.append(sum(a * b for a, b in zip(w, live_row)))
        realized_equal.append(sum(a * b for a, b in zip(equal_weights, live_row)))
        realized_benchmark.append(sum(a * b for a, b in zip(benchmark, live_row)))

    return {
        "cvar_strategy_cvar": cvar([-x for x in realized_cvar], alpha=alpha),
        "equal_weight_cvar": cvar([-x for x in realized_equal], alpha=alpha),
        "benchmark_cvar": cvar([-x for x in realized_benchmark], alpha=alpha),
        "cvar_strategy_mean_return": mean(realized_cvar),
        "equal_weight_mean_return": mean(realized_equal),
        "benchmark_mean_return": mean(realized_benchmark),
    }


def run_delivery_workflow(
    returns: Sequence[Sequence[Number]],
    target_return: Number,
    alpha: Number = 0.95,
    window: int = 12,
    benchmark_weights: Optional[Sequence[Number]] = None,
) -> Dict[str, object]:
    _validate_matrix(returns)

    solved_weights = solve_cvar_weights(returns=returns, target_return=target_return, alpha=alpha)
    validation = validate_solution(
        returns=returns,
        weights=solved_weights,
        target_return=target_return,
    )
    comparison = compare_tail_risk(
        returns=returns,
        target_return=target_return,
        alpha=alpha,
        benchmark_weights=benchmark_weights,
    )
    backtest = backtest_workflow(
        returns=returns,
        target_return=target_return,
        alpha=alpha,
        window=window,
        benchmark_weights=benchmark_weights,
    )

    report = {
        "solver": "grid-search CVaR under long-only and target return constraints",
        "weights": solved_weights,
        "validation": validation,
        "tail_risk_comparison": {
            "cvar_optimized": comparison.cvar_optimized,
            "equal_weight": comparison.equal_weight,
            "benchmark": comparison.benchmark,
        },
        "backtest": backtest,
    }

    return {
        "weights": solved_weights,
        "validation": validation,
        "comparison": comparison,
        "backtest": backtest,
        "report": report,
    }
