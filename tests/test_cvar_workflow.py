import unittest

from skill_portfolio_cvar_optim import (
    compare_tail_risk,
    portfolio_returns,
    run_delivery_workflow,
    solve_cvar_weights,
    validate_solution,
)


RETURNS = [
    [0.010, 0.020, 0.015],
    [0.011, 0.018, 0.014],
    [0.010, -0.180, -0.060],
    [0.009, 0.023, 0.016],
    [0.010, 0.019, 0.015],
    [0.010, -0.220, -0.070],
    [0.011, 0.021, 0.017],
    [0.010, 0.020, 0.016],
    [0.010, 0.022, 0.016],
    [0.010, -0.150, -0.050],
    [0.011, 0.019, 0.015],
    [0.010, 0.020, 0.016],
    [0.010, -0.130, -0.040],
    [0.011, 0.018, 0.015],
]


class TestCVaRWorkflow(unittest.TestCase):
    def test_solver_meets_constraints_and_reduces_tail_risk(self):
        target_return = 0.0101
        weights = solve_cvar_weights(RETURNS, target_return=target_return, alpha=0.8, resolution=0.05)

        checks = validate_solution(RETURNS, weights, target_return=target_return)
        self.assertTrue(checks["is_fully_invested"])
        self.assertTrue(checks["is_long_only"])
        self.assertTrue(checks["meets_target_return"])

        comparison = compare_tail_risk(
            RETURNS,
            target_return=target_return,
            alpha=0.8,
            benchmark_weights=[0.6, 0.2, 0.2],
        )
        self.assertLessEqual(comparison.cvar_optimized, comparison.equal_weight)
        self.assertLessEqual(comparison.cvar_optimized, comparison.benchmark)

    def test_delivery_workflow_returns_full_outputs(self):
        result = run_delivery_workflow(
            returns=RETURNS,
            target_return=0.0101,
            alpha=0.8,
            window=6,
            benchmark_weights=[0.6, 0.2, 0.2],
        )

        self.assertIn("weights", result)
        self.assertIn("validation", result)
        self.assertIn("comparison", result)
        self.assertIn("backtest", result)
        self.assertIn("report", result)

        self.assertTrue(result["validation"]["meets_target_return"])
        self.assertLessEqual(
            result["backtest"]["cvar_strategy_cvar"],
            result["backtest"]["equal_weight_cvar"],
        )

        strategy_returns = portfolio_returns(RETURNS, result["weights"])
        self.assertEqual(len(strategy_returns), len(RETURNS))


if __name__ == "__main__":
    unittest.main()
