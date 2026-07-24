# skill-portfolio-cvar-optim

For constructing equity portfolios that minimize extreme tail losses.

## What this repository now provides

A minimal, end-to-end CVaR portfolio workflow that supports:

- CVaR-minimizing long-only portfolio construction
- target-return-constrained weight solving
- tail-risk comparison versus equal-weight and benchmark allocations
- full delivery workflow validation (`solve -> validate -> backtest -> report`)

## Main module

- `skill_portfolio_cvar_optim.py`

Key functions:

- `solve_cvar_weights(...)`
- `validate_solution(...)`
- `compare_tail_risk(...)`
- `backtest_workflow(...)`
- `run_delivery_workflow(...)`

## Running tests

```bash
cd /home/runner/work/skill-portfolio-cvar-optim/skill-portfolio-cvar-optim
python -m unittest discover -s tests -p "test_*.py"
```
