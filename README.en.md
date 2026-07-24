# skill-portfolio-cvar-optim

[简体中文](README.md) | **English**

> CVaR tail-risk optimization portfolio: minimize portfolio 95% CVaR under an expected-return constraint, with extreme-value-theory (EVT/GPD) tail resampling, weight solving, out-of-sample validation, and HTML visualization report.

<p align="center">
  <img alt="libraries" src="https://img.shields.io/badge/libraries-CVaR%20Portfolio%20Optimization-blue">
  <img alt="model" src="https://img.shields.io/badge/model-CVAR1-brightgreen">
  <img alt="type" src="https://img.shields.io/badge/type-portfolio--risk-blue">
  <img alt="platform" src="https://img.shields.io/badge/platform-PandaAI-9cf">
  <img alt="solver" src="https://img.shields.io/badge/solver-scipy%20linprog-orange">
  <img alt="evt" src="https://img.shields.io/badge/EVT-GPD%20tail-red">
  <img alt="status" src="https://img.shields.io/badge/status-active-brightgreen">
  <img alt="license" src="https://img.shields.io/badge/license-GPLv3-blue">
</p>

`skill-portfolio-cvar-optim` is a portfolio tail-risk optimization Skill based on the Rockafellar-Uryasev linearization method, minimizing portfolio CVaR@95% under a target annualized return constraint, with weight-contract validation, out-of-sample backtesting, and offline reproduction.

This Skill is suitable for:

- Tail-risk-minimizing weight solving for equity portfolios (CVaR-LP)
- Scenario construction of tail losses under extreme market conditions (with EVT/GPD tail resampling)
- Complete workflows for portfolio weight validation, contract checking, and out-of-sample backtesting
- Triggering deterministic portfolio optimization and backtesting from Claude Code conversations

This Skill pulls A-share stock pool (`get_stock_daily`) and benchmark index (`get_index_daily`) data via the `panda_data` SDK, outputting weight tables, backtest metrics (CVaR/VaR/annualized/MDD/Calmar), a self-contained HTML visualization report, and offline fixtures for CI integration.

## Core Methods

- **CVaR linearization**: Rockafellar-Uryasev LP, solved by `scipy.optimize.linprog(method="highs")`
- **Extreme-value-theory tail resampling**: when historical tail samples are insufficient, fit GPD (Peaks-Over-Threshold) and resample to augment tail scenarios
- **Out-of-sample validation**: solve weights on the train segment, validate on the test segment with fixed weights, no look-ahead

## Repository Contents

| File | Description |
|---|---|
| `SKILL.md` | Skill contract document (internal Agent use, full formulas) |
| `scripts/factor.py` | CVaR weight solver entry (return matrix → EVT tail → LP → weight table, standalone) |
| `scripts/data_loader.py` | Data layer (stock/index daily + parquet cache + field normalization) |
| `scripts/validate.py` | Weight validation (8 contracts + no-look-ahead + target return + CVaR consistency) |
| `scripts/backtest.py` | Out-of-sample backtest (CVaR/annualized/MDD + equal-weight/benchmark comparison) |
| `scripts/backtest_report_data.py` | Backtest timeseries wrapper (HTML report data source) |
| `scripts/report.py` | HTML report generator entry (matplotlib static charts, 4 charts + 8 metric cards) |
| `scripts/save_fixture.py` | One-time offline fixture generator (stocks + index) |
| `scripts/analysis_report.py` | Parameter-sweep deep analysis (target return / β → efficient frontier) |
| `scripts/fixtures/` | Offline test data (Parquet format) |
| `requirements.txt` | Python dependencies |
| `references/data_guide.md` | PandaAI data API reference |
| `LICENSE` | GPLv3 license |
| `README.md` / `README.en.md` | Chinese / English README |

## Directory Structure

```text
skill-portfolio-cvar-optim/
├── SKILL.md
├── README.md
├── README.en.md
├── LICENSE
├── requirements.txt
├── references/
│   └── data_guide.md
├── scripts/
│   ├── factor.py                  # CVaR solver (EVT tail + LP + optimize_portfolio + main)
│   ├── data_loader.py             # Data layer (load_stock/index_data + cache + normalization)
│   ├── validate.py                # Validation (8 weight contracts + no-look-ahead + CVaR consistency)
│   ├── backtest.py                # Out-of-sample (CVaR/VaR/annualized/MDD/Calmar + equal-weight/benchmark)
│   ├── backtest_report_data.py    # Timeseries wrapper (curve / drawdown / loss_hist)
│   ├── report.py                  # HTML report (4 charts + 8 metric cards + 3-way comparison)
│   ├── save_fixture.py            # Offline fixture generator
│   ├── analysis_report.py         # Parameter sweep + CVaR-return efficient frontier
│   └── fixtures/
│       ├── sample_stocks.parquet  # Offline stock data
│       └── sample_index.parquet   # Offline index data
└── reports/                       # Report artifacts (.gitignored)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure data credentials
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 3. Online: solve + validate + backtest + report
python scripts/factor.py
python scripts/validate.py
python scripts/backtest.py
python scripts/report.py

# Or offline mode (requires running save_fixture.py once)
export PANDA_DATA_OFFLINE=1
python scripts/validate.py
python scripts/report.py --offline
```

In offline mode, if `panda_data` SDK is not installed locally, a stub module is auto-injected to bypass the top-level import; when pyarrow is incompatible with legacy parquet, it falls back to the `fastparquet` engine.

## Output Example

Weight table (one row per asset):

| symbol | weight | expected_annual_return | tail_contribution |
|---|---|---|---|
| 000333.SZ | 0.3000 | 16.78% | 0.004876 |
| 600900.SH | 0.2345 | -4.68% | 0.003192 |
| ... | ... | ... | ... |

Portfolio-level metrics: CVaR95 / VaR95 / expected annualized / target return / EVT resampling log, stored in `DataFrame.attrs`.

## License

[GPL-3.0](LICENSE) © 2026 PandaTest
