# skill-portfolio-cvar-optim

**简体中文** | [English](README.en.md)

> CVaR 尾部风险优化组合：在预期收益约束下最小化组合 95% CVaR，支持极值理论(EVT/GPD)尾部补样、组合权重求解、样本外验证与 HTML 可视化报告。

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

`skill-portfolio-cvar-optim` 是一个组合尾部风险优化 Skill，基于 Rockafellar-Uryasev 线性化方法，
在给定目标年化收益约束下最小化组合 CVaR@95%，并完成权重契约验证、样本外回测与离线复现。

这个 Skill 适合用于：

- 股票组合的尾部风险最小化权重求解（CVaR-LP）
- 极端市况下尾部损失的情景构建（含 EVT/GPD 尾部补样）
- 组合权重验证、契约检查、样本外回测的完整工作流
- Claude Code 对话中触发确定性组合优化与回测

本 Skill 通过 `panda_data` SDK 拉取 A 股股票池（`get_stock_daily`）与基准指数（`get_index_daily`），
输出权重表、回测指标（CVaR/VaR/年化/MDD/Calmar）、HTML 可视化报告以及离线 fixture，便于 CI 集成。

## 核心方法

- **CVaR 线性化**：Rockafellar-Uryasev LP，`scipy.optimize.linprog(method="highs")` 求解
- **极值理论补样**：尾部样本不足时用 GPD(Peaks-Over-Threshold) 拟合并重采样补足尾部情景
- **样本外验证**：train 段求权重，test 段用固定权重验证，无未来函数

## 仓库内容

| 文件 | 说明 |
|---|---|
| `SKILL.md` | Skill 契约文档（Agent 内部使用，含完整公式） |
| `scripts/factor.py` | CVaR 权重求解入口（收益矩阵 → EVT 补尾 → LP → 权重表，可独立运行） |
| `scripts/data_loader.py` | 数据加载层（股票/指数日线 + parquet 缓存 + 字段标准化） |
| `scripts/validate.py` | 权重验证（8 条契约 + 无未来函数 + 目标收益 + CVaR 一致性） |
| `scripts/backtest.py` | 样本外回测（CVaR/年化/MDD + 等权/benchmark 对比） |
| `scripts/backtest_report_data.py` | 回测时序数据包装层（HTML 报告数据源） |
| `scripts/report.py` | HTML 报告生成入口（matplotlib 静态图，4 图 + 8 指标卡） |
| `scripts/save_fixture.py` | 一次性生成离线测试 fixture（股票 + 指数） |
| `scripts/analysis_report.py` | 参数扫描深度分析（目标收益 / β → 有效前沿） |
| `scripts/fixtures/` | 离线测试数据（Parquet 格式） |
| `requirements.txt` | Python 依赖清单 |
| `references/data_guide.md` | PandaAI 数据接口参考 |
| `LICENSE` | GPLv3 协议 |
| `README.md` | 中文 README |

## 目录结构

```text
skill-portfolio-cvar-optim/
├── SKILL.md
├── README.md
├── LICENSE
├── requirements.txt
├── references/
│   └── data_guide.md
├── scripts/
│   ├── factor.py                  # CVaR 权重求解（EVT 补尾 + LP + optimize_portfolio + main）
│   ├── data_loader.py             # 数据加载层（load_stock/index_data + 缓存 + 字段标准化）
│   ├── validate.py                # 验证（8 条权重契约 + 无未来函数 + CVaR 一致性）
│   ├── backtest.py                # 样本外回测（CVaR/VaR/年化/MDD/Calmar + 等权/benchmark）
│   ├── backtest_report_data.py    # 回测时序数据包装层（保留 curve / drawdown / loss_hist）
│   ├── report.py                  # HTML 报告（4 图 + 8 指标卡 + 三方对比 + 权重明细）
│   ├── save_fixture.py            # 生成离线 fixture
│   ├── analysis_report.py         # 参数扫描 + CVaR-收益有效前沿
│   └── fixtures/
│       ├── sample_stocks.parquet  # 离线股票数据
│       └── sample_index.parquet   # 离线指数数据
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置数据源凭证
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 3. 联网求解 + 验证 + 回测 + 报告
python scripts/factor.py
python scripts/validate.py
python scripts/backtest.py
python scripts/report.py

# 或离线模式（需先跑一次 save_fixture.py）
export PANDA_DATA_OFFLINE=1
python scripts/validate.py
python scripts/report.py --offline
```

## 输出示例

权重表（每个资产一行）：

| symbol | weight | expected_annual_return | tail_contribution |
|---|---|---|---|
| 000333.SZ | 0.3000 | 16.78% | 0.004876 |
| 600900.SH | 0.2345 | -4.68% | 0.003192 |
| ... | ... | ... | ... |

组合级指标：CVaR95 / VaR95 / 预期年化 / 目标收益 / EVT 补样记录，存于 `DataFrame.attrs`。

## License

GPLv3
