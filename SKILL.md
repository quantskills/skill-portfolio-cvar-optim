---
name: portfolio-cvar-optim
description: 当需要开发、计算、验证 CVaR 尾部风险最小化组合时，使用此 skill。在预期收益约束下最小化组合 95% CVaR，支持极值理论(EVT/GPD)尾部补样、组合权重求解、样本外验证。
tags: [quant, portfolio, optimization, risk, cvar]
---

# CVaR 尾部风险优化

## 适用场景

- 当用户需要构建"最小化极端尾部损失"的股票组合时
- 当用户需要在给定目标收益下求解 CVaR 最优权重时
- 当用户需要对比 CVaR 优化组合与等权 / 指数基准的尾部风险时
- 当用户需要验证组合优化类模型的完整交付流程（求解 → 验证 → 回测 → 报告）时

## 模型原理

**CVaR（Conditional Value at Risk，条件在险价值）** 又称期望损失（Expected Shortfall），
衡量"最差 (1-β) 比例情景下的平均损失"，比 VaR 更能刻画尾部厚度，且是凸风险度量，可线性规划求解。

本模型基于 **Rockafellar-Uryasev (2000)** 的线性化方法，在"预期收益 ≥ 目标"约束下最小化组合 CVaR。

### 核心定义

设组合权重 `w ∈ R^N`，第 `t` 个情景的资产收益向量为 `r_t`（共 `T` 个情景），
组合在情景 `t` 的损失定义为：

```
L_t(w) = -(r_t · w)          # 损失 = 负收益
```

给定置信水平 `β`（本模型默认 `β = 0.95`）：

```
VaR_β(w)  = min{ α : P(L(w) ≤ α) ≥ β }      # β 分位损失
CVaR_β(w) = E[ L(w) | L(w) ≥ VaR_β(w) ]     # 尾部平均损失（≥ VaR）
```

### Rockafellar-Uryasev 线性化

CVaR 可写成如下凸优化（辅助变量 `α` 逼近 VaR）：

```
CVaR_β(w) = min_α  { α + 1/(1-β) · E[ (L(w) - α)_+ ] }
```

其中 `(x)_+ = max(x, 0)`。用 `T` 个离散情景 + 辅助变量 `u_t` 展开为 **线性规划(LP)**：

```
决策变量: w ∈ R^N,  α ∈ R,  u ∈ R^T

目标:     min   α + 1/((1-β)·T) · Σ_{t=1}^{T} u_t

约束:
  (尾部)   u_t ≥ L_t(w) - α  =  -(r_t · w) - α        ∀t
           u_t ≥ 0                                     ∀t
  (归一)   Σ_{i=1}^{N} w_i = 1
  (非负+上限) 0 ≤ w_i ≤ w_max                          ∀i
  (收益)   μ · w ≥ r_target                            # μ 为资产日均收益向量
```

- 最优目标值即为组合 `CVaR_β`，最优 `α` 即为 `VaR_β`（损失口径，正值代表损失）。
- 用 `scipy.optimize.linprog(method="highs")` 求解，变量拼接顺序 `x = [w, α, u]`（`N+1+T` 维）。
- **目标收益日频换算**：`r_target = (1 + 年化目标)^(1/252) - 1`（默认年化 8%）。

### 关键坑 1：历史尾部样本少 → 极值理论(EVT)补样

历史样本中真正落入最差 5% 的观测很少（如 250 天只有约 12 个），CVaR 估计噪声大、易低估。
本模型用 **Peaks-Over-Threshold + 广义帕累托分布(GPD)** 补充尾部情景：

```
1. 对每个资产的损失序列 L_i = -r_i，取尾部分位阈值 u = Quantile(L_i, 1 - q)   # q=10%
2. 超阈量 (L_i - u | L_i > u) 渐近服从广义帕累托分布 GPD(ξ, σ):
       F(x) = 1 - (1 + ξx/σ)^(-1/ξ),   x ≥ 0,  σ > 0
   （ξ 为形状参数/尾指数，ξ>0 为重尾；σ 为尺度参数）
3. 用 MLE 拟合 (ξ, σ)（scipy.stats.genpareto.fit, floc=0）
4. 从拟合的 GPD 重采样补足 EVT_RESAMPLE_SIZE 个尾部超阈样本，还原为损失 = u + 超阈量
5. 把补样情景（触发资产取 GPD 补出的损失，其它资产取【历史均值】——只补单资产边际尾部，不人为制造"全资产同崩"共尾）附加到情景矩阵
```

- **仅当**某资产历史尾部样本数 `< EVT_MIN_TAIL_SAMPLES`（默认 30）时才触发补样；
  样本充足则用纯历史情景（降级）。
- 拟合失败（超阈样本 < 5、ξ/σ 非有限、σ≤0）时该资产降级为纯历史，不中断流程。
- **口径注意**：去共尾（其它资产取均值）后，补样情景的组合损失比真实历史（多资产同跌）
  乐观，故 LP 目标值 `portfolio_cvar_95` 偏向 CVaR **下界**；真实尾部风险以**样本外实现 CVaR**为准。

### 关键坑 2：尾部驱动组合集中度更高

尾部风险最小化倾向于把权重压到"尾部最温和"的少数资产上，容易退化为单资产。
本模型用 **单资产权重上限 `w_max`（默认 0.30）** 约束强制分散。可通过参数调整。

### 关键坑 3：LP 不可行 → 三档降级（避免硬崩溃）

目标收益过高、或股票池太小（`N × w_max < 1` 导致权重无法归一）时 LP 不可行。
本模型用 `solve_cvar_with_relax` 三档兜底，**绝不直接抛异常**：

```
1. 原参数求解
2. 收益约束放松到 0（target_relaxed=True）
3. 权重上限放宽到 1.0（weight_upper_relaxed=True，等权必可行）
```

降级状态写入 attrs（`target_relaxed` / `weight_upper_relaxed` / `effective_weight_upper`），
调用方可据此判断结果是否"打了折扣"。

## 输入数据

本模型使用 Panda data SDK 拉取 A 股与指数日线。正式开发时，必须使用 PandaAI data 数据拉取库或项目指定数据源。

| 数据 | 接口 | 用途 |
|---|---|---|
| 股票池日线 | `get_stock_daily` | 可配置资产，构建 T×N 收益矩阵求权重 |
| 基准指数日线 | `get_index_daily` | benchmark，计算超额收益 / 信息比 |

### 输入契约

- **股票 symbol 格式**：带交易所后缀，如 `"600519.SH"`、`"000001.SZ"`（大写）
- **指数 symbol 格式**：如 `"000300.SH"`（沪深300）
- **默认股票池**：沪深300 中 20 只流动性好、跨行业的成分股（见 `data_loader.DEFAULT_STOCK_POOL`）
- **默认基准**：`000300.SH`（沪深300）
- **默认回溯窗口**：1 年（CVaR 尾部估计需要足够样本）
- **字段契约**：接口须返回可标准化为 `date` / `symbol` / `close` 的字段。
  `_normalize_price` 自动把 `code`/`ts_code`→`symbol`、`trade_date`→`date`，并打印 `[INFO]` 提示
- **调试模式**：`PANDA_DATA_DEBUG=1` 时首次成功调用打印接口字段名与前 3 行

## 输出结果

`optimize_portfolio()` 返回权重表 DataFrame，**每个资产一行**：

| 字段 | 说明 |
|---|---|
| trade_date | 权重生成日期（数据最新日，YYYY-MM-DD） |
| asset_type | 资产类型（stock） |
| symbol | 资产代码 |
| model_id | 模型编号（CVAR1） |
| model_name | 模型名称 |
| weight | 最优权重（∈ [0, w_max]，全部和为 1） |
| expected_return | 该资产样本期日均收益 μ_i |
| expected_annual_return | 年化预期收益 `(1+μ_i)^252 - 1` |
| tail_contribution | 边际尾部贡献 = `w_i · E[-r_i \| 组合处于最差 5% 尾部]`，正值=尾部风险来源 |
| data_version | 数据版本（real-v1 / offline-fixture） |
| update_time | 结果生成时间（ISO 8601，按数据最新日推导，可复现） |

**组合级指标**存于 `DataFrame.attrs`：

| attrs 键 | 说明 |
|---|---|
| portfolio_cvar_95 | 组合 95% CVaR（LP 目标最优值，日频损失口径） |
| portfolio_var_95 | 组合 95% VaR（最优 α，日频损失口径） |
| portfolio_expected_annual_return | 组合预期年化收益 |
| target_annual_return / target_daily_return | 目标收益（年化 / 日频） |
| beta / weight_upper | 置信水平 / 权重上限 |
| target_relaxed | 目标收益不可行时是否已放松为 0 |
| weight_upper_relaxed | 权重上限是否已自动放宽（N×upper<1 时放宽到 1.0） |
| effective_weight_upper | 实际生效的权重上限（放宽后=1.0，否则=weight_upper） |
| evt_resampled | `{symbol: 补样条数}`，记录哪些资产触发 EVT 补样 |
| n_scenarios / n_hist_scenarios | 总情景数 / 历史情景数 |

## 模型评价标准

组合优化任务需同时报告**尾部风险控制**与**收益表现**，并与基准对比，不能只给单一收益。

| 分类 | 指标 | 方向 | 说明 |
|---|---|---|---|
| 尾部风险 | 样本外 CVaR@95 | 越小越好 | test 段实现的尾部平均损失 |
| 尾部风险 | 样本外 VaR@95 | 越小越好 | test 段 5% 分位损失 |
| 尾部风险 | 尾部改善(CVaR-等权) | 越负越好 | CVaR组合 CVaR95 − 等权 CVaR95，负值=优化成功 |
| 收益 | 样本外年化收益 | 越大越好 | test 段几何年化 |
| 收益 | 年化波动 / 最大回撤 | 越小越好 | 风险副指标 |
| 综合 | Calmar / 夏普 | 越大越好 | 风险调整后收益 |
| 相对 | 超额年化 / 信息比 | 越大越好 | 相对 benchmark 指数 |

硬性要求：

- **不允许未来函数**：权重只能用 train 段（前半样本）数据求解，test 段仅用于验证。
- **收益口径**：组合日收益 = `Σ w_i · r_{i,t}`，`w` 固定为 train 段解，不随 test 段调整。
- **CVaR/VaR 为损失口径正值**（如 0.03 表示尾部日均损失 3%）。
- **必须对比等权(1/N)与 benchmark 指数**，验证 CVaR 优化是否真的降低尾部损失。
- **EVT 补样只用于求解阶段的情景池**，样本外评价用真实历史收益，不掺补样数据。

## 代码结构（对齐 Alpha 因子开发规则 V2 §4 标准分层）

- `scripts/factor.py` —— **因子计算主脚本，可独立运行**（`python scripts/factor.py`）
  收益矩阵 → GPD 极值理论补尾 → Rockafellar-Uryasev CVaR-LP → 权重表（含 main 入口）
- `scripts/data_loader.py` —— 数据加载层（取数与清洗，不含因子计算）
  panda_data SDK 拉股票池 + 基准指数日线 + parquet 缓存 + 字段标准化
- `scripts/backtest.py` —— 样本外回测 + 评价指标（CVaR/年化/MDD/Calmar + 等权/benchmark 对比）
- `scripts/validate.py` —— 权重契约 + 无未来函数 + 目标收益 + CVaR 一致性
- `scripts/report.py` / `analysis_report.py` —— HTML 报告 + 目标收益扫描有效前沿
- `references/data_guide.md` —— 数据源接口与字段说明（详见该文档）

> 与横截面选股因子不同：CVaR 是**组合优化因子**，输出每资产配置权重
> （非 IC/IR/分层/score/signal）。故 factor.py 的"因子值"= CVaR 最优权重。

## 使用方式

Agent 使用本 skill 时，应在 `scripts` 目录下执行，优先按下面顺序运行：

```bash
python scripts/factor.py      # 求 CVaR 最优权重，打印组合 CVaR/VaR/年化 + 非零权重
python scripts/validate.py    # 权重契约 + 无未来函数 + 目标收益 + CVaR 一致性
python scripts/backtest.py    # 样本外回测：CVaR/年化/MDD/Calmar + 等权/benchmark 对比
python scripts/report.py      # 生成 HTML 报告（4 图 + 8 指标卡 + 三方对比表 + 权重明细）
```

运行前需设置 `PANDA_DATA_USERNAME` 和 `PANDA_DATA_PASSWORD` 环境变量。
可选设置 `PANDA_DATA_START_DATE` / `PANDA_DATA_END_DATE` 控制样本区间。

`report.py` 生成 self-contained 单文件 HTML（`reports/report.html`），含：
- 图 1：三组合累计收益 + CVaR 组合回撤
- 图 2：样本外收益分布直方图 + VaR 标注（尾部对比）
- 图 3：组合权重分布条形图
- 图 4：各资产尾部风险贡献条形图
- 8 张核心指标卡 + CVaR/等权/Benchmark 三方对比表 + 权重明细表

### 离线验证模式（CI 友好）

```bash
# 1. 一次性生成 fixture（需联网 + 凭证）
python scripts/save_fixture.py
# 生成 fixtures/sample_stocks.parquet + sample_index.parquet

# 2. 后续验证无需联网
export PANDA_DATA_OFFLINE=1
python scripts/validate.py
python scripts/report.py --offline
```

- `PANDA_DATA_OFFLINE=1`：从 `fixtures/` 加载固定数据，跳过联网
- 离线模式下若未装 `panda_data` SDK，自动注入 stub 绕过顶层 import
- pyarrow 与旧版 parquet 不兼容时，自动 fallback 到 `fastparquet` 引擎
- 若仅有股票 fixture（无指数），benchmark 对比自动跳过

### 深度分析（参数扫描）

```bash
# 目标收益扫描 + CVaR-收益有效前沿
python scripts/analysis_report.py --targets 0.05,0.08,0.12,0.15,0.20

# 自定义置信水平与日期
python scripts/analysis_report.py --beta 0.99 --start 2024-01-01 --end 2026-07-06
```

生成 `reports/analysis_report.html`：目标收益扫描表 + CVaR-收益有效前沿图。

## Agent 执行规则

1. 先运行 `factor.py`，确认能拉到数据、LP 成功求解、权重和为 1。
2. 再运行 `validate.py`，确认权重契约、无未来函数、目标收益、CVaR 一致性全部通过。
3. 接着运行 `backtest.py`，输出样本外 CVaR、年化、回撤，并与等权 / benchmark 对比。
4. 最后运行 `report.py`，生成 HTML 报告。
5. 任一步失败，必须报告失败命令、错误信息和数据日期，不得直接进入生产。

## 成功标准

- `factor.py` 输出含 `weight`（和为 1）、`tail_contribution` 的权重表 + 组合 CVaR/VaR。
- `validate.py` 输出全部检查通过（权重契约 8 条 + 无未来函数 + 目标收益 + CVaR 一致性）。
- `backtest.py` 至少输出样本外 CVaR95、年化收益、最大回撤，及等权对比。
- `report.py` 生成含 4 图 + 8 指标卡 + 三方对比表 + 权重明细表的 self-contained HTML。
- CVaR/VaR 必须为损失口径正值；样本外评价基于真实历史收益，不掺 EVT 补样数据。
- 权重必须满足非负、上限、和为 1 三约束；目标收益不可行时须标记 `target_relaxed`。

## 验收要求

- 不允许未来函数（权重仅用 train 段求解）。
- 必须有样本外验证。
- 必须对比等权与 benchmark 基准。
- 正式任务必须使用 PandaAI data 或项目指定数据源。
- 不通过验证不得进入生产。

## 依赖

- Python 3.10+
- pandas / numpy
- **scipy**（`linprog` 求 LP，`stats.genpareto` 做 EVT 拟合）
- panda-data（数据源 SDK）
- matplotlib（HTML 报告，`report.py` / `analysis_report.py`）
- fastparquet（pyarrow 与旧版 parquet 不兼容时的 fallback 引擎）
