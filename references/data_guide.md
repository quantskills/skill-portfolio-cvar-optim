# 数据源与字段说明

## 数据来源

本模型使用 Panda data SDK 拉取 A 股股票与指数日线数据。

**正式组合优化开发必须使用 PandaAI data 数据拉取库或项目明确指定的数据源。**
不得使用来源不明、字段不稳定、个人临时整理的数据文件作为正式输入。

如不确定数据源、字段口径或权限配置，必须咨询项目工作人员后再开发。

## Panda data SDK 接口

本模型使用两个接口：

### 1. `get_stock_daily` — 股票池日线（可配置资产）

获取 A 股不复权日线行情，用于构建 T×N 收益矩阵求权重。

- 函数名：`get_stock_daily`
- 参数：
  - `start_date`：开始日期（格式 YYYYMMDD）
  - `end_date`：结束日期（格式 YYYYMMDD）
  - `symbol`：股票代码（带交易所后缀，如 `["600519.SH"]`）
- 返回字段经 `_normalize_price` 标准化为 `date` / `symbol` / `close`：
  - `date` → 交易日期（YYYYMMDD）
  - `symbol` → 股票代码（别名 `code`/`ts_code` 自动映射）
  - `close` → 收盘价（用于计算日收益 `close/close.shift(1) - 1`）

### 2. `get_index_daily` — 基准指数日线（benchmark）

获取指数日线行情，作为 benchmark 计算组合超额收益 / 信息比。

- 函数名：`get_index_daily`
- 参数：
  - `start_date`：开始日期（YYYYMMDD）
  - `end_date`：结束日期（YYYYMMDD）
  - `symbol`：指数代码（如 `"000300.SH"` 沪深300）
- 返回字段同样标准化为 `date` / `symbol` / `close`。

## 字段映射防御

`_normalize_price` 自动处理接口字段名差异：

| 接口实际字段 | 映射目标 | 说明 |
|---|---|---|
| `code` / `ts_code` / `sec_code` | `symbol` | 股票/指数代码 |
| `trade_date` / `trade_dt` / `datetime` | `date` | 交易日期 |

映射发生时打印 `[INFO]` 提示，便于排查契约不匹配。

## 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `PANDA_DATA_USERNAME` | 是 | Panda data 账号 |
| `PANDA_DATA_PASSWORD` | 是 | Panda data 密码 |
| `PANDA_DATA_START_DATE` | 否 | 样本起始日期（YYYY-MM-DD） |
| `PANDA_DATA_END_DATE` | 否 | 样本结束日期（YYYY-MM-DD） |
| `PANDA_DATA_OFFLINE` | 否 | 设为 1 启用离线模式（读 fixtures） |
| `PANDA_DATA_DEBUG` | 否 | 设为 1 打印接口返回字段与前 3 行 |

## 默认配置

- **默认股票池**：沪深300 中 20 只流动性好、跨行业的成分股（`data_loader.DEFAULT_STOCK_POOL`）
- **默认基准**：`000300.SH`（沪深300）
- **默认回溯窗口**：1 年（CVaR 尾部估计需要足够样本）

如需更换资产池或基准，传入 `symbols=` / `benchmark=` 参数即可。
