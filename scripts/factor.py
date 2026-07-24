"""CVaR 尾部风险优化 —— 因子计算层（factor）

"CVaR 因子" = 在"预期收益 ≥ 目标年化"约束下最小化组合 95% CVaR 的最优权重。
本模块纯因子计算，不含数据加载：
    收益矩阵 → GPD 极值理论补足尾部情景 → Rockafellar-Uryasev 线性化 CVaR-LP → 权重表。

与 Alpha 横截面选股因子不同：CVaR 是【组合优化因子】，输出每资产配置权重
（而非 IC/IR/分层/score/signal）。故 SKILL.md 的"因子逻辑/输出结果"按权重口径描述。

数据加载由 data_loader.py 负责（load_stock_data / load_index_data），
本模块仅 `from data_loader import load_stock_data` 供 main 入口使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import genpareto

from data_loader import load_stock_data

# === 因子/模型常量 ===============================================================
MODEL_ID = "CVAR1"
MODEL_NAME = "CVaR 尾部风险优化"
BETA = 0.95                    # CVaR 置信水平（95% → 关注最差 5% 的尾部损失）
TARGET_ANNUAL_RETURN = 0.08    # 预期收益约束：组合年化 ≥ 8%
WEIGHT_UPPER_BOUND = 0.30      # 单资产权重上限（防止尾部驱动的极端集中）
TRADING_DAYS = 252             # 年化换算用的交易日数

# EVT 尾部补样参数
EVT_MIN_TAIL_SAMPLES = 30      # 尾部超阈样本数 < 此值时触发 GPD 补样（默认1年数据尾部~25个也触发，补足尾部稳定 CVaR 估计）
EVT_TAIL_QUANTILE = 0.10       # 取最差 10% 收益作为"尾部"进行 GPD 拟合
EVT_RESAMPLE_SIZE = 200        # GPD 拟合后重采样补足的尾部情景数
EVT_RANDOM_SEED = 42           # 补样随机种子，保证结果可复现（无未来函数检查依赖此确定性）


def build_return_matrix(stock_df: pd.DataFrame) -> pd.DataFrame:
    """把长表股票数据透视为 T×N 日收益矩阵（index=date, columns=symbol）

    收益 = close / close.shift(1) - 1，按品种独立计算。
    ffill 显式前向填充等价原 pct_change() 默认 pad 行为（pandas 2.1+ 弃用 fill_method
    默认值，显式 fill_method=None 关闭 pct_change 内部填充由 ffill 负责）。
    股票停牌的中间缺失日用前收盘价填（收益记 0），dropna(how="any") 再丢前导缺失行。
    """
    price = stock_df.pivot(index="date", columns="symbol", values="close").sort_index()
    # 日收益率（简单收益）
    ret = price.ffill().pct_change(fill_method=None).dropna(how="any")
    if ret.empty or ret.shape[1] < 2:
        raise ValueError(f"收益矩阵为空或资产数不足 2（shape={ret.shape}）")
    return ret


def fit_gpd_tail_scenarios(losses: np.ndarray) -> np.ndarray | None:
    """对单资产损失序列的尾部用 GPD 拟合并重采样补足尾部情景

    极值理论（EVT）Peaks-Over-Threshold 方法：
      1. 取损失序列（损失 = -收益）的最差 EVT_TAIL_QUANTILE 分位作为阈值 u
      2. 超阈量 (loss - u | loss > u) 渐近服从广义帕累托分布 GPD(ξ, σ)
      3. 用 MLE 拟合 (ξ, σ)，再从 GPD 重采样补足 EVT_RESAMPLE_SIZE 个尾部超阈样本
      4. 返回补足后的完整损失情景（原尾部 + 重采样尾部），供 CVaR 情景池使用

    仅当尾部样本 < EVT_MIN_TAIL_SAMPLES 时才调用（历史尾部样本足够则不补）。

    Args:
        losses: 单资产损失序列（1D，损失 = -日收益，正值代表亏损）

    Returns:
        补足后的尾部损失样本（1D）；拟合失败返回 None（上层降级为纯历史）
    """
    threshold = np.quantile(losses, 1.0 - EVT_TAIL_QUANTILE)
    excess = losses[losses > threshold] - threshold
    if len(excess) < 5:
        return None  # 超阈样本太少，无法可靠拟合
    try:
        # floc=0：GPD 起点固定在阈值处（超阈量从 0 起）
        shape, loc, scale = genpareto.fit(excess, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            return None
        # 固定 random_state 保证补样可复现（否则同数据两次求解权重不同 → 无未来函数检查失败）
        resampled_excess = genpareto.rvs(
            shape, loc=0, scale=scale, size=EVT_RESAMPLE_SIZE,
            random_state=np.random.RandomState(EVT_RANDOM_SEED),
        )
        # 还原为损失量（阈值 + 超阈量），并只保留正的合理损失
        resampled_losses = threshold + resampled_excess
        resampled_losses = resampled_losses[np.isfinite(resampled_losses)]
        return resampled_losses
    except Exception:
        return None


def build_scenario_matrix(ret: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """构建 CVaR 优化用的情景矩阵（T'×N），必要时用 GPD 补足尾部情景

    对每个资产检查左尾样本数，不足则用 fit_gpd_tail_scenarios 补样。
    补样策略：把补出的尾部损失情景（转回收益 = -loss）作为触发资产 j 该行的收益，
    其它资产在该补充情景行取其【历史均值】（非尾部分位）——只补单资产 j 的边际尾部，
    不人为制造"全资产同日崩盘"的共尾情景，避免扭曲联合分布、系统性放大 CVaR。

    Returns:
        (scenario_returns, evt_info)
        scenario_returns: N 列的情景收益矩阵（历史 + 补充尾部）
        evt_info: {symbol: 补样条数} 记录哪些资产触发了 EVT 补样
    """
    hist = ret.values  # T×N
    symbols = ret.columns.tolist()
    n_assets = hist.shape[1]

    extra_rows: list[np.ndarray] = []
    evt_info: dict[str, int] = {}

    for j, sym in enumerate(symbols):
        col_ret = hist[:, j]
        losses = -col_ret
        # 尾部样本数 = 超过尾部分位阈值的样本数
        threshold = np.quantile(losses, 1.0 - EVT_TAIL_QUANTILE)
        tail_n = int((losses > threshold).sum())
        if tail_n >= EVT_MIN_TAIL_SAMPLES:
            continue  # 历史尾部样本充足，不补
        resampled_losses = fit_gpd_tail_scenarios(losses)
        if resampled_losses is None or len(resampled_losses) == 0:
            continue
        evt_info[sym] = len(resampled_losses)
        # 为每条补充尾部情景生成一整行收益：
        #   触发资产 j 用补出的损失（收益 = -loss）；
        #   其它资产 k 取其【历史均值】（非尾部分位）——只补 j 的边际尾部，
        #   不造共尾，避免人为放大组合 CVaR
        other_mean_ret = {k: float(np.mean(hist[:, k])) for k in range(n_assets)}
        for loss in resampled_losses:
            row = np.array([other_mean_ret[k] for k in range(n_assets)], dtype=float)
            row[j] = -loss
            extra_rows.append(row)

    if extra_rows:
        scenario = np.vstack([hist, np.array(extra_rows)])
        print(f"[EVT] 触发 GPD 尾部补样的资产: {evt_info}，情景数 {hist.shape[0]} → {scenario.shape[0]}")
    else:
        scenario = hist
        print(f"[EVT] 所有资产历史尾部样本充足（≥{EVT_MIN_TAIL_SAMPLES}），未补样")

    return scenario, evt_info


def solve_cvar_lp(
    scenario_returns: np.ndarray,
    expected_returns: np.ndarray,
    beta: float = BETA,
    target_daily_return: float = 0.0,
    weight_upper: float = WEIGHT_UPPER_BOUND,
) -> dict:
    """Rockafellar-Uryasev CVaR 最小化线性规划

    数学模型（β 置信水平，T 个情景，N 个资产）：
        决策变量：w ∈ R^N（权重）, α ∈ R（VaR 辅助）, u ∈ R^T（超损失辅助）
        目标：  min  α + 1/((1-β)·T) · Σ_t u_t
        约束：  u_t ≥ -(r_t · w) - α        (损失超过 VaR 的部分)
                u_t ≥ 0
                Σ_i w_i = 1                   (权重归一)
                0 ≤ w_i ≤ weight_upper        (非负 + 集中度上限)
                (μ · w) ≥ target_daily_return (预期收益约束)
        其中 r_t 是第 t 个情景的资产收益向量，损失 = -(r_t · w)，
        CVaR_β = 目标函数最优值 = 最差 (1-β) 比例情景的平均损失。

    linprog 变量顺序：x = [w_0..w_{N-1}, α, u_0..u_{T-1}]，共 N+1+T 维。

    Returns:
        {weights, var_95, cvar_95, expected_daily_return, success, message}
    """
    T, N = scenario_returns.shape
    n_var = N + 1 + T
    coef = 1.0 / ((1.0 - beta) * T)

    # --- 目标函数 c：min α + coef·Σu ---
    c = np.zeros(n_var)
    c[N] = 1.0                       # α 系数
    c[N + 1:] = coef                 # u 系数

    # --- 不等式约束 A_ub·x ≤ b_ub ---
    # (1) u_t ≥ -(r_t·w) - α  ⇔  -(r_t·w) - α - u_t ≤ 0
    #     即  -r_t·w - α - u_t ≤ 0
    A_scenario = np.zeros((T, n_var))
    A_scenario[:, :N] = -scenario_returns   # -r_t·w
    A_scenario[:, N] = -1.0                 # -α
    for t in range(T):
        A_scenario[t, N + 1 + t] = -1.0     # -u_t
    b_scenario = np.zeros(T)

    # (2) 预期收益约束：μ·w ≥ target ⇔ -μ·w ≤ -target
    A_ret = np.zeros((1, n_var))
    A_ret[0, :N] = -expected_returns
    b_ret = np.array([-target_daily_return])

    A_ub = np.vstack([A_scenario, A_ret])
    b_ub = np.concatenate([b_scenario, b_ret])

    # --- 等式约束 A_eq·x = b_eq：Σw = 1 ---
    A_eq = np.zeros((1, n_var))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # --- 变量边界 ---
    bounds = (
        [(0.0, weight_upper)] * N   # w：非负 + 上限
        + [(None, None)]            # α：自由
        + [(0.0, None)] * T         # u：非负
    )

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "weights": None,
            "var_95": None,
            "cvar_95": None,
            "expected_daily_return": None,
        }

    w = res.x[:N]
    alpha = res.x[N]  # VaR（损失口径，正值代表损失）
    cvar = res.fun    # 目标函数最优值 = CVaR
    exp_daily = float(expected_returns @ w)

    return {
        "success": True,
        "message": res.message,
        "weights": w,
        "var_95": float(alpha),
        "cvar_95": float(cvar),
        "expected_daily_return": exp_daily,
    }


def solve_cvar_with_relax(
    scenario_returns: np.ndarray,
    expected_returns: np.ndarray,
    n_assets: int,
    beta: float = BETA,
    target_daily_return: float = 0.0,
    weight_upper: float = WEIGHT_UPPER_BOUND,
) -> tuple[dict, dict]:
    """带三档降级的 CVaR-LP 求解（避免不可行时硬崩溃）

    降级顺序：
        1. 原参数求解
        2. 收益约束放松到 0（target_relaxed，应对目标收益过高/历史均值达不到）
        3. 上限放宽到 1.0（weight_upper_relaxed，应对 N×upper<1 导致归一不可行）
    第 3 档后 N 个资产各 1/N 必满足归一与非负，必可行（N≥1）。

    Returns:
        (result, relax_info) —— result 为 solve_cvar_lp 的 success dict；
        relax_info = {target_relaxed, weight_upper_relaxed, effective_upper}
    """
    relax = {"target_relaxed": False, "weight_upper_relaxed": False,
             "effective_upper": weight_upper}

    # 1. 原参数
    res = solve_cvar_lp(scenario_returns, expected_returns, beta=beta,
                        target_daily_return=target_daily_return, weight_upper=weight_upper)
    if res["success"]:
        return res, relax

    # 2. 收益约束放松到 0
    print(f"[WARN] LP 不可行（{res['message']}），放松收益约束为 0 重解")
    relax["target_relaxed"] = True
    res = solve_cvar_lp(scenario_returns, expected_returns, beta=beta,
                        target_daily_return=0.0, weight_upper=weight_upper)
    if res["success"]:
        return res, relax

    # 3. 上限过紧（N×upper<1 无法归一）→ 放宽到 1.0，等权必可行
    if n_assets * weight_upper < 1.0 - 1e-9:
        new_upper = 1.0
        print(f"[WARN] 权重上限 {weight_upper} 过紧（N×upper<1 无法归一），放宽到 {new_upper} 重解")
        relax["weight_upper_relaxed"] = True
        relax["effective_upper"] = new_upper
        res = solve_cvar_lp(scenario_returns, expected_returns, beta=beta,
                            target_daily_return=0.0, weight_upper=new_upper)
        if res["success"]:
            return res, relax

    raise ValueError(f"CVaR-LP 三档降级后仍不可行: {res['message']}")


def optimize_portfolio(
    stock_df: pd.DataFrame | None = None,
    target_annual_return: float = TARGET_ANNUAL_RETURN,
    beta: float = BETA,
    weight_upper: float = WEIGHT_UPPER_BOUND,
    update_time: str | None = None,
) -> pd.DataFrame:
    """完整流程：加载数据 → 收益矩阵 → EVT 补尾 → CVaR-LP → 权重表

    Args:
        stock_df: 可选，直接传入股票日线（date/symbol/close）。None 则联网加载默认池。
        target_annual_return: 目标年化收益（如 0.08 = 8%），转日频后作为 LP 约束。
        beta: CVaR 置信水平（默认 0.95）。
        weight_upper: 单资产权重上限（默认 0.30）。
        update_time: 结果生成时间；None 时按数据最新日期推导（可复现）。

    Returns:
        权重表 DataFrame，每个资产一行，字段见 SKILL.md 输出契约。
        组合级指标存放在 DataFrame.attrs（portfolio_cvar_95 等）。
    """
    if stock_df is None:
        stock_df = load_stock_data()

    ret = build_return_matrix(stock_df)
    symbols = ret.columns.tolist()

    # 预期收益：用样本期日均收益（μ_i）
    mu = ret.mean().values
    # 目标年化 → 日频（几何口径近似：(1+annual)^(1/252) - 1）
    target_daily = (1.0 + target_annual_return) ** (1.0 / TRADING_DAYS) - 1.0

    # 情景矩阵（历史 + GPD 尾部补样）
    scenario, evt_info = build_scenario_matrix(ret)

    # 三档降级求解：原参数 → target=0 → upper 放宽到 1.0，避免不可行时硬崩溃
    result, relax_info = solve_cvar_with_relax(
        scenario, mu, n_assets=len(symbols), beta=beta,
        target_daily_return=target_daily, weight_upper=weight_upper)
    relaxed = relax_info["target_relaxed"]

    w = result["weights"]

    # 边际尾部贡献：资产 i 对组合 CVaR 的近似贡献 = w_i · E[-r_i | 组合处于最差尾部]
    # 找出组合损失最差的 (1-β) 比例情景，计算各资产在这些情景的平均损失，乘以权重
    port_loss = -(scenario @ w)
    n_tail = max(1, int(np.ceil((1.0 - beta) * len(port_loss))))
    tail_idx = np.argsort(port_loss)[-n_tail:]   # 损失最大的 n_tail 个情景
    tail_asset_loss = -(scenario[tail_idx].mean(axis=0))  # 各资产在尾部情景的平均损失
    tail_contribution = w * tail_asset_loss

    update_time = update_time or _make_update_time(stock_df)
    latest_date = str(stock_df["date"].astype(str).max())
    trade_date = f"{latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:8]}"

    out = pd.DataFrame({
        "trade_date": trade_date,
        "asset_type": "stock",
        "symbol": symbols,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "weight": np.round(w, 6),
        "expected_return": np.round(mu, 6),                 # 日均预期收益
        "expected_annual_return": np.round((1.0 + mu) ** TRADING_DAYS - 1.0, 6),
        "tail_contribution": np.round(tail_contribution, 6),
        "data_version": "real-v1",
        "update_time": update_time,
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    # 组合级指标（存 attrs，供 validate / backtest / report 使用）
    out.attrs["portfolio_cvar_95"] = round(result["cvar_95"], 6)
    out.attrs["portfolio_var_95"] = round(result["var_95"], 6)
    out.attrs["portfolio_expected_daily_return"] = round(result["expected_daily_return"], 6)
    out.attrs["portfolio_expected_annual_return"] = round(
        (1.0 + result["expected_daily_return"]) ** TRADING_DAYS - 1.0, 6
    )
    out.attrs["target_annual_return"] = target_annual_return
    out.attrs["target_daily_return"] = round(target_daily, 8)
    out.attrs["beta"] = beta
    out.attrs["weight_upper"] = weight_upper
    out.attrs["effective_weight_upper"] = relax_info["effective_upper"]
    out.attrs["weight_upper_relaxed"] = relax_info["weight_upper_relaxed"]
    out.attrs["target_relaxed"] = relaxed
    out.attrs["evt_resampled"] = evt_info
    out.attrs["n_scenarios"] = int(scenario.shape[0])
    out.attrs["n_hist_scenarios"] = int(ret.shape[0])

    return out


def _make_update_time(stock_df: pd.DataFrame) -> str:
    """根据数据最新日期推导 update_time（数据最新日 + A股收盘 15:30，可复现）"""
    latest_date = str(stock_df["date"].astype(str).max())  # YYYYMMDD
    return f"{latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:8]}T15:30:00"


if __name__ == "__main__":
    result = optimize_portfolio()

    print(f"\n优化完成，组合含 {len(result)} 个资产")
    print(f"组合 95% CVaR（日频损失）: {result.attrs['portfolio_cvar_95']:.4%}")
    print(f"组合 95% VaR（日频损失）: {result.attrs['portfolio_var_95']:.4%}")
    print(f"组合预期年化收益: {result.attrs['portfolio_expected_annual_return']:.4%}"
          f"（目标 {result.attrs['target_annual_return']:.1%}）")
    print(f"情景数: {result.attrs['n_hist_scenarios']} 历史 + "
          f"{result.attrs['n_scenarios'] - result.attrs['n_hist_scenarios']} EVT 补样")

    print("\n非零权重资产:")
    nonzero = result[result["weight"] > 1e-6]
    print(nonzero[["symbol", "weight", "expected_annual_return", "tail_contribution"]].to_string(index=False))
