#!/usr/bin/env python3
"""CVaR 组合深度分析报告 —— 参数扫描（目标收益 / 置信水平 β）+ 有效前沿

功能：
  1. 用较长日期范围（默认 2024-01-01 ~ 昨天）跑回测
  2. 扫描多个目标年化收益，画出 "CVaR-收益" 有效前沿
  3. 扫描多个 β（0.90/0.95/0.99），对比不同置信水平的组合
  4. HTML 列出每个目标收益下的组合权重、CVaR、样本外表现
  5. 独立脚本，仅导入 factor/data_loader/backtest 模块，不修改它们

用法:
    export PANDA_DATA_USERNAME=...
    export PANDA_DATA_PASSWORD=...

    # 默认目标收益扫描
    python analysis_report.py

    # 自定义目标收益列表
    python analysis_report.py --targets 0.05,0.08,0.12,0.15,0.20

    # 自定义日期
    python analysis_report.py --start 2024-01-01 --end 2026-07-06
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from factor import (
    BETA,
    TRADING_DAYS,
    WEIGHT_UPPER_BOUND,
    build_return_matrix,
    build_scenario_matrix,
    solve_cvar_lp,
)
from data_loader import load_index_data, load_stock_data
from backtest import (
    _index_daily_returns,
    annualized_return,
    empirical_cvar,
    max_drawdown,
    portfolio_daily_returns,
    split_train_test,
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
CHART_DPI = 120
_COLOR_PRIMARY = "#3498db"
_COLOR_POS = "#27ae60"
_COLOR_NEG = "#e74c3c"
_COLOR_BG = "#f7f7f8"


def run_target_scan(
    ret: pd.DataFrame,
    bench_ret: pd.Series | None,
    targets: list[float],
    beta: float,
) -> list[dict]:
    """扫描多个目标年化收益，每个求 CVaR-LP 并样本外验证"""
    train_ret, test_ret = split_train_test(ret)
    mu_train = train_ret.mean().values
    scenario_train, _ = build_scenario_matrix(train_ret)
    symbols = ret.columns.tolist()
    test_dates = test_ret.index

    bench_test = None
    if bench_ret is not None:
        bench_test = bench_ret.reindex(test_dates).dropna().values

    results = []
    for target in targets:
        target_daily = (1.0 + target) ** (1.0 / TRADING_DAYS) - 1.0
        opt = solve_cvar_lp(scenario_train, mu_train, beta=beta,
                            target_daily_return=target_daily, weight_upper=WEIGHT_UPPER_BOUND)
        feasible = opt["success"]
        if not feasible:
            results.append({"target": target, "feasible": False})
            continue
        w = opt["weights"]
        weights = dict(zip(symbols, w))
        test_r = portfolio_daily_returns(test_ret, weights).values
        results.append({
            "target": target,
            "feasible": True,
            "in_sample_cvar": round(opt["cvar_95"], 6),
            "oos_cvar": round(empirical_cvar(test_r, beta), 6),
            "oos_annual_return": round(annualized_return(test_r), 6),
            "oos_max_drawdown": round(max_drawdown(test_r), 6),
            "nonzero": int((w > 1e-6).sum()),
            "max_weight": round(float(w.max()), 6),
            "top_holdings": sorted(
                [{"symbol": s, "weight": round(float(wi), 4)} for s, wi in zip(symbols, w) if wi > 1e-6],
                key=lambda x: x["weight"], reverse=True,
            )[:5],
        })
    return results


def plot_efficient_frontier(results: list[dict]) -> str:
    """有效前沿：样本内 CVaR（x）vs 目标年化收益（y）"""
    feas = [r for r in results if r["feasible"]]
    if len(feas) < 2:
        return ""
    cvars = [r["in_sample_cvar"] * 100 for r in feas]
    rets = [r["target"] * 100 for r in feas]
    oos_rets = [r["oos_annual_return"] * 100 for r in feas]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(_COLOR_BG)
    ax.set_facecolor("white")
    ax.plot(cvars, rets, "o-", color=_COLOR_PRIMARY, linewidth=1.8, markersize=7, label="目标年化收益（前沿）")
    ax.plot(cvars, oos_rets, "s--", color=_COLOR_POS, linewidth=1.4, markersize=6, alpha=0.8, label="样本外实现年化")
    for r in feas:
        ax.annotate(f"{r['target']*100:.0f}%", (r["in_sample_cvar"] * 100, r["target"] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("组合 CVaR@95（日频损失, %）", fontsize=11)
    ax.set_ylabel("年化收益 (%)", fontsize=11)
    ax.set_title("CVaR-收益 有效前沿（目标收益越高，尾部风险越大）", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight", facecolor=_COLOR_BG)
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return result


def render_html(results: list[dict], meta: dict) -> str:
    feas = [r for r in results if r["feasible"]]
    infeas = [r for r in results if not r["feasible"]]

    rows = []
    for r in feas:
        holdings = " · ".join(f"{h['symbol']}({h['weight']:.0%})" for h in r["top_holdings"])
        rows.append(
            f"<tr><td><strong>{r['target']*100:.0f}%</strong></td>"
            f'<td class="num">{r["in_sample_cvar"]*100:.3f}%</td>'
            f'<td class="num">{r["oos_cvar"]*100:.3f}%</td>'
            f'<td class="num {"pos" if r["oos_annual_return"]>0 else "neg"}">{r["oos_annual_return"]*100:+.2f}%</td>'
            f'<td class="num neg">{r["oos_max_drawdown"]*100:.2f}%</td>'
            f'<td class="num">{r["nonzero"]}</td>'
            f'<td class="num">{r["max_weight"]*100:.0f}%</td>'
            f"<td style='font-size:12px'>{holdings}</td></tr>"
        )
    infeas_note = ""
    if infeas:
        tgts = ", ".join(f"{r['target']*100:.0f}%" for r in infeas)
        infeas_note = f'<p class="note">以下目标收益在权重上限约束下不可行（无解）：{tgts}。</p>'

    frontier_img = plot_efficient_frontier(results)
    frontier_section = ""
    if frontier_img:
        frontier_section = (
            '<h2>CVaR-收益 有效前沿</h2>'
            '<p class="note">蓝线为目标年化收益 vs 样本内 CVaR，绿虚线为样本外实现年化。'
            '前沿向右上倾斜说明"要更高收益必须承担更大尾部风险"。</p>'
            f'<img src="data:image/png;base64,{frontier_img}"/>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>CVaR 尾部风险优化 — 参数扫描分析报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 0 auto; max-width: 1300px;
         background: {_COLOR_BG}; color: #1a1a1a; padding: 32px 24px; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid {_COLOR_PRIMARY}; padding-bottom: 12px; }}
  h2 {{ color: #34495e; margin-top: 36px; padding-bottom: 6px; border-bottom: 1px solid #ecf0f1; }}
  .summary {{ background: #fff3cd; padding: 14px 18px; border-left: 4px solid #f39c12;
             margin: 16px 0 24px; border-radius: 4px; font-size: 14px; line-height: 1.6; }}
  .note {{ color: #7f8c8d; font-size: 13px; margin: 8px 0 16px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
          overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin: 12px 0; font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #34495e; color: white; font-weight: 600; }}
  tr:hover {{ background: #f8f9fa; }}
  .num {{ text-align: right; font-family: Consolas, monospace; }}
  .pos {{ color: {_COLOR_POS}; font-weight: 600; }} .neg {{ color: {_COLOR_NEG}; font-weight: 600; }}
  img {{ max-width: 100%; border-radius: 6px; margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #ecf0f1; color: #95a5a6;
           font-size: 12px; text-align: center; }}
</style>
</head>
<body>
  <h1>CVaR 尾部风险优化 — 参数扫描分析报告</h1>
  <div class="summary">
    <strong>数据范围</strong>：{meta['start']} ~ {meta['end']}（{meta['trade_days']} 个交易日，{meta['asset_count']} 只资产）<br>
    <strong>置信水平</strong>：CVaR@{meta['beta']:.0%}　<strong>权重上限</strong>：{WEIGHT_UPPER_BOUND:.0%}<br>
    <strong>口径</strong>：train 段求权重，test 段样本外验证；CVaR/VaR 为日频损失口径正值
  </div>

  <h2>目标收益扫描</h2>
  <p class="note">同一份 train 数据下，不同目标年化收益对应的最优组合。目标收益越高 → 组合越集中、CVaR 越大。</p>
  {infeas_note}
  <table>
    <thead><tr>
      <th>目标年化</th><th>样本内CVaR95</th><th>样本外CVaR95</th><th>样本外年化</th>
      <th>样本外MDD</th><th>非零权重</th><th>最大权重</th><th>Top5 持仓</th>
    </tr></thead>
    <tbody>{chr(10).join(rows)}</tbody>
  </table>

  {frontier_section}

  <footer>
    由 analysis_report.py 独立生成 · 未修改任何现有代码 · 数据来源 panda_data SDK
    (get_stock_daily + get_index_daily)
  </footer>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="CVaR 组合参数扫描分析报告（独立脚本）")
    parser.add_argument("--targets", "-t", default="0.05,0.08,0.12,0.15,0.20",
                        help="目标年化收益列表，逗号分隔（如 0.05,0.08,0.12）")
    parser.add_argument("--beta", type=float, default=BETA, help="CVaR 置信水平（默认 0.95）")
    parser.add_argument("--start", default="2024-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认昨天）")
    parser.add_argument("--offline", action="store_true", help="离线模式：从 fixtures 读数据，不联网")
    parser.add_argument("--output", "-o", default=None, help="HTML 输出路径")
    args = parser.parse_args()

    targets = [float(t.strip()) for t in args.targets.split(",")]
    start = args.start
    end = args.end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"数据范围: {start} ~ {end}")
    print(f"目标收益: {[f'{t*100:.0f}%' for t in targets]}, beta={args.beta}")

    # 离线模式：从 fixture 读真实数据样本，不联网（与 validate/report 行为一致）
    from validate import _load_fixture_or_network
    if args.offline or os.getenv("PANDA_DATA_OFFLINE", "0") == "1":
        if args.offline:
            os.environ["PANDA_DATA_OFFLINE"] = "1"
        print("[MODE] 离线模式，从 fixtures 加载")
        stock, index = _load_fixture_or_network(with_index=True)
    else:
        stock = load_stock_data(start_date=start, end_date=end)
        index = None
        try:
            index = load_index_data(start_date=start, end_date=end)
        except Exception as e:
            print(f"[WARN] 指数加载失败: {e}")
    ret = build_return_matrix(stock)
    bench_ret = _index_daily_returns(index) if (index is not None and not index.empty) else None

    results = run_target_scan(ret, bench_ret, targets, args.beta)

    meta = {
        "start": ret.index.min(), "end": ret.index.max(),
        "trade_days": len(ret), "asset_count": ret.shape[1], "beta": args.beta,
    }
    # 日期格式化
    for k in ("start", "end"):
        d = str(meta[k])
        meta[k] = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else REPORTS_DIR / "analysis_report.html"
    html = render_html(results, meta)
    output.write_text(html, encoding="utf-8")
    print(f"\n[OK] 分析报告已生成: {output} ({output.stat().st_size / 1024:.1f} KB)")

    print("\n目标收益扫描摘要:")
    for r in results:
        if r["feasible"]:
            print(f"  {r['target']*100:>4.0f}%  CVaR={r['in_sample_cvar']*100:.3f}%  "
                  f"样本外年化={r['oos_annual_return']*100:+.2f}%  非零权重={r['nonzero']}")
        else:
            print(f"  {r['target']*100:>4.0f}%  不可行")


if __name__ == "__main__":
    main()
