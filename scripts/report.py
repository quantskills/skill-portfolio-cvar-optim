"""生成 CVaR 尾部风险优化组合的回测 HTML 报告

用法:
    # 离线模式（用 fixtures 数据，无需凭证）
    python scripts/report.py --offline

    # 联网模式（需要 PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD）
    python scripts/report.py

    # 自定义输出 / 自动打开浏览器
    python scripts/report.py --offline --html-path reports/custom.html --open

输出:
    reports/backtest_result.json — 中间产物，含全部时序数据
    reports/report.html          — 最终 HTML 报告（self-contained，base64 内嵌图表）
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，必须在 import pyplot 前设置
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体：图表标题/图例含中文，必须设置否则显示方块（Glyph missing）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from backtest_report_data import (  # noqa: E402
    _is_offline,
    run_backtest_with_series,
    save_backtest_result,
)

# === 配色 =======================================================================
_COLOR_CVAR = "#3b82f6"     # 蓝：CVaR 组合
_COLOR_EQUAL = "#f59e0b"    # 橙：等权组合
_COLOR_BENCH = "#94a3b8"    # 灰：benchmark
_COLOR_LOSS = "#ef4444"     # 红：损失 / 回撤
_COLOR_POS = "#22c55e"      # 绿

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  line-height: 1.6; color: #1f2937; background: #f9fafb;
  max-width: 1200px; margin: 0 auto; padding: 24px;
}
header { padding: 16px 0 24px; border-bottom: 2px solid #e5e7eb; margin-bottom: 24px; }
h1 { font-size: 24px; color: #111827; margin-bottom: 8px; font-weight: 600; }
h2 { font-size: 18px; color: #374151; margin: 32px 0 16px; padding-left: 12px; border-left: 4px solid #3b82f6; }
.meta { font-size: 13px; color: #6b7280; }
.cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 3px solid #3b82f6; transition: transform 0.15s; }
.card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.card .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 22px; font-weight: 600; color: #111827; margin-top: 4px; }
.card .unit { font-size: 13px; color: #6b7280; font-weight: normal; margin-left: 2px; }
.card.positive { border-left-color: #22c55e; } .card.positive .value { color: #16a34a; }
.card.negative { border-left-color: #ef4444; } .card.negative .value { color: #dc2626; }
img { max-width: 100%; height: auto; margin: 12px 0; border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: block; }
.note { font-size: 13px; color: #6b7280; margin: 8px 0; padding: 12px; background: #f3f4f6;
        border-radius: 6px; line-height: 1.7; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
        overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; }
th { background: #f3f4f6; padding: 12px; text-align: left; font-size: 13px; color: #374151; font-weight: 600; }
td { padding: 10px 12px; border-top: 1px solid #e5e7eb; font-size: 13px; color: #4b5563; }
tr:hover td { background: #f9fafb; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: #16a34a; font-weight: 600; } .neg { color: #dc2626; font-weight: 600; }
footer { margin-top: 40px; padding: 20px 24px; background: #fef3c7; border-radius: 8px;
         border-left: 4px solid #f59e0b; }
footer h3 { font-size: 14px; color: #92400e; margin-bottom: 8px; font-weight: 600; }
footer p { font-size: 13px; color: #78350f; line-height: 1.7; }
"""


# === 图表 =======================================================================

def _fig_to_base64(fig, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _format_x_dates(ax, dates: list[str]) -> None:
    if len(dates) > 8:
        step = max(1, len(dates) // 8)
        idx = list(range(0, len(dates), step))
        ax.set_xticks(idx)
        ax.set_xticklabels([dates[i] for i in idx], rotation=30, ha="right", fontsize=9)


def plot_equity_curve(ts: dict, dpi: int) -> str:
    """图 1: 三组合累计收益（上）+ CVaR 组合回撤（下）"""
    cvar = ts["cvar_curve"]
    equal = ts["equal_curve"]
    bench = ts.get("bench_curve") or []
    dd = ts["cvar_drawdown"]

    dates = [d["date"] for d in cvar]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dates, [d["cum_return"] for d in cvar], color=_COLOR_CVAR, linewidth=1.6, label="CVaR 组合")
    ax1.plot(dates, [d["cum_return"] for d in equal], color=_COLOR_EQUAL, linewidth=1.2, label="等权组合", alpha=0.85)
    if bench:
        bdates = [d["date"] for d in bench]
        ax1.plot(bdates, [d["cum_return"] for d in bench], color=_COLOR_BENCH, linewidth=1.2,
                 label="Benchmark", alpha=0.85, linestyle="--")
    ax1.axhline(0, color="#9ca3af", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    _format_x_dates(ax1, dates)

    dd_dates = [d["date"] for d in dd]
    ax2.fill_between(dd_dates, [d["drawdown"] for d in dd], 0, color=_COLOR_LOSS, alpha=0.4, label="CVaR 组合回撤")
    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    _format_x_dates(ax2, dd_dates)

    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_loss_distribution(hist: dict, dpi: int) -> str:
    """图 2: 样本外收益分布直方图 + VaR 标注（尾部对比）"""
    edges = np.array(hist["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.45

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(centers - width / 2, hist["cvar_counts"], width=width, color=_COLOR_CVAR, alpha=0.75, label="CVaR 组合")
    ax.bar(centers + width / 2, hist["equal_counts"], width=width, color=_COLOR_EQUAL, alpha=0.7, label="等权组合")
    # VaR 竖线（损失口径 → 收益侧取负）
    ax.axvline(-hist["cvar_var95"], color=_COLOR_CVAR, linestyle="--", linewidth=1.3,
               label=f"CVaR组合 VaR95={hist['cvar_var95']:.2%}")
    ax.axvline(-hist["equal_var95"], color=_COLOR_EQUAL, linestyle="--", linewidth=1.3,
               label=f"等权 VaR95={hist['equal_var95']:.2%}")
    ax.axvline(0, color="#9ca3af", linewidth=0.5)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_weights(weights: list[dict], dpi: int, top_n: int = 20) -> str:
    """图 3: 组合权重条形图（水平，Top N 非零权重）"""
    nonzero = [w for w in weights if w["weight"] > 1e-6][:top_n]
    nonzero = list(reversed(nonzero))  # Top 1 显示在最上
    syms = [w["symbol"] for w in nonzero]
    vals = [w["weight"] * 100 for w in nonzero]

    fig, ax = plt.subplots(figsize=(10, max(3, len(syms) * 0.35)))
    y = np.arange(len(syms))
    ax.barh(y, vals, color=_COLOR_CVAR, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(syms, fontsize=9)
    ax.set_xlabel("Weight (%)")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_tail_contribution(weights: list[dict], dpi: int, top_n: int = 15) -> str:
    """图 4: 各资产尾部风险贡献条形图（正=推高组合尾部损失）"""
    nonzero = [w for w in weights if abs(w["tail_contribution"]) > 1e-9]
    nonzero.sort(key=lambda w: w["tail_contribution"], reverse=True)
    nonzero = nonzero[:top_n]
    nonzero = list(reversed(nonzero))
    syms = [w["symbol"] for w in nonzero]
    vals = [w["tail_contribution"] * 100 for w in nonzero]
    colors = [_COLOR_LOSS if v > 0 else _COLOR_POS for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(3, len(syms) * 0.35)))
    y = np.arange(len(syms))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(syms, fontsize=9)
    ax.set_xlabel("Tail Contribution (%)")
    ax.axvline(0, color="#333", linewidth=0.6)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


# === HTML 拼接 ==================================================================

def _fmt_pct(v: float, digits: int = 4) -> str:
    return f"{v * 100:.{digits}f}%"


def render_head(meta: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CVaR 尾部风险优化组合回测报告</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>CVaR 尾部风险优化组合 (portfolio-cvar-optim) 回测报告</h1>
  <div class="meta">
    生成时间: {meta['generated_at']}
    &nbsp;|&nbsp; 数据版本: <code>{meta['data_version']}</code>
    &nbsp;|&nbsp; 样本外区间: {meta['sample_start']} ~ {meta['sample_end']}
    &nbsp;|&nbsp; train/test: {meta['train_days']}/{meta['test_days']} 交易日
    &nbsp;|&nbsp; 资产数: {meta['asset_count']}（非零权重 {meta['nonzero_weights']}）
    &nbsp;|&nbsp; CVaR@{meta['beta']:.0%} · 目标年化 {meta['target_annual_return']:.0%} · 权重上限 {meta['weight_upper']:.0%}
  </div>
</header>
"""


def render_summary_cards(m: dict) -> str:
    """CVaR 组合核心绩效 8 卡片"""
    c = m["CVaR组合"]
    cards_data = [
        ("CVaR@95 (日损失)", _fmt_pct(c["CVaR95"], 3), "negative"),
        ("VaR@95 (日损失)", _fmt_pct(c["VaR95"], 3), "negative"),
        ("年化收益", _fmt_pct(c["年化收益"], 2), "positive" if c["年化收益"] > 0 else "negative"),
        ("年化波动", _fmt_pct(c["年化波动"], 2), ""),
        ("最大回撤", _fmt_pct(c["最大回撤"], 2), "negative"),
        ("Calmar", f"{c['Calmar']:.3f}", "positive" if c["Calmar"] > 0 else "negative"),
        ("夏普", f"{c['夏普']:.3f}", "positive" if c["夏普"] > 0 else "negative"),
        ("样本内 CVaR@95", _fmt_pct(m["样本内CVaR95"], 3), ""),
    ]
    cards = []
    for label, value, cls in cards_data:
        cls_str = f" {cls}" if cls else ""
        cards.append(f'<div class="card{cls_str}"><div class="label">{label}</div><div class="value">{value}</div></div>')
    return '<section><h2>CVaR 组合核心绩效（样本外）</h2><div class="cards-grid">\n' + "\n".join(cards) + '\n</div></section>'


def render_comparison(m: dict) -> str:
    """CVaR / 等权 / Benchmark 三方对比表"""
    cols = ["CVaR95", "VaR95", "年化收益", "年化波动", "最大回撤", "Calmar", "夏普"]
    labels = {"CVaR组合": "CVaR 优化组合", "等权组合": "等权组合 (1/N)", "Benchmark": "Benchmark 指数"}
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for key, label in labels.items():
        data = m.get(key)
        if data is None:
            continue
        cells = []
        for c in cols:
            v = data[c]
            if c in ("CVaR95", "VaR95", "年化收益", "年化波动", "最大回撤"):
                cells.append(f'<td class="num">{_fmt_pct(v, 3)}</td>')
            else:
                cells.append(f'<td class="num">{v:.3f}</td>')
        rows.append(f"<tr><td><strong>{label}</strong></td>{''.join(cells)}</tr>")

    improve = m.get("尾部改善(CVaR-等权)", 0.0)
    improve_txt = (f'<div class="note">尾部改善 = CVaR组合 CVaR95 − 等权 CVaR95 = '
                   f'<strong class="{"pos" if improve < 0 else "neg"}">{_fmt_pct(improve, 3)}</strong>'
                   f'（负值表示 CVaR 优化成功降低了尾部损失，符合预期）。</div>')
    return (
        '<section><h2>组合对比（CVaR 优化 vs 等权 vs Benchmark）</h2>'
        + improve_txt
        + f'<table><thead><tr><th>组合</th>{header}</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table></section>'
    )


def render_chart_section(title: str, img_b64: str, note: str | None = None) -> str:
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<section><h2>{title}</h2>{note_html}\n<img src="data:image/png;base64,{img_b64}" alt="{title}"/></section>'


def render_weight_table(weights: list[dict]) -> str:
    """权重明细表（非零权重）"""
    nonzero = [w for w in weights if w["weight"] > 1e-6]
    rows = []
    for rank, w in enumerate(nonzero, 1):
        ar = w["expected_annual_return"]
        tc = w["tail_contribution"]
        rows.append(
            f"<tr><td>{rank}</td><td><strong>{w['symbol']}</strong></td>"
            f'<td class="num">{_fmt_pct(w["weight"], 2)}</td>'
            f'<td class="num {"pos" if ar > 0 else "neg"}">{_fmt_pct(ar, 2)}</td>'
            f'<td class="num {"neg" if tc > 0 else "pos"}">{_fmt_pct(tc, 3)}</td></tr>'
        )
    return (
        '<section><h2>组合权重明细（非零权重，按权重降序）</h2>'
        '<div class="note">tail_contribution = 该资产权重 × 其在组合最差 5% 尾部情景的平均损失，'
        '正值（红）表示该资产是组合尾部风险的主要来源。</div>'
        '<table><thead><tr><th>排名</th><th>资产</th><th>权重</th><th>预期年化</th><th>尾部贡献</th></tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table></section>'
    )


def render_footer(text: str) -> str:
    return f'<footer><h3>评估口径（避免未来函数）</h3>\n<p>{text}</p></footer>\n</body>\n</html>\n'


def validate_payload(data: dict) -> None:
    """报告渲染前字段自检"""
    for k in ["meta", "metrics", "timeseries", "weights", "loss_histogram", "评估口径"]:
        if k not in data:
            raise ValueError(f"报告数据缺少顶层字段: {k}")
    if not data["timeseries"].get("cvar_curve"):
        raise ValueError("timeseries.cvar_curve 不能为空")
    if not data["weights"]:
        raise ValueError("weights 不能为空")
    if "CVaR组合" not in data["metrics"]:
        raise ValueError("metrics 缺少 CVaR组合")


def render_html(data: dict, dpi: int = 100) -> str:
    validate_payload(data)
    meta, m, ts = data["meta"], data["metrics"], data["timeseries"]

    img_equity = plot_equity_curve(ts, dpi)
    img_loss = plot_loss_distribution(data["loss_histogram"], dpi)
    img_weights = plot_weights(data["weights"], dpi)
    img_tail = plot_tail_contribution(data["weights"], dpi)

    parts = [
        render_head(meta),
        render_summary_cards(m),
        render_comparison(m),
        render_chart_section(
            "累计收益率与回撤（样本外）",
            img_equity,
            note="三条曲线为 test 段用固定权重计算的累计收益：CVaR 组合（蓝）、等权组合（橙）、Benchmark（灰虚线）。"
                 "下图为 CVaR 组合回撤序列。权重仅用 train 段求解，无未来函数。",
        ),
        render_chart_section(
            "样本外收益分布与尾部对比",
            img_loss,
            note="左尾（负收益侧）越薄、VaR 竖线越靠右（损失越小），说明尾部风险控制越好。"
                 "对比 CVaR 组合与等权组合的左尾厚度即可看出 CVaR 优化的效果。",
        ),
        render_chart_section(
            "组合权重分布",
            img_weights,
            note=f"CVaR-LP 求出的最优权重（单资产上限 {meta['weight_upper']:.0%}）。"
                 "尾部驱动的组合通常集中度较高，权重上限约束用于防止退化到单一资产。",
        ),
        render_chart_section(
            "各资产尾部风险贡献",
            img_tail,
            note="红色=推高组合尾部损失的资产（尾部风险来源），绿色=在尾部情景中对冲/贡献正收益的资产。",
        ),
        render_weight_table(data["weights"]),
        render_footer(data["评估口径"]),
    ]
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CVaR 尾部风险优化组合回测 HTML 报告")
    parser.add_argument("--offline", action="store_true", help="强制离线模式（用 fixtures，无需凭证）")
    parser.add_argument("--json-path", type=Path,
                        default=Path(__file__).parent.parent / "reports" / "backtest_result.json")
    parser.add_argument("--html-path", type=Path,
                        default=Path(__file__).parent.parent / "reports" / "report.html")
    parser.add_argument("--dpi", type=int, default=100, help="图表 DPI（默认 100）")
    parser.add_argument("--open", action="store_true", help="生成后自动用浏览器打开")
    args = parser.parse_args()

    offline_mode = args.offline or _is_offline(None)
    if offline_mode:
        os.environ["PANDA_DATA_OFFLINE"] = "1"
    print(f"[1/3] 运行回测 (offline={offline_mode}) ...")
    result = run_backtest_with_series(offline=args.offline or None)

    print(f"[2/3] 保存中间结果 → {args.json_path}")
    save_backtest_result(result, args.json_path)

    print(f"[3/3] 渲染 HTML 报告（dpi={args.dpi}）...")
    html = render_html(result, dpi=args.dpi)
    args.html_path.parent.mkdir(parents=True, exist_ok=True)
    args.html_path.write_text(html, encoding="utf-8")

    m = result["metrics"]["CVaR组合"]
    print("\n" + "=" * 60)
    print(f"[OK] 报告已生成")
    print(f"     HTML: {args.html_path}")
    print(f"     JSON: {args.json_path}")
    print(f"     样本外: {result['meta']['sample_start']} ~ {result['meta']['sample_end']}"
          f" ({result['meta']['test_days']} 交易日, {result['meta']['asset_count']} 资产)")
    print(f"     CVaR95={m['CVaR95']:.3%}, 年化={m['年化收益']:.2%}, MDD={m['最大回撤']:.2%}, Calmar={m['Calmar']:.3f}")
    print("=" * 60)

    if args.open:
        import webbrowser
        webbrowser.open(args.html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
