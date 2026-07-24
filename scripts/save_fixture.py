"""联网拉取一份固定股票 + 指数数据样本，保存为 Parquet fixture 供离线验证使用

用法：
    export PANDA_DATA_USERNAME=...
    export PANDA_DATA_PASSWORD=...
    python save_fixture.py

生成的文件：
    fixtures/sample_stocks.parquet — 股票池日线（date/symbol/close）
    fixtures/sample_index.parquet  — 基准指数日线（用于 benchmark 对比）

后续运行 validate.py / report.py 时设置 PANDA_DATA_OFFLINE=1 即可使用 fixtures。
"""
from __future__ import annotations

from pathlib import Path

from data_loader import load_index_data, load_stock_data

FIXTURE_DIR = Path(__file__).parent / "fixtures"
STOCK_FIXTURE = FIXTURE_DIR / "sample_stocks.parquet"
INDEX_FIXTURE = FIXTURE_DIR / "sample_index.parquet"


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1/2: 拉取股票池数据")
    print("=" * 60)
    stock = load_stock_data()
    print(f"[OK] 股票数据 {len(stock)} 行，{stock['symbol'].nunique()} 只")

    print("\n" + "=" * 60)
    print("Step 2/2: 拉取基准指数数据")
    print("=" * 60)
    dates = stock["date"].astype(str)
    start_8, end_8 = dates.min(), dates.max()
    start = f"{start_8[:4]}-{start_8[4:6]}-{start_8[6:8]}"
    end = f"{end_8[:4]}-{end_8[4:6]}-{end_8[6:8]}"
    try:
        index = load_index_data(start_date=start, end_date=end)
        print(f"[OK] 指数数据 {len(index)} 行")
    except Exception as e:
        print(f"[WARN] 指数数据拉取失败: {e}")
        print("[WARN] 仅保存股票 fixture，离线模式 benchmark 对比将被跳过")
        index = None

    stock.to_parquet(STOCK_FIXTURE, index=False)
    print(f"\n[SAVE] {STOCK_FIXTURE} ({STOCK_FIXTURE.stat().st_size / 1024:.1f} KB)")
    if index is not None and not index.empty:
        index.to_parquet(INDEX_FIXTURE, index=False)
        print(f"[SAVE] {INDEX_FIXTURE} ({INDEX_FIXTURE.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("Fixture 生成完成。后续离线验证：")
    print("  export PANDA_DATA_OFFLINE=1")
    print("  python validate.py")
    print("  python report.py --offline")
    print("=" * 60)


if __name__ == "__main__":
    main()
