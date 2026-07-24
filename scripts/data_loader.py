"""CVaR 尾部风险优化 —— 数据加载层（data_loader）

从 panda_data SDK 拉取 A 股股票池 + 基准指数日线，统一为长表(date/symbol/close)。
含本地 parquet 缓存（先查本地→未命中联网→写缓存）、日期解析、字段标准化。
本层只负责"取数与清洗"，因子计算（CVaR-LP / GPD 补尾）见 factor.py。

数据源：
    get_stock_daily —— 股票池日线（可配置资产）
    get_index_daily —— 基准指数日线（benchmark，超额收益/信息比）

PandaAI data 实现说明详见 references/data_guide.md。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import panda_data

# === 全局常量 ===================================================================
BATCH_SIZE = 1                 # 单次 API 调用品种数（对齐 alpha-f1，避免服务器超限）

# === 本地缓存路径 ===============================================================
_CACHE_DIR = Path(__file__).parent / "fixtures"
_STOCK_CACHE = _CACHE_DIR / "cache_stocks.parquet"
_INDEX_CACHE = _CACHE_DIR / "cache_index.parquet"

# 默认股票池：沪深300 中流动性较好的 20 只（跨行业分散，降低尾部相关）
DEFAULT_STOCK_POOL = [
    "600519.SH",  # 贵州茅台
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "000858.SZ",  # 五粮液
    "601166.SH",  # 兴业银行
    "600276.SH",  # 恒瑞医药
    "000333.SZ",  # 美的集团
    "600900.SH",  # 长江电力
    "601888.SH",  # 中国中免
    "002594.SZ",  # 比亚迪
    "600030.SH",  # 中信证券
    "000651.SZ",  # 格力电器
    "601012.SH",  # 隆基绿能
    "600887.SH",  # 伊利股份
    "600309.SH",  # 万华化学
    "000001.SZ",  # 平安银行
    "601899.SH",  # 紫金矿业
    "600028.SH",  # 中国石化
    "601088.SH",  # 中国神华
    "600585.SH",  # 海螺水泥
]
DEFAULT_BENCHMARK = "000300.SH"  # 沪深300 指数


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"请先设置环境变量 {name}")
    return value


def _init_token():
    """初始化 panda_data 凭证（幂等，重复调用无副作用）"""
    panda_data.init_token(username=_get_env("PANDA_DATA_USERNAME"),
                          password=_get_env("PANDA_DATA_PASSWORD"))


# === 本地缓存层 =================================================================
# 策略：每次获取数据前先查本地 parquet 缓存，命中则直接返回；
# 未命中才联网获取，获取后写入缓存供后续复用。

def _read_cache(path: Path) -> pd.DataFrame | None:
    """读取 parquet 缓存，文件不存在返回 None"""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # pyarrow 兼容性问题时尝试 fastparquet
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            return None


def _load_with_cache(
    cache_path: Path,
    start_date_8: str,
    end_date_8: str,
    symbols: list[str],
    fetch_fn,
) -> pd.DataFrame:
    """通用缓存层：先查本地 parquet → 未命中则联网 → 写入缓存

    缓存命中条件：缓存日期区间覆盖请求区间 且 缓存品种包含请求品种。
    """
    cached = _read_cache(cache_path)
    if cached is not None and not cached.empty:
        cached_dates = cached["date"].astype(str)
        cache_start = cached_dates.min()
        cache_end = cached_dates.max()
        cache_symbols = set(cached["symbol"].unique())
        req_symbols = set(symbols)
        if cache_start <= start_date_8 and cache_end >= end_date_8 and req_symbols.issubset(cache_symbols):
            mask = (
                (cached["date"].astype(str) >= start_date_8)
                & (cached["date"].astype(str) <= end_date_8)
                & cached["symbol"].isin(req_symbols)
            )
            result = cached[mask]
            if not result.empty:
                print(f"[CACHE] 从本地缓存加载数据 ({len(result)} 行) → {cache_path.name}")
                return result.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 缓存未命中 → 联网获取
    print(f"[CACHE] 本地缓存未命中，联网获取 → {cache_path.name}")
    result = fetch_fn()
    # 写入缓存（与已有缓存合并，避免覆盖历史数据）
    if result is not None and not result.empty:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, result], ignore_index=True).drop_duplicates(
                subset=["date", "symbol"], keep="last",
            )
        else:
            merged = result
        merged.to_parquet(cache_path, index=False)
        print(f"[CACHE] 已写入本地缓存 ({len(merged)} 行) → {cache_path}")
    return result


def _chunked(items: list, size: int = BATCH_SIZE):
    """把列表切成固定大小的批次（生成器），用于 API 分批调用避免超限"""
    for i in range(0, len(items), size):
        yield items[i:i + size]


_TRADE_DATE_CACHE: str | None = None


def _latest_trade_date(exchange: str = "SH") -> str:
    """最近已收盘的交易日，返回 'YYYY-MM-DD'。

    用官方交易日历（get_last_trade_date / get_prev_trade_date）精确处理周末与法定假日，
    避免默认结束日期落在非交易日（如周日 7/19）导致取数为空。
    16:00 收盘后取最新交易日；收盘前取前一个交易日（当天数据未完成）。
    结果模块级缓存，单次运行复用，避免重复联网。
    """
    global _TRADE_DATE_CACHE
    if _TRADE_DATE_CACHE is not None:
        return _TRADE_DATE_CACHE
    _init_token()
    now = datetime.now()
    after_close = now >= now.replace(hour=16, minute=0, second=0, microsecond=0)
    if after_close:
        # 收盘后：最新交易日（今天若为交易日则含今天）
        latest = panda_data.get_last_trade_date(exchange=exchange)
    else:
        # 收盘前：今天数据未完成，取前一个交易日
        today8 = now.strftime("%Y%m%d")
        latest = panda_data.get_prev_trade_date(date=today8, exchange=exchange, n=1)
    # 接口直接返回 'YYYYMMDD' 字符串（实测确认，非 DataFrame；文档示例画成表具误导性）
    # 接口承诺 None 表示无数据
    if not latest:
        raise RuntimeError(f"交易日历接口返回空（exchange={exchange}），无法确定最近交易日")
    latest = str(latest)
    _TRADE_DATE_CACHE = f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"
    return _TRADE_DATE_CACHE


def get_default_end_date() -> str:
    """默认结束日期：最近已收盘交易日（交易日历获取，精确处理周末/假日）"""
    return _latest_trade_date()


def _resolve_date_range(
    start_date: str | None,
    end_date: str | None,
    lookback_days: int = 365,
) -> tuple[str, str]:
    """解析日期范围，支持环境变量与默认回溯窗口。返回 (start_date, end_date) YYYY-MM-DD"""
    end_date = end_date or os.getenv("PANDA_DATA_END_DATE", get_default_end_date())
    if start_date is None:
        start_date = os.getenv("PANDA_DATA_START_DATE")
        if not start_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")
    return start_date, end_date


def _normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    """价格数据字段标准化：统一出 date / symbol / close 三列

    防御接口字段名差异：code/ts_code→symbol，trade_date→date，收盘价别名映射。
    """
    df = df.copy()
    # 字段别名映射
    rename_map = {}
    for src in ("code", "ts_code", "sec_code"):
        if src in df.columns and "symbol" not in df.columns:
            rename_map[src] = "symbol"
    for src in ("trade_date", "trade_dt", "datetime"):
        if src in df.columns and "date" not in df.columns:
            rename_map[src] = "date"
    if rename_map:
        print(f"[INFO] 接口字段自动映射: {rename_map}")
        df = df.rename(columns=rename_map)

    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"价格数据缺少必要字段: {sorted(missing)}。接口实际字段: {sorted(df.columns.tolist())}"
        )

    df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df[["date", "symbol", "close"]]


# === 数据加载 ===================================================================
def load_stock_data(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从 panda_data.get_stock_daily 获取股票池日线，标准化为 date/symbol/close

    默认回溯 1 年（CVaR 尾部估计需要足够样本）。
    """
    _init_token()

    symbols = symbols or DEFAULT_STOCK_POOL
    start_date, end_date = _resolve_date_range(start_date, end_date)
    start_date_8 = start_date.replace("-", "")
    end_date_8 = end_date.replace("-", "")
    print(f"获取股票数据范围: {start_date} ~ {end_date}, {len(symbols)} 只")

    def _fetch() -> pd.DataFrame:
        debug_mode = os.getenv("PANDA_DATA_DEBUG", "0") == "1"
        debug_printed = False
        total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
        parts = []
        for batch_idx, batch in enumerate(_chunked(symbols), start=1):
            print(f"  批次 {batch_idx}/{total_batches}: {batch}", flush=True)
            df = panda_data.get_stock_daily(
                start_date=start_date_8,
                end_date=end_date_8,
                symbol=batch,
            )
            if df is not None and not df.empty:
                if debug_mode and not debug_printed:
                    print(f"[DEBUG] get_stock_daily 返回字段: {df.columns.tolist()}")
                    print(f"[DEBUG] 前 3 行:\n{df.head(3)}")
                    debug_printed = True
                parts.append(df)
        if not parts:
            raise ValueError(f"未获取到股票数据 (日期: {start_date} ~ {end_date})")
        return _normalize_price(pd.concat(parts, ignore_index=True))

    stock_df = _load_with_cache(_STOCK_CACHE, start_date_8, end_date_8, symbols, _fetch)
    print(f"[OK] 股票数据共 {len(stock_df)} 行")
    return stock_df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_index_data(
    benchmark: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从 panda_data.get_index_daily 获取基准指数日线，标准化为 date/symbol/close"""
    _init_token()

    benchmark = benchmark or DEFAULT_BENCHMARK
    start_date, end_date = _resolve_date_range(start_date, end_date)
    start_date_8 = start_date.replace("-", "")
    end_date_8 = end_date.replace("-", "")
    print(f"获取指数数据: {benchmark}, {start_date} ~ {end_date}")

    def _fetch() -> pd.DataFrame:
        df = panda_data.get_index_daily(
            start_date=start_date_8,
            end_date=end_date_8,
            symbol=benchmark,
        )
        if df is None or df.empty:
            raise ValueError(f"未获取到指数数据 ({benchmark}, {start_date} ~ {end_date})")
        return _normalize_price(df)

    index_df = _load_with_cache(_INDEX_CACHE, start_date_8, end_date_8, [benchmark], _fetch)
    print(f"[OK] 指数数据共 {len(index_df)} 行")
    return index_df.sort_values(["symbol", "date"]).reset_index(drop=True)
