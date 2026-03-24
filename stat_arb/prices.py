import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_CACHE_DIRS = [
    Path("price_cache"),
    Path("cache"),
    Path("data/cache"),
    Path("data/price_cache"),
]


def resolve_cache_dirs(cache_dir_arg: str | None) -> list[Path]:
    if cache_dir_arg:
        return [Path(cache_dir_arg)]
    env_dir = os.getenv("STAT_ARB_CACHE_DIR")
    if env_dir:
        return [Path(env_dir)]
    return DEFAULT_CACHE_DIRS


def find_cache_file(symbol: str, cache_dirs: Sequence[Path]) -> Path:
    candidates: list[Path] = []
    for root in cache_dirs:
        if not root.exists():
            continue
        candidates.extend(
            [
                root / f"{symbol}.csv",
                root / f"{symbol}.parquet",
                root / f"{symbol.lower()}.csv",
                root / f"{symbol.lower()}.parquet",
                root / "yfinance" / f"{symbol}.csv",
                root / "yfinance" / f"{symbol}.parquet",
                root / "yfinance" / f"{symbol.lower()}.csv",
                root / "yfinance" / f"{symbol.lower()}.parquet",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No cache file found for {symbol} in {[str(root) for root in cache_dirs]}")


def find_single_parquet_cache(cache_dirs: Sequence[Path]) -> Path | None:
    parquet_files: list[Path] = []
    for root in cache_dirs:
        if not root.exists():
            continue
        parquet_files.extend(sorted(root.rglob("*.parquet")))

    if not parquet_files:
        return None

    preferred = [path for path in parquet_files if path.name.startswith("yf_")]
    if len(preferred) == 1:
        return preferred[0]
    if len(parquet_files) == 1:
        return parquet_files[0]

    hashed = [path for path in parquet_files if re.match(r"^yf_[0-9a-f]+\.parquet$", path.name)]
    if len(hashed) == 1:
        return hashed[0]

    return None


def _normalize_price_frame(df: pd.DataFrame, symbol: str) -> pd.Series:
    cols = {str(col).lower(): col for col in df.columns}

    date_col = None
    for candidate in ["date", "datetime", "timestamp", "index"]:
        if candidate in cols:
            date_col = cols[candidate]
            break

    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df.assign(__date__=dates).dropna(subset=["__date__"]).set_index("__date__")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    price_col = None
    for candidate in ["adj close", "adj_close", "adjusted_close", "close", "adjclose"]:
        if candidate in cols:
            price_col = cols[candidate]
            break

    if price_col is None:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) == 1:
            price_col = numeric_cols[0]
        elif "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(f"Could not infer price column for {symbol}. Columns: {list(df.columns)}")

    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices.name = symbol
    return prices


