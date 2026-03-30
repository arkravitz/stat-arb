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


def extract_symbol_series_from_wide_cache(df: pd.DataFrame, symbol: str) -> pd.Series:
    symbol = symbol.upper()

    if isinstance(df.columns, pd.MultiIndex):
        for symbol_level in range(df.columns.nlevels):
            values = [str(value).upper() for value in df.columns.get_level_values(symbol_level)]
            if symbol not in values:
                continue

            field_levels = [idx for idx in range(df.columns.nlevels) if idx != symbol_level]
            close_candidates = ["ADJ CLOSE", "ADJ_CLOSE", "ADJCLOSE", "CLOSE"]
            for field_level in field_levels:
                field_values = [str(value).upper() for value in df.columns.get_level_values(field_level)]
                for close_name in close_candidates:
                    if close_name not in field_values:
                        continue
                    symbol_values = pd.Index(
                        [str(value).upper() for value in df.columns.get_level_values(symbol_level)]
                    )
                    field_values_index = pd.Index(
                        [str(value).upper() for value in df.columns.get_level_values(field_level)]
                    )
                    mask = (symbol_values == symbol) & (field_values_index == close_name)
                    subset = df.loc[:, mask]
                    if subset.shape[1] >= 1:
                        series = pd.to_numeric(subset.iloc[:, 0], errors="coerce").dropna()
                        series = series[~series.index.duplicated(keep="last")].sort_index()
                        series.name = symbol
                        return series

            subset = df.xs(symbol, axis=1, level=symbol_level, drop_level=True)
            if isinstance(subset, pd.Series):
                subset = subset.to_frame(name="Close")
            else:
                subset = subset.copy()
            subset.columns = [str(col) for col in subset.columns]
            return _normalize_price_frame(subset.reset_index(), symbol)

        raise ValueError(f"Symbol {symbol} not found in MultiIndex columns.")

    upper_cols = {str(col).upper(): col for col in df.columns}
    for candidate in [
        symbol,
        f"ADJ CLOSE_{symbol}",
        f"ADJ_CLOSE_{symbol}",
        f"ADJCLOSE_{symbol}",
        f"CLOSE_{symbol}",
        f"{symbol}_ADJ CLOSE",
        f"{symbol}_ADJ_CLOSE",
        f"{symbol}_ADJCLOSE",
        f"{symbol}_CLOSE",
    ]:
        if candidate in upper_cols:
            subset = pd.DataFrame({symbol: df[upper_cols[candidate]]})
            subset.index = df.index
            return _normalize_price_frame(subset.reset_index(), symbol)

    cols_upper = {str(col).upper(): col for col in df.columns}
    sym_col = next((cols_upper[name] for name in ["SYMBOL", "TICKER", "ASSET"] if name in cols_upper), None)
    if sym_col is not None:
        subset = df[df[sym_col].astype(str).str.upper() == symbol].copy()
        if not subset.empty:
            return _normalize_price_frame(subset, symbol)

    raise ValueError(
        f"Could not extract {symbol} from shared cache. Columns look like: {list(df.columns)[:12]}"
    )


def _sample_parquet_paths(cache_dirs: Iterable[Path], limit: int = 10) -> list[str]:
    sample: list[str] = []
    for root in cache_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*.parquet"):
            sample.append(str(path))
            if len(sample) >= limit:
                return sample
    return sample


def load_price_series(symbol: str, cache_dirs: Sequence[Path]) -> pd.Series:
    try:
        path = find_cache_file(symbol, cache_dirs)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported cache file type: {path}")
        return _normalize_price_frame(df, symbol)
    except FileNotFoundError as exc:
        shared_cache = find_single_parquet_cache(cache_dirs)
        if shared_cache is None:
            sample = _sample_parquet_paths(cache_dirs)
            raise FileNotFoundError(
                f"No per-symbol cache file found for {symbol}, and no unique shared parquet cache found. "
                f"Saw parquet files: {sample}"
            ) from exc
        df = pd.read_parquet(shared_cache)
        return extract_symbol_series_from_wide_cache(df, symbol)


def load_price_panel(symbols: Sequence[str], cache_dirs: Sequence[Path]) -> pd.DataFrame:
    series = [load_price_series(symbol, cache_dirs) for symbol in symbols]
    prices = pd.concat(series, axis=1).sort_index()
    return prices.dropna(how="all")
