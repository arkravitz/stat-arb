from __future__ import annotations

"""
Prototype pipeline for discovering candidate statistical-arbitrage baskets.

What this does
--------------
1. Downloads historical asset data from a pluggable provider.
2. Builds rolling-window features per asset.
3. Trains a contrastive encoder to learn asset embeddings.
4. Uses nearest neighbors in embedding space to propose candidate buckets.
5. Scores each bucket with an explicit mean-reversion / tradability objective.
6. Fits hedge weights and emits trade ideas like:
      long 5.00 X, short 2.10 Y, short 1.40 Z
7. Saves results to CSV / JSON for later review by an LLM.

This is research code, not production trading infrastructure.
It is designed to be readable and hackable.
"""

import argparse
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------


def _safe_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices = prices.replace(0, np.nan).ffill().dropna(how="all")
    return np.log(prices)


class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, sleep_seconds: float = 12.5):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds

    def fetch_daily_adjusted(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        outputsize: str = "full",
    ) -> pd.Series:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "Time Series (Daily)" not in payload:
            raise ValueError(f"Unexpected response for {symbol}: {payload}")

        ts = payload["Time Series (Daily)"]
        series = pd.Series(
            {pd.Timestamp(k): float(v["5. adjusted close"]) for k, v in ts.items()},
            name=symbol,
        ).sort_index()

        if start is not None:
            series = series.loc[pd.Timestamp(start) :]
        if end is not None:
            series = series.loc[: pd.Timestamp(end)]

        time.sleep(self.sleep_seconds)
        return series


class CSVFolderClient:
    """
    Reads one CSV per symbol from a folder.
    Expected columns: date, close
    filename: <SYMBOL>.csv
    """

    def __init__(self, folder: str):
        self.folder = folder

    def fetch_daily_adjusted(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        outputsize: str = "full",
    ) -> pd.Series:
        path = f"{self.folder}/{symbol}.csv"
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        s = pd.Series(df["close"].values, index=df["date"], name=symbol).sort_index()
        if start is not None:
            s = s.loc[pd.Timestamp(start) :]
        if end is not None:
            s = s.loc[: pd.Timestamp(end)]
        return s


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_daily_adjusted(
        self,
        symbol: str,
        start: str,
        end: str,
        outputsize: str = "full",
    ) -> pd.Series:
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())

        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": start_ts,
            "to": end_ts,
            "token": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("s") != "ok":
            raise ValueError(f"Unexpected response for {symbol}: {payload}")

        ts = pd.to_datetime(payload["t"], unit="s")
        closes = payload["c"]
        return pd.Series(closes, index=ts, name=symbol).sort_index()


class YFinanceClient:
    """
    Batch-download prices from yfinance and cache the resulting panel to disk.
    This avoids repeated vendor calls after the first successful download.
    """

    def __init__(self, cache_dir: str = "price_cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbols: list[str], start: str, end: str) -> Path:
        key = {
            "provider": "yfinance",
            "symbols": sorted(symbols),
            "start": start,
            "end": end,
            "auto_adjust": True,
            "interval": "1d",
        }
        digest = hashlib.md5(repr(key).encode("utf-8")).hexdigest()
        return self.cache_dir / f"yf_{digest}.parquet"

    def fetch_price_panel(
        self,
        symbols: list[str],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbols, start, end)

        if self.use_cache and cache_path.exists() and not force_refresh:
            print(f"Loading cached prices from {cache_path}")
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.sort_index()

        print("Downloading prices from yfinance...")
        df = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            interval="1d",
            threads=False,
            group_by="column",
            multi_level_index=True,
        )

        if df is None or df.empty:
            raise ValueError("No data returned from yfinance")

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))

            if "Close" in lvl0:
                close_df = df["Close"].copy()
            elif "Adj Close" in lvl0:
                close_df = df["Adj Close"].copy()
            elif "Close" in lvl1:
                close_df = df.xs("Close", axis=1, level=1).copy()
            elif "Adj Close" in lvl1:
                close_df = df.xs("Adj Close", axis=1, level=1).copy()
            else:
                raise ValueError(
                    f"Could not find Close/Adj Close in columns: {df.columns}"
                )
        else:
            if "Close" in df.columns:
                close_df = df[["Close"]].copy()
            elif "Adj Close" in df.columns:
                close_df = df[["Adj Close"]].copy()
            else:
                raise ValueError(
                    f"Could not find Close/Adj Close in columns: {df.columns}"
                )

            if len(symbols) == 1:
                close_df.columns = [symbols[0]]

        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame()

        if len(close_df.columns) == 1 and len(symbols) == 1:
            close_df.columns = [symbols[0]]

        present = [s for s in symbols if s in close_df.columns]
        if not present:
            raise ValueError(
                f"No requested symbols found in returned data. Got columns: {list(close_df.columns)}"
            )

        close_df = close_df[present].copy()
        close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
        close_df = close_df.sort_index()
        close_df = close_df.dropna(axis=1, how="all")

        if close_df.empty:
            raise ValueError("No usable Close prices returned from yfinance")

        if self.use_cache:
            close_df.to_parquet(cache_path)
            print(f"Saved price cache to {cache_path}")

        return close_df


UNIVERSE_PRESETS: Dict[str, List[str]] = {
    "semis_fast": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "SMH", "SOXX", "QQQ"],
    "energy_fast": ["XLE", "CVX", "XOM", "SLB", "HAL", "COP", "OXY"],
    "travel_fuel_fast": ["UAL", "DAL", "AAL", "LUV", "BA", "JETS", "XLE"],
    "etf_style_fast": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"],
}

SYMBOL_PROFILES: Dict[str, Dict[str, str]] = {
    "NVDA": {"asset_type": "equity", "sector": "semis", "cluster": "semis_design"},
    "AMD": {"asset_type": "equity", "sector": "semis", "cluster": "semis_design"},
    "AVGO": {"asset_type": "equity", "sector": "semis", "cluster": "semis_infra"},
    "QCOM": {"asset_type": "equity", "sector": "semis", "cluster": "semis_design"},
    "INTC": {
        "asset_type": "equity",
        "sector": "semis",
        "cluster": "semis_manufacturing",
    },
    "SMH": {"asset_type": "etf", "sector": "semis", "cluster": "semis_etf"},
    "SOXX": {"asset_type": "etf", "sector": "semis", "cluster": "semis_etf"},
    "QQQ": {"asset_type": "etf", "sector": "growth", "cluster": "broad_growth"},
    "XLE": {"asset_type": "etf", "sector": "energy", "cluster": "energy_etf"},
    "CVX": {"asset_type": "equity", "sector": "energy", "cluster": "energy_integrated"},
    "XOM": {"asset_type": "equity", "sector": "energy", "cluster": "energy_integrated"},
    "SLB": {"asset_type": "equity", "sector": "energy", "cluster": "energy_services"},
    "HAL": {"asset_type": "equity", "sector": "energy", "cluster": "energy_services"},
    "COP": {"asset_type": "equity", "sector": "energy", "cluster": "energy_upstream"},
    "OXY": {"asset_type": "equity", "sector": "energy", "cluster": "energy_upstream"},
    "UAL": {"asset_type": "equity", "sector": "travel", "cluster": "airlines"},
    "DAL": {"asset_type": "equity", "sector": "travel", "cluster": "airlines"},
    "AAL": {"asset_type": "equity", "sector": "travel", "cluster": "airlines"},
    "LUV": {"asset_type": "equity", "sector": "travel", "cluster": "airlines"},
    "BA": {"asset_type": "equity", "sector": "travel", "cluster": "aerospace"},
    "JETS": {"asset_type": "etf", "sector": "travel", "cluster": "travel_etf"},
    "SPY": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "IWM": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "DIA": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "XLK": {"asset_type": "etf", "sector": "technology", "cluster": "sector_etf"},
    "XLF": {"asset_type": "etf", "sector": "financials", "cluster": "sector_etf"},
    "XLV": {"asset_type": "etf", "sector": "healthcare", "cluster": "sector_etf"},
}


def symbol_profile(symbol: str) -> Dict[str, str]:
    return SYMBOL_PROFILES.get(
        symbol.upper(),
        {"asset_type": "unknown", "sector": "unknown", "cluster": symbol.upper()},
    )


def _economic_similarity(left: str, right: str) -> int:
    left_profile = symbol_profile(left)
    right_profile = symbol_profile(right)
    score = 0
    if (
        left_profile["sector"] == right_profile["sector"]
        and left_profile["sector"] != "unknown"
    ):
        score += 2
    if left_profile["cluster"] == right_profile["cluster"]:
        score += 1
    if left_profile["asset_type"] == right_profile["asset_type"]:
        score += 1
    return score


def _bucket_cluster(bucket: Tuple[str, ...]) -> str:
    sectors = {symbol_profile(symbol)["sector"] for symbol in bucket}
    sectors.discard("unknown")
    if len(sectors) == 1:
        return next(iter(sectors))
    return "mixed"


def _bucket_passes_constraints(bucket: Tuple[str, ...]) -> bool:
    profiles = [symbol_profile(symbol) for symbol in bucket]
    etf_count = sum(profile["asset_type"] == "etf" for profile in profiles)
    equity_count = sum(profile["asset_type"] == "equity" for profile in profiles)
    sectors = {profile["sector"] for profile in profiles}

    if etf_count > 1:
        return False
    if len(bucket) >= 3 and equity_count < 2:
        return False
    if len(sectors) > 2:
        return False
    if (
        len(sectors) == 2
        and "unknown" not in sectors
        and "broad_market" not in sectors
        and "growth" not in sectors
    ):
        return False
    return True


def load_price_panel(
    symbols: Sequence[str],
    provider: str,
    api_key: str | None,
    csv_folder: str | None,
    start: str,
    end: str,
    cache_dir: str = "price_cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    if provider == "yfinance":
        client = YFinanceClient(cache_dir=cache_dir, use_cache=True)
        prices = client.fetch_price_panel(
            symbols=list(symbols),
            start=start,
            end=end,
            force_refresh=force_refresh,
        )
        prices = prices.loc[pd.Timestamp(start) : pd.Timestamp(end)].ffill()
        prices = prices.dropna(axis=1, thresh=max(30, int(0.7 * len(prices))))
        prices = prices.dropna()
        if prices.empty:
            raise RuntimeError("No price series loaded after filtering")
        return prices

    if provider == "alpha_vantage":
        if not api_key:
            raise ValueError("--api-key is required for alpha_vantage")
        client = AlphaVantageClient(api_key)
    elif provider == "finnhub":
        if not api_key:
            raise ValueError("--api-key is required for finnhub")
        client = FinnhubClient(api_key)
    elif provider == "csv":
        if not csv_folder:
            raise ValueError("--csv-folder is required for csv provider")
        client = CSVFolderClient(csv_folder)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    series_list = []
    for symbol in tqdm(symbols, desc="Downloading / loading prices"):
        try:
            s = client.fetch_daily_adjusted(symbol, start=start, end=end)
            if isinstance(s, pd.DataFrame):
                if s.shape[1] != 1:
                    raise ValueError(
                        f"Expected 1 price column for {symbol}, got shape {s.shape}"
                    )
                s = s.iloc[:, 0]
            s.name = symbol
            series_list.append(s)
        except Exception as exc:
            print(f"Skipping {symbol}: {exc}")

    if not series_list:
        raise RuntimeError("No price series loaded")

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.loc[pd.Timestamp(start) : pd.Timestamp(end)].ffill()
    prices = prices.dropna(axis=1, thresh=max(30, int(0.7 * len(prices))))
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError("No price series loaded after filtering")

    return prices


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------


def make_asset_feature_tensor(
    prices: pd.DataFrame,
    lookback: int = 60,
    vol_lookback: int = 20,
) -> Tuple[np.ndarray, List[pd.Timestamp], List[str]]:
    """Return tensor shaped [num_samples, num_assets, num_features]."""
    logp = _safe_log_prices(prices)
    rets = logp.diff()
    vol = rets.rolling(vol_lookback).std()
    mom_5 = logp.diff(5)
    mom_20 = logp.diff(20)

    feature_blocks = []
    timestamps = []
    symbols = list(prices.columns)

    min_start = max(lookback, vol_lookback, 20) + 1

    for end_ix in range(min_start, len(prices)):
        sl = slice(end_ix - lookback, end_ix)
        window_rets = rets.iloc[sl].copy()

        last_row = pd.concat(
            [
                window_rets.mean().rename("mean_ret"),
                window_rets.std().rename("std_ret"),
                vol.iloc[end_ix].rename("vol"),
                mom_5.iloc[end_ix].rename("mom_5"),
                mom_20.iloc[end_ix].rename("mom_20"),
            ],
            axis=1,
        )

        window_rets = window_rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        last_row = last_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        window_matrix = window_rets.T.to_numpy(dtype=np.float32)
        summary_matrix = last_row.to_numpy(dtype=np.float32)
        feats = np.concatenate([window_matrix, summary_matrix], axis=1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        feature_blocks.append(feats)
        timestamps.append(prices.index[end_ix])

    tensor = np.stack(feature_blocks, axis=0).astype(np.float32)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor, timestamps, symbols


# -----------------------------------------------------------------------------
# Contrastive encoder
# -----------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class ArcMarginProduct(nn.Module):
    """
    ArcFace-style margin head.
    Used only as a research toy for identity structure across rolling windows.
    """

    def __init__(
        self, in_features: int, out_features: int, s: float = 10.0, m: float = 0.1
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits


@dataclass
class EmbeddingTrainingConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 512
    emb_dim: int = 32
    hidden_dim: int = 128
    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


