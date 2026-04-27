from typing import Iterable

SYMBOL_PROFILES: dict[str, dict[str, str]] = {
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
    "SPY": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "IWM": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "DIA": {"asset_type": "etf", "sector": "broad_market", "cluster": "index_etf"},
    "XLK": {"asset_type": "etf", "sector": "technology", "cluster": "sector_etf"},
    "XLF": {"asset_type": "etf", "sector": "financials", "cluster": "sector_etf"},
    "XLV": {"asset_type": "etf", "sector": "healthcare", "cluster": "sector_etf"},
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
    "TLT": {"asset_type": "etf", "sector": "rates", "cluster": "treasury_duration"},
    "IEF": {"asset_type": "etf", "sector": "rates", "cluster": "treasury_duration"},
    "GLD": {"asset_type": "etf", "sector": "commodity", "cluster": "gold"},
    "USO": {"asset_type": "etf", "sector": "commodity", "cluster": "oil"},
    "VIXY": {"asset_type": "etf", "sector": "volatility", "cluster": "volatility"},
}


def symbol_profile(symbol: str) -> dict[str, str]:
    return SYMBOL_PROFILES.get(
        symbol.upper(),
        {"asset_type": "unknown", "sector": "unknown", "cluster": symbol.upper()},
    )


def economic_cluster(symbols: Iterable[str]) -> str:
    sectors = {symbol_profile(symbol)["sector"] for symbol in symbols}
    sectors.discard("unknown")
    if len(sectors) == 1:
        return next(iter(sectors))
    if not sectors:
        return "unknown"
    return "mixed"


def trade_archetype(symbols: Iterable[str]) -> str:
    profiles = [symbol_profile(symbol) for symbol in symbols]
    etf_count = sum(profile["asset_type"] == "etf" for profile in profiles)
    equity_count = sum(profile["asset_type"] == "equity" for profile in profiles)
    sectors = {
        profile["sector"] for profile in profiles if profile["sector"] != "unknown"
    }

    if etf_count == 1 and equity_count >= 1 and len(sectors) <= 2:
        return "constituents_vs_etf"
    if etf_count == 0 and len(sectors) == 1:
        return "intra_sector_relative_value"
    if etf_count >= 1 and "broad_market" in sectors:
        return "sector_vs_market"
    if len(sectors) > 1:
        return "cross_sector_relative_value"
    return "custom"
