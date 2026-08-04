import pandas as pd

from stat_arb.prices import load_price_panel, load_price_series


def test_load_price_panel_from_symbol_csvs(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    pd.DataFrame({"date": dates, "close": [100, 101, 102, 103]}).to_csv(
        tmp_path / "AAA.csv", index=False
    )
    pd.DataFrame({"date": dates, "close": [50, 49, 51, 52]}).to_csv(
        tmp_path / "BBB.csv", index=False
    )

    prices = load_price_panel(["AAA", "BBB"], [tmp_path])

    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.shape == (4, 2)
    assert prices.iloc[0, 0] == 100


def test_load_price_series_from_shared_parquet_multiindex(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            ("Close", "NVDA"): [100, 101, 102],
            ("Close", "AMD"): [50, 51, 52],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "yf_deadbeef.parquet")

    series = load_price_series("AMD", [tmp_path])

    assert series.name == "AMD"
    assert series.iloc[-1] == 52
    assert list(series.index) == list(dates)
