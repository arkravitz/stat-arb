import pytest

from stat_arb.baskets import (
    BasketParseError,
    parse_basket_string,
    parse_weights_json,
    slugify_trade,
)


def test_parse_basket_string_combines_duplicate_symbols() -> None:
    spec = parse_basket_string(
        "long 1 units of avgo, short 2 units of SMH, long 0.5 units of AVGO"
    )
    assert spec.weights == {"AVGO": 1.5, "SMH": -2.0}


def test_parse_weights_json_normalizes_symbols() -> None:
    spec = parse_weights_json("{'nvda': 1, 'smh': -3}")
    assert spec.weights == {"NVDA": 1.0, "SMH": -3.0}


def test_parse_basket_string_rejects_invalid_segment() -> None:
    with pytest.raises(BasketParseError):
        parse_basket_string("buy 1 units of NVDA")


def test_slugify_trade_truncates_and_sanitizes() -> None:
    slug = slugify_trade("Long 1 units of NVDA / Short 5 units of SMH", max_len=18)
    assert slug == "long_1_units_of_nv"
