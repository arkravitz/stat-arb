import ast
import re
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BasketSpec:
    weights: Dict[str, float]


class BasketParseError(ValueError):
    pass


_BASKET_PATTERN = re.compile(
    r"\s*(long|short)\s+([0-9]*\.?[0-9]+)\s+units?\s+of\s+([A-Za-z0-9_\-\.]+)\s*",
    re.IGNORECASE,
)


def parse_basket_string(expr: str) -> BasketSpec:
    weights: Dict[str, float] = {}

    for raw_part in expr.split(","):
        part = raw_part.strip()
        if not part:
            continue
        match = _BASKET_PATTERN.fullmatch(part)
        if not match:
            raise BasketParseError(
                f"Could not parse basket segment: {part!r}. Expected format like 'long 2 units of NVDA'."
            )
        side, qty_str, symbol = match.groups()
        qty = float(qty_str)
        signed = qty if side.lower() == "long" else -qty
        symbol = symbol.upper()
        weights[symbol] = weights.get(symbol, 0.0) + signed

    if not weights:
        raise BasketParseError("No valid positions found in basket string.")
    return BasketSpec(weights=weights)


def parse_weights_json(text: str) -> BasketSpec:
    data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise BasketParseError("Weights input must evaluate to a dict like {'NVDA': 1, 'SMH': -2}.")
    weights = {str(k).upper(): float(v) for k, v in data.items()}
    if not weights:
        raise BasketParseError("No weights found.")
    return BasketSpec(weights=weights)


def slugify_trade(text: str, max_len: int = 100) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "trade"


def render_weight_lines(weights: Dict[str, float]) -> list[str]:
    lines = []
    for symbol, weight in weights.items():
        side = "LONG" if weight > 0 else "SHORT"
        lines.append(f"  {side:5s} {abs(weight):8.3f} units of {symbol}")
    return lines
