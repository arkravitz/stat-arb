FACTOR_PRESETS: dict[str, tuple[str, ...]] = {
    "core_tech": ("QQQ", "XLK", "SMH", "SOXX"),
    "broad_macro": ("SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "USO", "VIXY"),
    "sector_growth": ("SPY", "QQQ", "XLK", "SMH", "SOXX", "XLE", "XLF", "XLV"),
}


def resolve_factor_symbols(
    custom_symbols: list[str] | None,
    preset: str | None,
) -> list[str]:
    symbols: list[str] = []
    if preset:
        symbols.extend(FACTOR_PRESETS[preset])
    if custom_symbols:
        symbols.extend(custom_symbols)
    # preserve order, remove duplicates
    seen: set[str] = set()
    resolved: list[str] = []
    for symbol in symbols:
        symbol = symbol.upper()
        if symbol not in seen:
            seen.add(symbol)
            resolved.append(symbol)
    return resolved
