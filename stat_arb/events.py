from pathlib import Path

import pandas as pd


def load_event_calendar(csv_path: Path) -> pd.DataFrame:
    events = pd.read_csv(csv_path)
    required = {"symbol", "date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            f"Event calendar must contain columns {sorted(required)}, missing {sorted(missing)}"
        )
    events = events.copy()
    events["symbol"] = events["symbol"].astype(str).str.upper()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    if "event_type" not in events.columns:
        events["event_type"] = "generic"
    if "polarity" not in events.columns:
        events["polarity"] = 0
    if "impact" not in events.columns:
        events["impact"] = 1.0
    if "window_pre" not in events.columns:
        events["window_pre"] = 0
    if "window_post" not in events.columns:
        events["window_post"] = 0
    events["event_type"] = events["event_type"].astype(str)
    events["polarity"] = pd.to_numeric(events["polarity"], errors="coerce").fillna(0.0)
    events["impact"] = pd.to_numeric(events["impact"], errors="coerce").fillna(1.0)
    events["window_pre"] = (
        pd.to_numeric(events["window_pre"], errors="coerce").fillna(0).astype(int)
    )
    events["window_post"] = (
        pd.to_numeric(events["window_post"], errors="coerce").fillna(0).astype(int)
    )
    return events.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_event_blackout_mask(
    index: pd.DatetimeIndex,
    symbols: list[str],
    events: pd.DataFrame | None,
    days_before: int = 0,
    days_after: int = 0,
) -> pd.Series | None:
    if events is None or events.empty:
        return None

    normalized_index = pd.DatetimeIndex(index).normalize()
    mask = pd.Series(False, index=index, name="event_blackout")
    relevant = events[events["symbol"].isin([symbol.upper() for symbol in symbols])]
    if relevant.empty:
        return None

    for event_date in relevant["date"]:
        positions = normalized_index.get_indexer([event_date], method="nearest")
        pos = int(positions[0])
        if pos < 0:
            continue
        start = max(0, pos - days_before)
        end = min(len(mask) - 1, pos + days_after)
        mask.iloc[start : end + 1] = True

    return mask if mask.any() else None


def build_catalyst_signal(
    index: pd.DatetimeIndex,
    symbols: list[str],
    events: pd.DataFrame | None,
    days_after: int = 20,
) -> pd.Series | None:
    if events is None or events.empty:
        return None

    normalized_index = pd.DatetimeIndex(index).normalize()
    signal = pd.Series(0.0, index=index, name="catalyst_signal")
    relevant = events[events["symbol"].isin([symbol.upper() for symbol in symbols])]
    if relevant.empty:
        return None

    for _, row in relevant.iterrows():
        event_date = row["date"]
        positions = normalized_index.get_indexer([event_date], method="nearest")
        pos = int(positions[0])
        if pos < 0:
            continue
        start = pos
        end = min(len(signal) - 1, pos + max(days_after, int(row["window_post"])))
        signal.iloc[start : end + 1] += float(row["polarity"]) * float(row["impact"])

    signal = signal.clip(lower=-5.0, upper=5.0)
    return signal if (signal != 0).any() else None
