from pathlib import Path

import pandas as pd

from stat_arb.events import (
    build_catalyst_signal,
    build_event_blackout_mask,
    load_event_calendar,
)


def test_load_event_calendar_and_build_blackout_mask(tmp_path: Path) -> None:
    calendar_path = tmp_path / "events.csv"
    calendar_path.write_text("symbol,date\nAAA,2024-01-05\n", encoding="utf-8")

    events = load_event_calendar(calendar_path)
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    mask = build_event_blackout_mask(
        index, ["AAA"], events, days_before=1, days_after=1
    )

    assert mask is not None
    assert mask.sum() == 3
    assert bool(mask.loc["2024-01-04"])
    assert bool(mask.loc["2024-01-05"])
    assert bool(mask.loc["2024-01-06"])


def test_build_catalyst_signal_uses_polarity_and_impact(tmp_path: Path) -> None:
    calendar_path = tmp_path / "events.csv"
    calendar_path.write_text(
        "symbol,date,event_type,polarity,impact\nAAA,2024-01-05,earnings,1,2\n",
        encoding="utf-8",
    )
    events = load_event_calendar(calendar_path)
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    signal = build_catalyst_signal(index, ["AAA"], events, days_after=1)

    assert signal is not None
    assert float(signal.loc["2024-01-05"]) == 2.0
    assert float(signal.loc["2024-01-06"]) == 2.0
