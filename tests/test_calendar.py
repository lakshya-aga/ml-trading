"""NSE calendar behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from afml_india.data.calendar import NSE, NSECalendar


def test_weekends_are_not_trading_days():
    assert not NSE.is_trading_day("2024-01-06")  # Saturday
    assert not NSE.is_trading_day("2024-01-07")  # Sunday
    assert NSE.is_trading_day("2024-01-05")      # Friday


def test_republic_day_is_a_holiday():
    assert not NSE.is_trading_day("2024-01-26")
    # 26 Jan 2024 was a Friday, so the next session is the following Monday.
    assert NSE.next_session("2024-01-25") == pd.Timestamp("2024-01-29")


def test_sessions_exclude_holidays():
    sessions = NSE.sessions("2024-01-01", "2024-01-31")
    assert pd.Timestamp("2024-01-26") not in sessions
    assert sessions.is_monotonic_increasing
    assert (sessions.weekday < 5).all()


def test_shift_round_trips():
    day = pd.Timestamp("2024-03-01")
    assert NSE.shift(NSE.shift(day, 5), -5) == day


def test_add_sessions_preserves_time_of_day_and_skips_holidays():
    stamps = pd.DatetimeIndex(["2024-01-24 10:30", "2024-01-25 14:45"])
    shifted = NSE.add_sessions(stamps, 3)
    assert [s.time().isoformat() for s in shifted] == ["10:30:00", "14:45:00"]
    # Every landing day must itself be a trading session.
    assert all(NSE.is_trading_day(s) for s in shifted)


def test_add_sessions_is_timezone_preserving():
    stamps = pd.DatetimeIndex(["2024-01-24 10:30"], tz="Asia/Kolkata")
    shifted = NSE.add_sessions(stamps, 2)
    assert str(shifted.tz) == "Asia/Kolkata"


def test_add_sessions_rejects_negative_horizon():
    with pytest.raises(ValueError, match="non-negative"):
        NSE.add_sessions(pd.DatetimeIndex(["2024-01-24"]), -1)


def test_session_length_is_375_minutes():
    assert NSE.minutes_in_session() == 375


def test_is_open_respects_session_bounds():
    assert NSE.is_open("2024-01-25 09:15")
    assert NSE.is_open("2024-01-25 15:30")
    assert not NSE.is_open("2024-01-25 09:14")
    assert not NSE.is_open("2024-01-25 15:31")
    assert not NSE.is_open("2024-01-26 11:00")  # holiday


def test_filter_session_drops_out_of_hours_rows():
    index = pd.DatetimeIndex(
        ["2024-01-25 08:00", "2024-01-25 10:00", "2024-01-25 16:30", "2024-01-26 11:00"]
    )
    frame = pd.Series(range(4), index=index)
    kept = NSE.filter_session(frame)
    assert list(kept.index) == [pd.Timestamp("2024-01-25 10:00")]


def test_custom_holidays_override_the_bundled_file():
    calendar = NSECalendar(holidays=["2024-01-25"])
    assert not calendar.is_trading_day("2024-01-25")
    assert calendar.is_trading_day("2024-01-26")  # not a holiday in this calendar
