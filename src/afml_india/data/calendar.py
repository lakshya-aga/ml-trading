"""NSE/BSE trading calendar.

Both exchanges keep the same equity-segment session and, in practice, the same
holiday list, so one calendar serves both. The bundled holiday file is
best-effort reference data compiled from published exchange circulars: verify it
against the exchange notice before relying on it for settlement-sensitive work,
and override it with ``NSECalendar(holidays=...)`` or ``AFML_NSE_HOLIDAYS`` when
you have an authoritative file.
"""

from __future__ import annotations

import os
from datetime import date, time
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"

#: Continuous equity session on NSE/BSE (IST).
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

#: Pre-open call-auction window; orders here set the 09:15 opening price.
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)

#: Post-close session, used here only to bound "same-day" timestamps.
POST_CLOSE_END = time(16, 0)

_HOLIDAY_ENV = "AFML_NSE_HOLIDAYS"


@lru_cache(maxsize=4)
def _load_holiday_file(path: str | None) -> tuple[pd.Timestamp, ...]:
    if path:
        frame = pd.read_csv(path)
    else:
        ref = resources.files("afml_india.data.resources").joinpath("nse_holidays.csv")
        with resources.as_file(ref) as file_path:
            frame = pd.read_csv(file_path)
    stamps = pd.to_datetime(frame["date"]).dt.normalize()
    return tuple(sorted(set(stamps)))


class NSECalendar:
    """Trading-day arithmetic for the Indian equity cash segment.

    Parameters
    ----------
    holidays:
        Optional explicit holiday list. Defaults to the bundled reference file,
        or to the CSV named by the ``AFML_NSE_HOLIDAYS`` environment variable.
    """

    def __init__(self, holidays: pd.DatetimeIndex | list[str | date] | None = None) -> None:
        if holidays is None:
            loaded = _load_holiday_file(os.environ.get(_HOLIDAY_ENV))
            self.holidays = pd.DatetimeIndex(loaded)
        else:
            self.holidays = pd.DatetimeIndex(pd.to_datetime(list(holidays))).normalize().unique()
        self.holidays = self.holidays.sort_values()
        self._holiday_set = set(self.holidays)

    # ------------------------------------------------------------------ #
    # Day-level queries
    # ------------------------------------------------------------------ #
    def is_trading_day(self, day: pd.Timestamp | str | date) -> bool:
        """True when ``day`` is a weekday that is not an exchange holiday."""
        stamp = pd.Timestamp(day).normalize()
        if stamp.weekday() >= 5:
            return False
        return stamp not in self._holiday_set

    def sessions(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
        """Trading days in ``[start, end]`` as midnight-normalised timestamps."""
        days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
        mask = ~days.isin(self.holidays)
        return days[mask]

    def num_sessions(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> int:
        """Count of trading days in ``[start, end]``."""
        return len(self.sessions(start, end))

    def shift(self, day: pd.Timestamp | str, n: int) -> pd.Timestamp:
        """Trading day ``n`` sessions away from ``day`` (negative shifts back)."""
        stamp = pd.Timestamp(day).normalize()
        if n == 0:
            return stamp if self.is_trading_day(stamp) else self.next_session(stamp)
        step = 1 if n > 0 else -1
        remaining = abs(n)
        cursor = stamp
        while remaining:
            cursor = cursor + pd.Timedelta(days=step)
            if self.is_trading_day(cursor):
                remaining -= 1
        return cursor

    def next_session(self, day: pd.Timestamp | str) -> pd.Timestamp:
        """First trading day strictly after ``day``."""
        return self.shift(day, 1)

    def previous_session(self, day: pd.Timestamp | str) -> pd.Timestamp:
        """Last trading day strictly before ``day``."""
        return self.shift(day, -1)

    # ------------------------------------------------------------------ #
    # Intraday queries
    # ------------------------------------------------------------------ #
    def session_bounds(self, day: pd.Timestamp | str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Continuous-session open and close timestamps for ``day``.

        Returned naive when ``day`` is naive, and IST-localised otherwise, so
        the result always compares cleanly with the caller's own index.
        """
        stamp = pd.Timestamp(day)
        tz = stamp.tz
        base = stamp.tz_localize(None).normalize()
        open_ts = base + pd.Timedelta(hours=MARKET_OPEN.hour, minutes=MARKET_OPEN.minute)
        close_ts = base + pd.Timedelta(hours=MARKET_CLOSE.hour, minutes=MARKET_CLOSE.minute)
        if tz is not None:
            open_ts = open_ts.tz_localize(tz)
            close_ts = close_ts.tz_localize(tz)
        return open_ts, close_ts

    def is_open(self, stamp: pd.Timestamp | str) -> bool:
        """True when ``stamp`` falls inside a continuous trading session."""
        ts = pd.Timestamp(stamp)
        if not self.is_trading_day(ts):
            return False
        open_ts, close_ts = self.session_bounds(ts)
        return open_ts <= ts <= close_ts

    def filter_session(self, obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Drop rows outside the continuous session or on non-trading days."""
        index = pd.DatetimeIndex(obj.index)
        day_ok = ~index.normalize().isin(self.holidays) & (index.weekday < 5)
        minutes = index.hour * 60 + index.minute
        open_min = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
        close_min = MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute
        time_ok = (minutes >= open_min) & (minutes <= close_min)
        return obj[day_ok & time_ok]

    def minutes_in_session(self) -> int:
        """Length of the continuous session in minutes (375 for NSE equities)."""
        open_min = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
        close_min = MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute
        return close_min - open_min

    # ------------------------------------------------------------------ #
    # Vertical barriers
    # ------------------------------------------------------------------ #
    def add_sessions(self, stamps: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
        """Vectorised ``shift`` over an index, preserving each stamp's time of day.

        This is what the triple-barrier vertical leg needs: "``n`` trading days
        later" on a market that closes for Diwali is not ``n`` calendar days.
        """
        if n < 0:
            raise ValueError("add_sessions expects a non-negative horizon")
        index = pd.DatetimeIndex(stamps)
        if len(index) == 0:
            return index
        tz = index.tz
        naive = index.tz_localize(None) if tz is not None else index
        days = naive.normalize()
        times = naive - days

        lo = days.min() - pd.Timedelta(days=10)
        hi = days.max() + pd.Timedelta(days=int(n * 2 + 30))
        grid = self.sessions(lo, hi)
        if len(grid) == 0:
            raise ValueError("no trading sessions found in the requested range")

        # Position of each stamp's day within the session grid; a stamp landing
        # on a holiday snaps forward to the next session before shifting.
        pos = np.searchsorted(grid.values, days.values, side="left")
        target = np.clip(pos + n, 0, len(grid) - 1)
        shifted = pd.DatetimeIndex(grid.values[target]) + times
        return shifted.tz_localize(tz) if tz is not None else shifted


#: Module-level default; construct your own ``NSECalendar`` to override holidays.
NSE = NSECalendar()
