"""Index membership helpers for the Indian equity universe.

The bundled NIFTY 50 list is a point-in-time snapshot and is offered as a
convenience for examples only. Any research that uses it as a *historical*
universe is survivorship-biased — index reconstitution in India is frequent, and
the delisted or demoted names are exactly the losers a backtest needs to see.
Pass a dated membership table to :class:`Universe` for real work.
"""

from __future__ import annotations

import pandas as pd

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

#: Point-in-time NIFTY 50 constituents (NSE symbols, no exchange suffix).
NIFTY_50_SNAPSHOT: tuple[str, ...] = (
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
)

#: Suffixes Yahoo Finance uses for the two Indian exchanges.
YAHOO_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


def to_yahoo_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Map an exchange symbol to its Yahoo Finance ticker."""
    exchange = exchange.upper()
    if exchange not in YAHOO_SUFFIX:
        raise ValueError(f"unknown exchange {exchange!r}; expected one of {sorted(YAHOO_SUFFIX)}")
    if symbol.endswith((".NS", ".BO")):
        return symbol
    return f"{symbol}{YAHOO_SUFFIX[exchange]}"


def from_yahoo_symbol(ticker: str) -> str:
    """Strip the Yahoo exchange suffix from a ticker."""
    for suffix in YAHOO_SUFFIX.values():
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


class Universe:
    """A tradable universe, optionally with dated index membership.

    Parameters
    ----------
    members:
        Either a flat sequence of symbols (a static universe) or a frame with
        ``symbol``, ``start`` and ``end`` columns describing membership
        intervals. ``end`` may be null for current members.
    """

    def __init__(self, members: pd.DataFrame | list[str] | tuple[str, ...]) -> None:
        if isinstance(members, (list, tuple)):
            self._static = tuple(members)
            self._dated: pd.DataFrame | None = None
            logger.debug("Universe built from a static list of %d symbols", len(self._static))
        else:
            missing = {"symbol", "start"} - set(members.columns)
            if missing:
                raise KeyError(f"dated membership frame is missing columns: {sorted(missing)}")
            frame = members.copy()
            frame["start"] = pd.to_datetime(frame["start"])
            frame["end"] = pd.to_datetime(frame["end"]) if "end" in frame else pd.NaT
            self._dated = frame
            self._static = tuple(sorted(frame["symbol"].unique()))

    @classmethod
    def nifty50(cls) -> Universe:
        """The bundled NIFTY 50 snapshot. Survivorship-biased; see module docs."""
        logger.warning(
            "Using the bundled NIFTY 50 snapshot: this is point-in-time and "
            "survivorship-biased. Supply a dated membership table for research."
        )
        return cls(NIFTY_50_SNAPSHOT)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every symbol that has ever been a member."""
        return self._static

    def members_on(self, day: pd.Timestamp | str) -> tuple[str, ...]:
        """Symbols in the index on ``day``.

        Falls back to the full static list when no dated table was supplied,
        which is the survivorship-biased case the constructor warns about.
        """
        if self._dated is None:
            return self._static
        stamp = pd.Timestamp(day)
        frame = self._dated
        started = frame["start"] <= stamp
        not_ended = frame["end"].isna() | (frame["end"] >= stamp)
        return tuple(sorted(frame.loc[started & not_ended, "symbol"].unique()))

    def is_member(self, symbol: str, day: pd.Timestamp | str) -> bool:
        """True when ``symbol`` was in the index on ``day``."""
        return symbol in self.members_on(day)

    def yahoo_tickers(self, exchange: str = "NSE") -> tuple[str, ...]:
        """Every symbol mapped to its Yahoo Finance ticker."""
        return tuple(to_yahoo_symbol(s, exchange) for s in self._static)

    def __len__(self) -> int:
        return len(self._static)

    def __repr__(self) -> str:
        kind = "dated" if self._dated is not None else "static"
        return f"Universe({kind}, n={len(self._static)})"
