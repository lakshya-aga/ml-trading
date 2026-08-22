"""Free intraday data sources for Indian equities.

Nothing here is a substitute for a paid feed — the free sources all trade away
either history, resolution, or reliability. What they are good for is developing
and debugging the pipeline before committing to a data budget, and for that they
are more than adequate.

Every loader returns the package's canonical OHLCV schema, so downstream code
cannot tell where the bytes came from.

.. warning::
   The Yahoo loader below could not be tested against the live API from the
   environment this was written in — outbound access to Yahoo was blocked by
   policy. The chunking logic is unit-tested against a stub, but the first thing
   to do on your own machine is run
   ``python -m afml_india.data.free_sources --check RELIANCE`` and confirm the
   limits it reports match what you actually get.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd

from afml_india.data.calendar import IST, NSE
from afml_india.data.loaders import _finalise
from afml_india.data.universe import to_yahoo_symbol
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

#: Yahoo's documented per-request span and total history, by interval.
#: The first number is how much one request may span; the second is how far back
#: the interval is served at all. Verify with ``--check``: Yahoo changes these
#: without notice and they differ by exchange.
YAHOO_LIMITS: dict[str, tuple[int, int]] = {
    "1m": (7, 30),
    "2m": (60, 60),
    "5m": (60, 60),
    "15m": (60, 60),
    "30m": (60, 60),
    "60m": (730, 730),
    "1h": (730, 730),
    "1d": (36500, 36500),
}


def _require_yfinance():
    try:
        import yfinance  # noqa: PLC0415

        return yfinance
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "yfinance is needed for the free intraday loaders:\n    pip install 'afml-india[data]'"
        ) from exc


def yahoo_intraday(
    symbol: str,
    interval: str = "1m",
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    exchange: str = "NSE",
    session_only: bool = True,
) -> pd.DataFrame:
    """Intraday OHLCV from Yahoo Finance, chunked around its per-request limit.

    Yahoo serves one-minute data only seven days at a time and only for the last
    thirty days. Asking for more in a single call returns an empty frame rather
    than an error, which is the usual way people conclude "Yahoo has no minute
    data for India" — it does, just not in one request. This walks the window in
    legal chunks and stitches the result.

    Parameters
    ----------
    symbol:
        NSE symbol (``RELIANCE``) or a Yahoo ticker (``RELIANCE.NS``).
    interval:
        One of :data:`YAHOO_LIMITS`. ``1m`` is the finest available.
    start, end:
        Window to fetch. Defaults to the maximum the interval allows.
    session_only:
        Drop bars outside 09:15-15:30 IST. Yahoo occasionally emits a stray bar
        at the session boundary, and one bad bar corrupts a volume threshold.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV indexed by IST timestamp. Empty if Yahoo returned
        nothing — check the interval against the limits before assuming the
        symbol is wrong.
    """
    yfinance = _require_yfinance()

    if interval not in YAHOO_LIMITS:
        raise ValueError(f"unknown interval {interval!r}; expected one of {sorted(YAHOO_LIMITS)}")
    chunk_days, max_history = YAHOO_LIMITS[interval]

    today = dt.date.today()
    end_date = pd.Timestamp(end).date() if end else today
    if start:
        start_date = pd.Timestamp(start).date()
    else:
        start_date = end_date - dt.timedelta(days=max_history - 1)

    earliest = today - dt.timedelta(days=max_history)
    if start_date < earliest:
        logger.warning(
            "Yahoo serves %s only %d days back; clamping start from %s to %s",
            interval,
            max_history,
            start_date,
            earliest,
        )
        start_date = earliest

    ticker = to_yahoo_symbol(symbol, exchange)
    frames: list[pd.DataFrame] = []
    cursor = start_date

    while cursor <= end_date:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), end_date + dt.timedelta(days=1))
        raw = yfinance.download(
            ticker,
            start=cursor.isoformat(),
            end=chunk_end.isoformat(),
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if raw is not None and len(raw):
            frames.append(raw)
        else:
            logger.debug("%s: no rows for %s to %s", ticker, cursor, chunk_end)
        cursor = chunk_end

    if not frames:
        logger.warning(
            "Yahoo returned nothing for %s at %s over %s to %s. Check the symbol, "
            "and note %s is served only %d days back.",
            ticker,
            interval,
            start_date,
            end_date,
            interval,
            max_history,
        )
        return pd.DataFrame()

    combined = pd.concat(frames)
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.get_level_values(0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    frame = _finalise(combined.reset_index(), IST, symbol)
    if session_only:
        before = len(frame)
        frame = NSE.filter_session(frame)
        if len(frame) < before:
            logger.info("Dropped %d bars outside the NSE session", before - len(frame))
    return frame


def yahoo_intraday_many(
    symbols: Iterable[str],
    interval: str = "1m",
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    exchange: str = "NSE",
) -> dict[str, pd.DataFrame]:
    """:func:`yahoo_intraday` across several symbols, skipping the ones that fail.

    One symbol returning nothing is common — a recent listing, a renamed ticker,
    a Yahoo gap — and should not abort the batch.
    """
    out: dict[str, pd.DataFrame] = {}
    symbols = list(symbols)
    for n, symbol in enumerate(symbols, 1):
        logger.info("[%d/%d] %s", n, len(symbols), symbol)
        try:
            frame = yahoo_intraday(symbol, interval, start, end, exchange)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the batch
            logger.warning("%s failed: %s", symbol, exc)
            continue
        if frame.empty:
            logger.warning("%s returned no rows", symbol)
            continue
        out[symbol] = frame
    logger.info("Fetched %d/%d symbols", len(out), len(symbols))
    return out


def check_yahoo_limits(symbol: str = "RELIANCE", exchange: str = "NSE") -> pd.DataFrame:
    """Empirically measure what Yahoo actually serves for one symbol.

    Yahoo's documented limits are approximate and change without notice. Run
    this once on your own machine and trust the output over
    :data:`YAHOO_LIMITS`.
    """
    rows = []
    for interval in ("1m", "5m", "15m", "1h", "1d"):
        try:
            frame = yahoo_intraday(symbol, interval=interval, exchange=exchange)
        except Exception as exc:  # noqa: BLE001
            rows.append({"interval": interval, "rows": 0, "error": str(exc)[:60]})
            continue
        if frame.empty:
            rows.append({"interval": interval, "rows": 0, "error": "empty"})
            continue
        rows.append(
            {
                "interval": interval,
                "rows": len(frame),
                "first": frame.index[0],
                "last": frame.index[-1],
                "days": frame.index.normalize().nunique(),
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe free intraday data availability.")
    parser.add_argument(
        "--check",
        metavar="SYMBOL",
        default="RELIANCE",
        help="NSE symbol to test (default: RELIANCE)",
    )
    parser.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"])
    args = parser.parse_args(argv)

    print(f"Probing Yahoo Finance for {args.check} on {args.exchange}...\n")
    print(check_yahoo_limits(args.check, args.exchange).to_string(index=False))
    print("\nTrust these numbers over the documented limits in YAHOO_LIMITS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
