#!/usr/bin/env python3
"""Point-in-time index snapshot + tick/trade pull from Bloomberg, zipped.

One self-contained script. It does four things:

1.  Resolves the **point-in-time composition** of an index as of a historical
    date (default: ten years before today) via ``INDX_MWEIGHT_HIST``. This is
    the survivorship-bias-free member list — it includes names that have since
    been dropped from the index or delisted.
2.  Pulls **daily history** for every member from the as-of date to today.
3.  Pulls **tick data** — trades and quotes — for every member, as separate
    ``TRADE`` and ``BID``/``ASK`` event streams.
4.  Writes everything to a directory tree and compresses it into a single zip.

    IMPORTANT — Bloomberg tick history is ~140 days
    ------------------------------------------------
    ``IntradayTickRequest`` and ``IntradayBarRequest`` only retain roughly 140
    calendar days of history. Tick data from ten years ago is not retrievable
    over blpapi at any price; the limit is on the B-PIPE/Server API side, not
    on this script. So the two halves of the pull use different windows:

      * the **member list** is point-in-time as of ``--asof`` (10y back),
      * the **daily bars** span ``--asof`` to today,
      * the **tick/trade data** covers the last ``--tick-days`` days (default
        20, capped at 140) *for those same point-in-time members*.

    Pass ``--tick-start`` to force an older window if your terminal's
    entitlements differ; the script will ask Bloomberg and report what actually
    came back rather than silently returning nothing.

Running it
----------
Requires a Bloomberg Terminal or B-PIPE reachable on ``localhost:8194`` and the
``blpapi`` Python package (``pip install --index-url \
https://blpapi.bloomberg.com/repository/releases/python/simple blpapi``).

    python scripts/fetch_bloomberg_snapshot.py --index "NIFTY Index"
    python scripts/fetch_bloomberg_snapshot.py --index "NIFTY Index" \
        --asof 2016-08-22 --tick-days 20 --out data/snapshots

No terminal handy? ``--offline-demo`` generates a synthetic dataset with the
exact same schema and zip layout, so the downstream notebook runs anywhere:

    python scripts/fetch_bloomberg_snapshot.py --offline-demo --members 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("bbg_snapshot")

# --------------------------------------------------------------------------- #
# Bloomberg constants
# --------------------------------------------------------------------------- #

REFDATA_SVC = "//blp/refdata"

#: Bloomberg's documented retention for intraday tick and bar history.
MAX_TICK_LOOKBACK_DAYS = 140

#: Daily fields pulled for every member. PX_* are split/dividend adjusted when
#: the adjustment flags below are set on the request.
DEFAULT_DAILY_FIELDS = (
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME",
    "TURNOVER",
    "EQY_WEIGHTED_AVG_PX",
    "CUR_MKT_CAP",
)

#: Tick event types. TRADE is the trade tape; BID/ASK are the quote stream.
TRADE_EVENTS = ("TRADE",)
QUOTE_EVENTS = ("BID", "ASK")

#: NSE continuous session in IST. Ticks are requested in UTC, so 09:15-15:30
#: IST is 03:45-10:00 UTC.
SESSION_START_UTC = dt.time(3, 45)
SESSION_END_UTC = dt.time(10, 0)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class SnapshotConfig:
    """Everything the pull needs, resolved from the command line."""

    index: str = "NIFTY Index"
    asof: dt.date = field(default_factory=lambda: _years_ago(10))
    tick_days: int = 20
    tick_start: dt.date | None = None
    tick_end: dt.date | None = None
    out_dir: Path = Path("data/snapshots")
    host: str = "localhost"
    port: int = 8194
    max_members: int | None = None
    daily_fields: tuple[str, ...] = DEFAULT_DAILY_FIELDS
    skip_ticks: bool = False
    skip_daily: bool = False
    keep_tree: bool = False
    tickers_file: Path | None = None
    list_members_only: bool = False
    tick_interval_minutes: int | None = None
    probe_only: bool = False
    offline_demo: bool = False
    demo_members: int = 5
    seed: int = 7

    @property
    def label(self) -> str:
        """Slug used for the output directory and the zip file name."""
        index_slug = self.index.replace(" ", "_").replace("/", "-").lower()
        return f"{index_slug}_{self.asof:%Y%m%d}"


def _years_ago(years: int) -> dt.date:
    today = dt.date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:  # 29 Feb
        return today.replace(year=today.year - years, day=28)


# --------------------------------------------------------------------------- #
# blpapi session plumbing
# --------------------------------------------------------------------------- #


class BloombergSession:
    """Thin context manager over a synchronous blpapi session.

    blpapi's event loop is easy to get subtly wrong — a request is not finished
    until a ``RESPONSE`` event arrives, and everything before it is a
    ``PARTIAL_RESPONSE`` that must also be drained. This wrapper hides that so
    the fetch functions read as plain calls.
    """

    def __init__(self, host: str = "localhost", port: int = 8194) -> None:
        self.host = host
        self.port = port
        self._session = None
        self._blpapi = None

    def __enter__(self) -> BloombergSession:
        try:
            import blpapi
        except ImportError as exc:  # pragma: no cover - needs a terminal
            raise ImportError(
                "blpapi is not installed. Install it from Bloomberg's index:\n"
                "  pip install --index-url "
                "https://blpapi.bloomberg.com/repository/releases/python/simple blpapi\n"
                "Or run this script with --offline-demo to generate a synthetic dataset."
            ) from exc

        self._blpapi = blpapi
        options = blpapi.SessionOptions()
        options.setServerHost(self.host)
        options.setServerPort(self.port)
        LOG.info("Connecting to Bloomberg at %s:%d", self.host, self.port)

        self._session = blpapi.Session(options)
        if not self._session.start():
            raise ConnectionError(
                f"could not start a Bloomberg session on {self.host}:{self.port}. "
                "Is the Terminal running and logged in?"
            )
        if not self._session.openService(REFDATA_SVC):
            raise ConnectionError(f"could not open {REFDATA_SVC}")
        LOG.info("Connected; %s is open", REFDATA_SVC)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._session is not None:
            self._session.stop()
            LOG.debug("Bloomberg session stopped")

    @property
    def blpapi(self):
        if self._blpapi is None:
            raise RuntimeError("session is not open; use it as a context manager")
        return self._blpapi

    def service(self):
        return self._session.getService(REFDATA_SVC)

    def create_request(self, request_type: str):
        """Create a request on ``//blp/refdata``."""
        return self.service().createRequest(request_type)

    def send(self, request) -> list:
        """Send ``request`` and return every message up to and including RESPONSE."""
        blpapi = self.blpapi
        self._session.sendRequest(request)
        messages = []
        while True:
            event = self._session.nextEvent(timeout=30_000)
            event_type = event.eventType()
            if event_type in (blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE):
                for msg in event:
                    _raise_on_response_error(msg)
                    messages.append(msg)
                if event_type == blpapi.Event.RESPONSE:
                    return messages
            elif event_type == blpapi.Event.TIMEOUT:
                raise TimeoutError("Bloomberg did not respond within 30s")


def _raise_on_response_error(msg) -> None:
    """Turn a request-level ``responseError`` into a Python exception."""
    if msg.hasElement("responseError"):
        err = msg.getElement("responseError")
        raise RuntimeError(
            f"Bloomberg responseError: {err.getElementAsString('message')} "
            f"(category={err.getElementAsString('category')})"
        )


def _element_value(element, name: str, default=None):
    """Read ``name`` off a blpapi element, tolerating absent or null fields."""
    if not element.hasElement(name):
        return default
    sub = element.getElement(name)
    if sub.isNull():
        return default
    return sub.getValue()


# --------------------------------------------------------------------------- #
# 1. Point-in-time index composition
# --------------------------------------------------------------------------- #


def fetch_index_members(
    session: BloombergSession,
    index_ticker: str,
    asof: dt.date,
) -> pd.DataFrame:
    """Index composition as it stood on ``asof``.

    Uses the ``INDX_MWEIGHT_HIST`` bulk field with ``END_DATE_OVERRIDE``, which
    is the only way to get a genuinely point-in-time member list — the plain
    ``INDX_MWEIGHT`` field always returns today's composition and would bake
    survivorship bias into everything downstream.

    Returns a frame of ``ticker``, ``weight`` and ``as_of``, where ``ticker`` is
    a fully qualified Bloomberg security (``RIL IN Equity``).
    """
    request = session.create_request("ReferenceDataRequest")
    request.getElement("securities").appendValue(index_ticker)
    request.getElement("fields").appendValue("INDX_MWEIGHT_HIST")

    overrides = request.getElement("overrides")
    override = overrides.appendElement()
    override.setElement("fieldId", "END_DATE_OVERRIDE")
    override.setElement("value", asof.strftime("%Y%m%d"))

    LOG.info("Requesting %s composition as of %s", index_ticker, asof)
    messages = session.send(request)

    rows: list[dict] = []
    for msg in messages:
        if not msg.hasElement("securityData"):
            continue
        for sec_data in msg.getElement("securityData").values():
            if sec_data.hasElement("securityError"):
                err = sec_data.getElement("securityError")
                raise RuntimeError(
                    f"{index_ticker}: {err.getElementAsString('message')}. "
                    "Check the ticker and your index entitlements."
                )
            field_data = sec_data.getElement("fieldData")
            if not field_data.hasElement("INDX_MWEIGHT_HIST"):
                continue
            for member in field_data.getElement("INDX_MWEIGHT_HIST").values():
                ticker = _element_value(member, "Index Member")
                weight = _element_value(member, "Percent Weight")
                if ticker is None:
                    continue
                rows.append({"ticker": str(ticker).strip(), "weight": weight})

    if not rows:
        raise RuntimeError(
            f"{index_ticker} returned no members for {asof}. Historical index "
            "weights need the INDX_MWEIGHT_HIST entitlement, and some indices "
            "only carry history back a limited number of years."
        )

    frame = pd.DataFrame(rows)
    # INDX_MWEIGHT_HIST returns the short form ("RIL IN"); qualify it so the
    # ticker can be used directly in subsequent data requests.
    frame["ticker"] = frame["ticker"].map(qualify_ticker)
    frame["as_of"] = pd.Timestamp(asof)
    frame = frame.sort_values("weight", ascending=False, na_position="last")
    frame = frame.reset_index(drop=True)
    LOG.info("Resolved %d members for %s as of %s", len(frame), index_ticker, asof)
    return frame


def load_ticker_file(path: Path) -> pd.DataFrame:
    """Read a universe from a CSV, accepting either a plain or a mapped layout.

    A ``bloomberg_ticker`` column is used directly. Failing that, a ``ticker``
    column is used. An ``nse_symbol`` column alone is not enough — Bloomberg
    tickers are not derivable from NSE symbols by any rule (``INFY`` is
    ``INFO IN``, ``HDFCBANK`` is ``HDFCB IN``), so the script refuses to guess.
    """
    frame = pd.read_csv(path)
    columns = {c.strip().lower(): c for c in frame.columns}

    for candidate in ("bloomberg_ticker", "ticker"):
        if candidate in columns:
            out = pd.DataFrame({"ticker": frame[columns[candidate]].map(qualify_ticker)})
            for extra in ("nse_symbol", "company", "sector", "weight"):
                if extra in columns:
                    out[extra] = frame[columns[extra]]
            if "weight" not in out.columns:
                out["weight"] = float("nan")
            LOG.info("Loaded %d tickers from %s", len(out), path)
            return out

    raise ValueError(
        f"{path} needs a 'bloomberg_ticker' or 'ticker' column. Found: "
        f"{sorted(frame.columns)}. An nse_symbol column alone is not usable — "
        "Bloomberg tickers cannot be derived from NSE symbols."
    )


def qualify_ticker(ticker: str, default_market_sector: str = "Equity") -> str:
    """Append the market sector when ``ticker`` lacks one (``RIL IN`` -> ``RIL IN Equity``)."""
    ticker = str(ticker).strip()
    known_sectors = ("Equity", "Index", "Curncy", "Comdty", "Corp", "Govt", "Mtge", "Pfd")
    if ticker.split()[-1] in known_sectors:
        return ticker
    return f"{ticker} {default_market_sector}"


# --------------------------------------------------------------------------- #
# 2. Daily history
# --------------------------------------------------------------------------- #


def fetch_daily_history(
    session: BloombergSession,
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    fields: tuple[str, ...] = DEFAULT_DAILY_FIELDS,
    chunk_size: int = 25,
) -> dict[str, pd.DataFrame]:
    """Split-and-dividend-adjusted daily bars for each ticker.

    Requests are chunked because ``HistoricalDataRequest`` caps the number of
    securities per request; a 50-name index in one shot is routinely rejected.
    """
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        LOG.info(
            "Daily history %d-%d of %d (%s to %s)",
            i + 1, min(i + chunk_size, len(tickers)), len(tickers), start, end,
        )
        request = session.create_request("HistoricalDataRequest")
        for ticker in chunk:
            request.getElement("securities").appendValue(ticker)
        for fld in fields:
            request.getElement("fields").appendValue(fld)
        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
        # Corporate-action adjustment: without these, a 1:1 bonus shows up as a
        # -50% return and every volatility estimate downstream is wrong.
        request.set("adjustmentSplit", True)
        request.set("adjustmentAbnormal", True)
        request.set("adjustmentNormal", True)

        for msg in session.send(request):
            if not msg.hasElement("securityData"):
                continue
            sec_data = msg.getElement("securityData")
            ticker = sec_data.getElementAsString("security")
            if sec_data.hasElement("securityError"):
                err = sec_data.getElement("securityError")
                LOG.warning("%s: %s", ticker, err.getElementAsString("message"))
                continue
            rows = []
            for point in sec_data.getElement("fieldData").values():
                row = {"date": _element_value(point, "date")}
                for fld in fields:
                    row[fld.lower()] = _element_value(point, fld)
                rows.append(row)
            if not rows:
                LOG.warning("%s: no daily rows returned", ticker)
                continue
            frame = pd.DataFrame(rows)
            frame["date"] = pd.to_datetime(frame["date"])
            out[ticker] = frame.set_index("date").sort_index()
    LOG.info("Daily history collected for %d/%d tickers", len(out), len(tickers))
    return out


# --------------------------------------------------------------------------- #
# 3. Tick data: trades and quotes
# --------------------------------------------------------------------------- #


def fetch_ticks(
    session: BloombergSession,
    ticker: str,
    day: dt.date,
    event_types: tuple[str, ...],
    include_condition_codes: bool = True,
) -> pd.DataFrame:
    """One trading day of tick events for one security.

    Requesting a single day at a time keeps every response comfortably under
    Bloomberg's per-request cap and makes a partial failure cost one day rather
    than the whole pull.
    """
    request = session.create_request("IntradayTickRequest")
    request.set("security", ticker)
    for event in event_types:
        request.getElement("eventTypes").appendValue(event)
    request.set("startDateTime", dt.datetime.combine(day, SESSION_START_UTC))
    request.set("endDateTime", dt.datetime.combine(day, SESSION_END_UTC))
    request.set("includeConditionCodes", include_condition_codes)
    request.set("includeExchangeCodes", True)
    request.set("includeNonPlottableEvents", False)

    rows: list[dict] = []
    for msg in session.send(request):
        if not msg.hasElement("tickData"):
            continue
        tick_data = msg.getElement("tickData").getElement("tickData")
        for tick in tick_data.values():
            rows.append(
                {
                    "date_time": _element_value(tick, "time"),
                    "type": _element_value(tick, "type"),
                    "price": _element_value(tick, "value"),
                    "volume": _element_value(tick, "size", 0),
                    "condition_codes": _element_value(tick, "conditionCodes", ""),
                    "exchange_code": _element_value(tick, "exchangeCode", ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["date_time", "type", "price", "volume", "condition_codes", "exchange_code"]
        )

    frame = pd.DataFrame(rows)
    frame["date_time"] = pd.to_datetime(frame["date_time"], utc=True)
    # Bloomberg returns tick times in UTC; convert to IST so the data lines up
    # with the 09:15-15:30 session everyone actually reasons about.
    frame["date_time"] = frame["date_time"].dt.tz_convert("Asia/Kolkata")
    return frame.sort_values("date_time").reset_index(drop=True)


def fetch_intraday_bars(
    session: BloombergSession,
    ticker: str,
    start: dt.date,
    end: dt.date,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """Intraday OHLCV bars, as a fallback when tick data is not entitled.

    ``IntradayBarRequest`` carries the same ~140-day retention as tick data but
    is entitled far more widely, and one-minute bars still resample into
    respectable rupee-value bars — you lose the true trade tape, not the method.
    """
    request = session.create_request("IntradayBarRequest")
    request.set("security", ticker)
    request.set("eventType", "TRADE")
    request.set("interval", int(interval_minutes))
    request.set("startDateTime", dt.datetime.combine(start, SESSION_START_UTC))
    request.set("endDateTime", dt.datetime.combine(end, SESSION_END_UTC))
    request.set("gapFillInitialBar", False)

    rows: list[dict] = []
    for msg in session.send(request):
        if not msg.hasElement("barData"):
            continue
        for bar in msg.getElement("barData").getElement("barTickData").values():
            rows.append(
                {
                    "date_time": _element_value(bar, "time"),
                    "open": _element_value(bar, "open"),
                    "high": _element_value(bar, "high"),
                    "low": _element_value(bar, "low"),
                    "close": _element_value(bar, "close"),
                    "volume": _element_value(bar, "volume", 0),
                    "num_events": _element_value(bar, "numEvents", 0),
                    "value": _element_value(bar, "value", 0),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["date_time", "open", "high", "low", "close", "volume", "num_events", "value"]
        )
    frame = pd.DataFrame(rows)
    frame["date_time"] = pd.to_datetime(frame["date_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    return frame.sort_values("date_time").reset_index(drop=True)


def resolve_tick_window(config: SnapshotConfig) -> tuple[dt.date, dt.date]:
    """Decide the tick window, warning when the 140-day retention bites.

    The point-in-time member list is from ten years ago, but tick history is
    not, so this is the one place the script knowingly departs from a literal
    reading of "tick data from the as-of date".
    """
    end = config.tick_end or dt.date.today() - dt.timedelta(days=1)
    if config.tick_start is not None:
        start = config.tick_start
    else:
        start = end - dt.timedelta(days=config.tick_days)

    age = (dt.date.today() - start).days
    if age > MAX_TICK_LOOKBACK_DAYS:
        LOG.warning(
            "Tick window starts %d days back, beyond Bloomberg's ~%d-day tick "
            "retention. Expect empty responses. Ticks will still be requested "
            "so the failure is visible rather than assumed.",
            age, MAX_TICK_LOOKBACK_DAYS,
        )
    return start, end


def business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays in ``[start, end]``. Exchange holidays simply return no ticks."""
    return [d.date() for d in pd.bdate_range(start, end)]


# --------------------------------------------------------------------------- #
# Entitlement and retention probe
# --------------------------------------------------------------------------- #


def probe_tick_availability(
    session: BloombergSession,
    tickers: list[str],
    max_lookback_days: int = 400,
) -> pd.DataFrame:
    """Find out what tick data this terminal will actually return.

    Two things this settles before a multi-hour pull, both of which are cheaper
    to discover now than halfway through:

    1. **Which securities have a trade tape at all.** An index does not — there
       are no trades in ``NIFTY Index``, only in its constituents and its
       futures. Pointing a tick request at an index returns an empty response
       that looks identical to a permissions failure.
    2. **How far back tick history actually goes.** Bloomberg documents roughly
       140 days, but the effective boundary depends on the terminal, so it is
       located here by bisection rather than assumed.

    Returns one row per ticker with the most recent session that returned ticks,
    the oldest that did, and the implied retention in days.
    """
    today = dt.date.today()
    rows = []

    for ticker in tickers:
        LOG.info("Probing %s", ticker)

        def has_ticks(day: dt.date, security: str = ticker) -> bool:
            try:
                return not fetch_ticks(session, security, day, TRADE_EVENTS).empty
            except Exception as exc:  # noqa: BLE001
                LOG.debug("%s on %s: %s", security, day, exc)
                return False

        # Find a recent session that works at all; a run of holidays or a
        # missing tape both look like "no data" on any single day.
        recent = None
        for back in range(1, 15):
            day = today - dt.timedelta(days=back)
            if day.weekday() >= 5:
                continue
            if has_ticks(day):
                recent = day
                break

        if recent is None:
            rows.append({
                "ticker": ticker,
                "has_trade_tape": False,
                "most_recent": None,
                "oldest": None,
                "retention_days": 0,
                "note": "no ticks in the last 14 days — an index, an unlisted "
                        "security, or no tick entitlement",
            })
            continue

        # Bisect for the oldest session that still returns ticks.
        lo, hi = max_lookback_days, 1  # lo = known-bad, hi = known-good, in days back
        while lo - hi > 2:
            mid = (lo + hi) // 2
            day = today - dt.timedelta(days=mid)
            while day.weekday() >= 5:
                day -= dt.timedelta(days=1)
            if has_ticks(day):
                hi = mid
            else:
                lo = mid

        oldest = today - dt.timedelta(days=hi)
        rows.append({
            "ticker": ticker,
            "has_trade_tape": True,
            "most_recent": recent,
            "oldest": oldest,
            "retention_days": (today - oldest).days,
            "note": "",
        })
        LOG.info("  %s: ticks back to %s (%d days)", ticker, oldest, (today - oldest).days)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Offline demo data
# --------------------------------------------------------------------------- #


def _synthetic_ticks(
    rng: np.random.Generator,
    day: dt.date,
    start_price: float,
    kind: str,
) -> pd.DataFrame:
    """A plausible intraday tape: U-shaped volume, clustered volatility, 5p ticks.

    Not a substitute for real data — it exists so the notebook and the zip
    layout can be exercised without a terminal.
    """
    n = int(rng.integers(4_000, 9_000))
    session_open = pd.Timestamp(day, tz="Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
    # U-shaped arrival intensity: trades cluster at the open and the close.
    u = rng.beta(0.45, 0.45, n)
    offsets = np.sort(u) * (375 * 60)
    stamps = session_open + pd.to_timedelta(offsets, unit="s")

    vol = 0.00035 * (1.0 + 1.5 * np.abs(u - 0.5) * 2)
    steps = rng.normal(0.0, 1.0, n) * vol
    price = start_price * np.exp(np.cumsum(steps))
    price = np.round(price / 0.05) * 0.05  # NSE tick size

    if kind == "trade":
        size = rng.lognormal(mean=3.2, sigma=1.1, size=n).astype(int) + 1
        return pd.DataFrame(
            {
                "date_time": stamps,
                "type": "TRADE",
                "price": price,
                "volume": size,
                "condition_codes": "",
                "exchange_code": "NS",
            }
        )

    # Quote stream: alternating BID/ASK around the same latent price.
    half_spread = np.maximum(0.05, np.round(price * 0.0002 / 0.05) * 0.05)
    side = np.where(np.arange(n) % 2 == 0, "BID", "ASK")
    quote = np.where(side == "BID", price - half_spread, price + half_spread)
    return pd.DataFrame(
        {
            "date_time": stamps,
            "type": side,
            "price": np.round(quote / 0.05) * 0.05,
            "volume": rng.integers(1, 5000, n),
            "condition_codes": "",
            "exchange_code": "NS",
        }
    )


def build_offline_demo(config: SnapshotConfig, root: Path) -> dict:
    """Generate a synthetic snapshot with the real schema and layout."""
    LOG.warning("Running in OFFLINE DEMO mode: all data below is synthetic.")
    rng = np.random.default_rng(config.seed)

    demo_names = [
        ("RIL IN Equity", 9.8, 980.0), ("HDFCB IN Equity", 8.1, 1120.0),
        ("INFO IN Equity", 6.4, 1015.0), ("ICICIBC IN Equity", 5.2, 245.0),
        ("TCS IN Equity", 4.9, 2450.0), ("ITC IN Equity", 4.1, 245.0),
        ("LT IN Equity", 3.6, 1480.0), ("SBIN IN Equity", 3.0, 220.0),
        ("HUVR IN Equity", 2.8, 860.0), ("BHARTI IN Equity", 2.4, 340.0),
    ]
    demo_names = demo_names[: max(1, min(config.demo_members, len(demo_names)))]

    members = pd.DataFrame(
        {
            "ticker": [t for t, _, _ in demo_names],
            "weight": [w for _, w, _ in demo_names],
            "as_of": pd.Timestamp(config.asof),
        }
    )
    (root / "index_members.csv").write_text(members.to_csv(index=False))

    # Daily history from the as-of date to today.
    daily_dir = root / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    sessions = pd.bdate_range(config.asof, dt.date.today())
    for ticker, _, px0 in demo_names:
        n = len(sessions)
        rets = rng.normal(0.0004, 0.014, n)
        close = px0 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, 0.008, n))
        frame = pd.DataFrame(
            {
                "px_open": close * (1 + rng.normal(0, 0.004, n)),
                "px_high": close * (1 + intraday),
                "px_low": close * (1 - intraday),
                "px_last": close,
                "px_volume": rng.lognormal(14.5, 0.6, n).round(),
                "turnover": 0.0,
                "eqy_weighted_avg_px": close * (1 + rng.normal(0, 0.001, n)),
                "cur_mkt_cap": close * 1e7,
            },
            index=sessions,
        )
        frame["turnover"] = frame["px_last"] * frame["px_volume"]
        frame.index.name = "date"
        frame.to_csv(daily_dir / f"{_safe_name(ticker)}.csv")

    # Ticks for the recent window only, matching the live path's behaviour.
    start, end = resolve_tick_window(config)
    days = business_days(start, end)
    report_rows = []
    for kind, subdir in (("trade", "trades"), ("quote", "quotes")):
        out_dir = root / "ticks" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for ticker, _, px0 in demo_names:
            # Carry the close across days with a small overnight gap, so the
            # concatenated tape has no artificial day-boundary jumps.
            level = px0 * 1.6
            for day in days:
                frame = _synthetic_ticks(rng, day, level, kind)
                level = float(frame["price"].iloc[-1]) * float(np.exp(rng.normal(0, 0.004)))
                path = out_dir / f"{_safe_name(ticker)}_{day:%Y%m%d}.csv"
                frame.to_csv(path, index=False)
                report_rows.append(
                    {"ticker": ticker, "kind": kind, "day": day, "rows": len(frame), "status": "ok"}
                )

    pd.DataFrame(report_rows).to_csv(root / "fetch_report.csv", index=False)
    return {
        "mode": "offline-demo",
        "synthetic": True,
        "index": config.index,
        "as_of": str(config.asof),
        "members": len(demo_names),
        "tick_window": {"start": str(start), "end": str(end), "sessions": len(days)},
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _safe_name(ticker: str) -> str:
    """Filesystem-safe form of a Bloomberg ticker (``RIL IN Equity`` -> ``RIL_IN``)."""
    parts = ticker.split()
    if parts and parts[-1] in ("Equity", "Index", "Curncy", "Comdty", "Corp", "Govt"):
        parts = parts[:-1]
    return "_".join(parts).replace("/", "-")


def run_live_pull(config: SnapshotConfig, root: Path) -> dict:
    """Execute the full Bloomberg pull into ``root``."""
    manifest: dict = {
        "mode": "bloomberg",
        "synthetic": False,
        "index": config.index,
        "as_of": str(config.asof),
    }

    with BloombergSession(config.host, config.port) as session:
        # 1. Universe: point-in-time membership, or an explicit list
        if config.tickers_file is not None:
            members = load_ticker_file(config.tickers_file)
            members["as_of"] = pd.Timestamp(config.asof)
            manifest["universe_source"] = str(config.tickers_file)
            manifest["point_in_time"] = False
        else:
            members = fetch_index_members(session, config.index, config.asof)
            manifest["universe_source"] = f"INDX_MWEIGHT_HIST @ {config.asof}"
            manifest["point_in_time"] = True

        if config.max_members:
            members = members.head(config.max_members)
        members.to_csv(root / "index_members.csv", index=False)
        tickers = members["ticker"].tolist()
        manifest["members"] = len(tickers)

        if config.probe_only:
            probe_tickers = tickers[: config.max_members or 5]
            report = probe_tick_availability(session, probe_tickers)
            report.to_csv(root / "tick_probe.csv", index=False)
            print("\nTick availability on this terminal:")
            print(report.to_string(index=False))
            manifest["mode"] = "probe"
            return manifest

        if config.list_members_only:
            LOG.info("Resolved %d members; --list-members set, stopping here", len(tickers))
            for n, ticker in enumerate(tickers, 1):
                print(f"{n:>3}. {ticker}")
            manifest["mode"] = "list-members"
            return manifest

        report_rows: list[dict] = []

        # 2. Daily history over the full ten years
        if not config.skip_daily:
            daily = fetch_daily_history(
                session, tickers, config.asof, dt.date.today(), config.daily_fields
            )
            daily_dir = root / "daily"
            daily_dir.mkdir(parents=True, exist_ok=True)
            for ticker, frame in daily.items():
                frame.to_csv(daily_dir / f"{_safe_name(ticker)}.csv")
                report_rows.append(
                    {"ticker": ticker, "kind": "daily", "day": "", "rows": len(frame), "status": "ok"}
                )
            manifest["daily_tickers"] = len(daily)

        # 3. Tick and trade data over the recent window
        if not config.skip_ticks:
            start, end = resolve_tick_window(config)
            days = business_days(start, end)
            manifest["tick_window"] = {
                "start": str(start), "end": str(end), "sessions": len(days),
                "note": (
                    "Bloomberg retains roughly 140 days of intraday tick history, "
                    "so this window is recent even though the member list is not."
                ),
            }
            if config.tick_interval_minutes:
                # Intraday-bar mode: one file per ticker covering the window.
                bar_dir = root / "intraday_bars"
                bar_dir.mkdir(parents=True, exist_ok=True)
                manifest["intraday_bar_minutes"] = config.tick_interval_minutes
                for n, ticker in enumerate(tickers, 1):
                    LOG.info("[%d/%d] %d-min bars for %s",
                             n, len(tickers), config.tick_interval_minutes, ticker)
                    try:
                        frame = fetch_intraday_bars(
                            session, ticker, start, end, config.tick_interval_minutes
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("%s intraday bars failed: %s", ticker, exc)
                        report_rows.append({"ticker": ticker, "kind": "intraday_bar",
                                            "day": "", "rows": 0, "status": f"error: {exc}"})
                        continue
                    status = "ok" if len(frame) else "empty"
                    if len(frame):
                        frame.to_csv(bar_dir / f"{_safe_name(ticker)}.csv", index=False)
                    report_rows.append({"ticker": ticker, "kind": "intraday_bar",
                                        "day": "", "rows": len(frame), "status": status})

            for kind, subdir, events in (
                ("trade", "trades", TRADE_EVENTS),
                ("quote", "quotes", QUOTE_EVENTS),
            ):
                out_dir = root / "ticks" / subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                for n, ticker in enumerate(tickers, 1):
                    LOG.info("[%d/%d] %s ticks for %s", n, len(tickers), kind, ticker)
                    for day in days:
                        try:
                            frame = fetch_ticks(session, ticker, day, events)
                        except Exception as exc:  # noqa: BLE001 - one bad day must not kill the pull
                            LOG.warning("%s %s %s failed: %s", ticker, kind, day, exc)
                            report_rows.append(
                                {"ticker": ticker, "kind": kind, "day": day,
                                 "rows": 0, "status": f"error: {exc}"}
                            )
                            continue
                        if frame.empty:
                            report_rows.append(
                                {"ticker": ticker, "kind": kind, "day": day,
                                 "rows": 0, "status": "empty"}
                            )
                            continue
                        path = out_dir / f"{_safe_name(ticker)}_{day:%Y%m%d}.csv"
                        frame.to_csv(path, index=False)
                        report_rows.append(
                            {"ticker": ticker, "kind": kind, "day": day,
                             "rows": len(frame), "status": "ok"}
                        )

        pd.DataFrame(report_rows).to_csv(root / "fetch_report.csv", index=False)
    return manifest


def zip_directory(root: Path, zip_path: Path) -> Path:
    """Compress ``root`` into ``zip_path``, storing paths relative to ``root``."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    LOG.info("Compressing %d files into %s", len(files), zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(root)))
    size_mb = zip_path.stat().st_size / 1e6
    LOG.info("Wrote %s (%.1f MB)", zip_path, size_mb)
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index", default="NIFTY Index",
                        help="Bloomberg index ticker (default: 'NIFTY Index')")
    parser.add_argument("--asof", type=_parse_date, default=None,
                        help="Point-in-time date for index composition (default: 10 years ago)")
    parser.add_argument("--tick-days", type=int, default=20,
                        help="Trading-day window of tick data to pull (default: 20)")
    parser.add_argument("--tick-start", type=_parse_date, default=None,
                        help="Explicit tick window start; overrides --tick-days")
    parser.add_argument("--tick-end", type=_parse_date, default=None,
                        help="Explicit tick window end (default: yesterday)")
    parser.add_argument("--out", dest="out_dir", type=Path, default=Path("data/snapshots"),
                        help="Output directory for the tree and the zip")
    parser.add_argument("--host", default="localhost", help="Bloomberg host")
    parser.add_argument("--port", type=int, default=8194, help="Bloomberg port")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Cap the member list, useful for a quick trial run")
    parser.add_argument("--tickers-file", type=Path, default=None,
                        help="CSV with a 'bloomberg_ticker' or 'ticker' column, used "
                             "instead of resolving point-in-time index membership")
    parser.add_argument("--list-members", action="store_true",
                        help="Resolve and print the universe, then stop (no data pulled)")
    parser.add_argument("--probe", action="store_true",
                        help="Report which securities have a trade tape and how far back "
                             "tick history actually goes on this terminal, then stop")
    parser.add_argument("--intraday-bars", type=int, default=None, metavar="MINUTES",
                        help="Also pull N-minute intraday bars. Use when tick data is "
                             "not entitled: same ~140-day retention, far wider access")
    parser.add_argument("--skip-ticks", action="store_true", help="Daily history only")
    parser.add_argument("--skip-daily", action="store_true", help="Tick data only")
    parser.add_argument("--keep-tree", action="store_true",
                        help="Keep the uncompressed tree alongside the zip")
    parser.add_argument("--offline-demo", action="store_true",
                        help="Generate synthetic data instead of calling Bloomberg")
    parser.add_argument("--members", dest="demo_members", type=int, default=5,
                        help="Number of synthetic members in --offline-demo mode")
    parser.add_argument("--seed", type=int, default=7, help="Seed for --offline-demo")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = SnapshotConfig(
        index=args.index,
        asof=args.asof or _years_ago(10),
        tick_days=min(args.tick_days, MAX_TICK_LOOKBACK_DAYS),
        tick_start=args.tick_start,
        tick_end=args.tick_end,
        out_dir=args.out_dir,
        host=args.host,
        port=args.port,
        max_members=args.max_members,
        skip_ticks=args.skip_ticks,
        skip_daily=args.skip_daily,
        keep_tree=args.keep_tree,
        offline_demo=args.offline_demo,
        demo_members=args.demo_members,
        seed=args.seed,
        tickers_file=args.tickers_file,
        list_members_only=args.list_members,
        tick_interval_minutes=args.intraday_bars,
        probe_only=args.probe,
    )

    root = config.out_dir / config.label
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        if config.offline_demo:
            manifest = build_offline_demo(config, root)
        else:
            manifest = run_live_pull(config, root)
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback wall
        LOG.error("Snapshot failed: %s", exc)
        if args.verbose:
            raise
        return 1

    if manifest.get("mode") == "probe":
        destination = config.out_dir / f"{config.label}_tick_probe.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "tick_probe.csv", destination)
        shutil.rmtree(root)
        print(f"\nProbe written to {destination}")
        return 0

    if manifest.get("mode") == "list-members":
        listing = config.out_dir / f"{config.label}_members.csv"
        listing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "index_members.csv", listing)
        shutil.rmtree(root)
        print(f"\nMember list written to {listing}")
        return 0

    manifest["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["schema"] = {
        "index_members.csv": ["ticker", "weight", "as_of"],
        "daily/<TICKER>.csv": ["date", *[f.lower() for f in config.daily_fields]],
        "ticks/trades/<TICKER>_<YYYYMMDD>.csv": [
            "date_time", "type", "price", "volume", "condition_codes", "exchange_code"
        ],
        "ticks/quotes/<TICKER>_<YYYYMMDD>.csv": [
            "date_time", "type", "price", "volume", "condition_codes", "exchange_code"
        ],
        "fetch_report.csv": ["ticker", "kind", "day", "rows", "status"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    zip_path = config.out_dir / f"{config.label}.zip"
    zip_directory(root, zip_path)
    if not config.keep_tree:
        shutil.rmtree(root)
        LOG.info("Removed the uncompressed tree; pass --keep-tree to retain it")

    print(f"\nSnapshot written to {zip_path}")
    return 0


def _parse_date(text: str) -> dt.date:
    return dt.datetime.strptime(text, "%Y-%m-%d").date()


if __name__ == "__main__":
    sys.exit(main())
