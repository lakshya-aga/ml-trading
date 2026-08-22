"""Loading OHLCV and tick data for Indian equities.

Two paths are supported: local files (CSV/Parquet — the normal case for NSE
bhavcopy or a vendor dump) and an optional Yahoo Finance fetch for quick
examples. Everything downstream consumes the same normalised schema, so the
research code never learns where the bytes came from.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from afml_india.data.calendar import IST
from afml_india.data.universe import from_yahoo_symbol, to_yahoo_symbol
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

#: Column spellings seen in NSE bhavcopy, Zerodha, Upstox and Yahoo exports.
_COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adj_close": "adj_close",
    "openprice": "open",
    "highprice": "high",
    "lowprice": "low",
    "closeprice": "close",
    "last": "close",
    "ltp": "close",
    "prevclose": "prev_close",
    "prev_close": "prev_close",
    "totaltradedqty": "volume",
    "ttl_trd_qnty": "volume",
    "qty": "volume",
    "vol": "volume",
    "volume": "volume",
    "turnover": "turnover",
    "totaltradedval": "turnover",
    "symbol": "symbol",
    "tradingsymbol": "symbol",
    "ticker": "symbol",
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and alias column names onto the package's canonical schema."""
    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = _COLUMN_ALIASES.get(key, _COLUMN_ALIASES.get(key.replace("_", " "), key))
    return frame.rename(columns=renamed)


def _finalise(
    frame: pd.DataFrame,
    tz: str | None,
    symbol: str | None,
) -> pd.DataFrame:
    frame = normalise_columns(frame)
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index)
    if tz is not None:
        frame.index = (
            frame.index.tz_localize(tz)
            if frame.index.tz is None
            else frame.index.tz_convert(tz)
        )
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "timestamp"

    for col in OHLCV_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if symbol is not None:
        frame["symbol"] = symbol
    return frame


def load_ohlcv_csv(
    path: str | Path,
    tz: str | None = IST,
    symbol: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read an OHLCV CSV and normalise it to the canonical schema.

    Parameters
    ----------
    path:
        CSV file with a date/datetime column and OHLCV columns under any of the
        common Indian-vendor spellings.
    tz:
        Timezone to localise or convert the index to. ``None`` leaves it naive.
    symbol:
        Attached as a ``symbol`` column when the file does not carry one.
    """
    frame = pd.read_csv(path, **read_csv_kwargs)
    return _finalise(frame, tz, symbol)


def load_ohlcv_parquet(
    path: str | Path,
    tz: str | None = IST,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Read an OHLCV Parquet file and normalise it."""
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    return _finalise(frame, tz, symbol)


def load_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    tz: str | None = IST,
) -> dict[str, pd.DataFrame]:
    """Load every matching file in ``directory``, keyed by file stem.

    The stem is treated as the symbol, which matches how bhavcopy-style dumps
    are normally laid out (``RELIANCE.csv``, ``INFY.csv``, ...).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    out: dict[str, pd.DataFrame] = {}
    for file in sorted(directory.glob(pattern)):
        symbol = from_yahoo_symbol(file.stem.upper())
        loader = load_ohlcv_parquet if file.suffix == ".parquet" else load_ohlcv_csv
        out[symbol] = loader(file, tz=tz, symbol=symbol)
    logger.info("Loaded %d symbols from %s", len(out), directory)
    return out


def load_yahoo(
    symbols: str | Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    interval: str = "1d",
    exchange: str = "NSE",
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance for NSE/BSE symbols.

    Requires the optional ``yfinance`` dependency and network access. Yahoo's
    Indian history is adequate for demos but has known gaps and unadjusted
    corporate actions on some names; use a vendor feed for research.
    """
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_yahoo needs the optional 'yfinance' package: pip install 'afml-india[data]'"
        ) from exc

    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = list(symbols)
    tickers = [to_yahoo_symbol(s, exchange) for s in symbols]

    raw = yfinance.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    for symbol, ticker in zip(symbols, tickers):
        frame = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        frame = frame.dropna(how="all")
        if frame.empty:
            logger.warning("Yahoo returned no rows for %s", ticker)
            continue
        out[symbol] = _finalise(frame.reset_index(), IST, symbol)
    return out


def to_panel(frames: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Pivot a symbol-keyed dict of OHLCV frames into a wide panel of ``field``."""
    series = {sym: frame[field] for sym, frame in frames.items() if field in frame}
    if not series:
        raise KeyError(f"no frame contains the column {field!r}")
    panel = pd.DataFrame(series).sort_index()
    panel.columns.name = "symbol"
    return panel


def validate_ohlcv(frame: pd.DataFrame, symbol: str = "?") -> pd.DataFrame:
    """Report structural problems in an OHLCV frame without mutating it.

    Returns a one-row frame of counts. Indian retail data commonly carries
    zero-volume rows on illiquid names and stale prints around circuits; both
    corrupt volume bars and volatility estimates if left in.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"{symbol}: OHLCV frame is missing columns {missing}")

    high, low = frame["high"], frame["low"]
    close, open_ = frame["close"], frame["open"]
    report = {
        "rows": len(frame),
        "nan_close": int(close.isna().sum()),
        "non_positive_close": int((close <= 0).sum()),
        "zero_volume": int((frame["volume"] <= 0).sum()),
        "high_below_low": int((high < low).sum()),
        "close_outside_range": int(((close > high) | (close < low)).sum()),
        "open_outside_range": int(((open_ > high) | (open_ < low)).sum()),
        "duplicate_index": int(frame.index.duplicated().sum()),
        "unsorted": int(not frame.index.is_monotonic_increasing),
    }
    return pd.DataFrame([report], index=[symbol])


def clean_ohlcv(
    frame: pd.DataFrame,
    drop_zero_volume: bool = True,
    max_return: float | None = 0.5,
) -> pd.DataFrame:
    """Drop structurally impossible rows and implausible single-bar returns.

    ``max_return`` guards against unadjusted splits and bad prints. It defaults
    to 50%, comfortably outside the 20% circuit band that caps a normal NSE
    session, so genuine limit moves survive while a 1:2 split does not.
    """
    out = frame.copy()
    ok = out["close"].notna() & (out["close"] > 0)
    if {"high", "low"} <= set(out.columns):
        ok &= out["high"] >= out["low"]
    if drop_zero_volume and "volume" in out.columns:
        ok &= out["volume"] > 0
    out = out[ok]

    if max_return is not None and len(out) > 1:
        rets = out["close"].pct_change()
        bad = rets.abs() > max_return
        if bad.any():
            logger.warning("Dropping %d bars with |return| > %.0f%%", int(bad.sum()), max_return * 100)
        out = out[~bad.fillna(False)]
    return out


def adjust_for_splits(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    price_columns: Iterable[str] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Back-adjust prices and volume for splits and bonus issues.

    ``actions`` needs an ``ex_date`` column and a ``ratio`` column, where the
    ratio is the multiplicative factor applied to the share count (2.0 for a 1:1
    bonus, 5.0 for a Rs 10 to Rs 2 split). Prices before the ex-date are divided
    by the cumulative factor and volumes multiplied, so returns stay continuous.
    """
    if actions.empty:
        return frame.copy()
    for col in ("ex_date", "ratio"):
        if col not in actions.columns:
            raise KeyError(f"actions is missing required column {col!r}")

    out = frame.copy()
    events = actions.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    if out.index.tz is not None:
        events["ex_date"] = events["ex_date"].dt.tz_localize(out.index.tz)
    events = events.sort_values("ex_date")

    # Cumulative factor applied to each bar: the product of every ratio whose
    # ex-date is strictly after that bar.
    factor = pd.Series(1.0, index=out.index)
    for _, event in events.iterrows():
        factor[out.index < event["ex_date"]] *= float(event["ratio"])

    for col in price_columns:
        if col in out.columns:
            out[col] = out[col] / factor
    if "volume" in out.columns:
        out["volume"] = out["volume"] * factor
    return out
