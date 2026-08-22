"""Read the zip produced by ``scripts/fetch_bloomberg_snapshot.py``.

The snapshot is deliberately a plain zip of CSVs rather than a database: it is
inspectable, portable between a Bloomberg-connected desktop and wherever the
research actually runs, and small enough to keep beside the code. This module
reads it without unpacking to disk.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

TICK_KINDS = ("trades", "quotes")


class Snapshot:
    """A point-in-time index snapshot with daily history and tick data.

    Parameters
    ----------
    path:
        Path to the ``.zip`` written by the fetch script, or to an unpacked
        directory with the same layout.
    tz:
        Timezone applied to tick timestamps. Defaults to IST.
    """

    def __init__(self, path: str | Path, tz: str = "Asia/Kolkata") -> None:
        self.path = Path(path)
        self.tz = tz
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist. Generate one with:\n"
                "    python scripts/fetch_bloomberg_snapshot.py --offline-demo"
            )
        self._zip = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None
        self._names = (
            set(self._zip.namelist())
            if self._zip
            else {str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file()}
        )

    # ------------------------------------------------------------------ #
    # Low-level access
    # ------------------------------------------------------------------ #
    def _read(self, name: str) -> bytes:
        if name not in self._names:
            raise KeyError(f"{name} is not in {self.path.name}")
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def _read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self._read(name)), **kwargs)

    def contents(self, prefix: str = "") -> list[str]:
        """Every file in the snapshot, optionally filtered by path prefix."""
        return sorted(n for n in self._names if n.startswith(prefix))

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @property
    def manifest(self) -> dict:
        """Provenance: index, as-of date, tick window, and whether it is synthetic."""
        return json.loads(self._read("manifest.json"))

    @property
    def is_synthetic(self) -> bool:
        """True for a snapshot built with ``--offline-demo``."""
        return bool(self.manifest.get("synthetic", False))

    @property
    def members(self) -> pd.DataFrame:
        """Point-in-time index composition: ``ticker``, ``weight``, ``as_of``."""
        frame = self._read_csv("index_members.csv")
        frame["as_of"] = pd.to_datetime(frame["as_of"])
        return frame

    @property
    def tickers(self) -> list[str]:
        """Filesystem-safe ticker stems present in the snapshot (``RIL_IN``)."""
        stems = {Path(n).stem for n in self._names if n.startswith("daily/")}
        if not stems:
            stems = {
                Path(n).stem.rsplit("_", 1)[0]
                for n in self._names
                if n.startswith("ticks/")
            }
        return sorted(stems)

    @property
    def report(self) -> pd.DataFrame:
        """Per-request outcome log: which pulls returned rows, which came back empty."""
        return self._read_csv("fetch_report.csv")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def daily(self, ticker: str) -> pd.DataFrame:
        """Daily OHLCV history for one ticker, indexed by date."""
        frame = self._read_csv(f"daily/{_stem(ticker)}.csv")
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date").sort_index()

    def daily_panel(self, field: str = "px_last", tickers: Iterable[str] | None = None) -> pd.DataFrame:
        """Wide panel of one daily field across tickers."""
        tickers = list(tickers) if tickers is not None else self.tickers
        series = {}
        for ticker in tickers:
            try:
                series[ticker] = self.daily(ticker)[field]
            except (KeyError, FileNotFoundError):
                logger.warning("No daily %s for %s", field, ticker)
        if not series:
            raise KeyError(f"no daily data for field {field!r}")
        return pd.DataFrame(series).sort_index()

    def tick_days(self, ticker: str, kind: str = "trades") -> list[str]:
        """Dates (``YYYYMMDD``) available for a ticker's tick data."""
        if kind not in TICK_KINDS:
            raise ValueError(f"kind must be one of {TICK_KINDS}")
        stem = _stem(ticker)
        prefix = f"ticks/{kind}/{stem}_"
        return sorted(Path(n).stem.rsplit("_", 1)[1] for n in self._names if n.startswith(prefix))

    def ticks(
        self,
        ticker: str,
        kind: str = "trades",
        days: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Concatenated tick data for a ticker, indexed by timestamp.

        ``kind='trades'`` is the trade tape (what bars are built from);
        ``kind='quotes'`` is the BID/ASK stream (what spread and
        microstructure features are built from).
        """
        if kind not in TICK_KINDS:
            raise ValueError(f"kind must be one of {TICK_KINDS}")
        stem = _stem(ticker)
        wanted = list(days) if days is not None else self.tick_days(ticker, kind)
        if not wanted:
            raise KeyError(f"no {kind} tick files for {ticker} in {self.path.name}")

        frames = []
        for day in wanted:
            name = f"ticks/{kind}/{stem}_{day}.csv"
            if name not in self._names:
                logger.warning("Missing %s", name)
                continue
            frames.append(self._read_csv(name))
        if not frames:
            raise KeyError(f"none of the requested days exist for {ticker}")

        out = pd.concat(frames, ignore_index=True)
        out["date_time"] = pd.to_datetime(out["date_time"], format="mixed", utc=True)
        out["date_time"] = out["date_time"].dt.tz_convert(self.tz)
        out = out.sort_values("date_time").set_index("date_time")
        out.index.name = "timestamp"
        return out

    def most_liquid(self, n: int = 1, kind: str = "trades") -> list[str]:
        """The ``n`` tickers with the most tick rows — a good default to experiment on."""
        report = self.report
        if report.empty or not {"kind", "ticker", "rows"} <= set(report.columns):
            return self.tickers[:n]
        wanted_kind = "trade" if kind == "trades" else "quote"
        counts = report[report["kind"] == wanted_kind]
        if counts.empty:
            return self.tickers[:n]
        totals = counts.groupby("ticker")["rows"].sum().sort_values(ascending=False)
        return [_stem(t) for t in totals.head(n).index]

    def __repr__(self) -> str:
        info = self.manifest
        tag = " SYNTHETIC" if info.get("synthetic") else ""
        return (
            f"Snapshot({self.path.name}{tag}, index={info.get('index')}, "
            f"as_of={info.get('as_of')}, members={info.get('members')})"
        )


def _stem(ticker: str) -> str:
    """Normalise ``'RIL IN Equity'``, ``'RIL IN'`` or ``'RIL_IN'`` to ``'RIL_IN'``."""
    ticker = str(ticker).strip()
    parts = ticker.replace("_", " ").split()
    if parts and parts[-1] in ("Equity", "Index", "Curncy", "Comdty", "Corp", "Govt"):
        parts = parts[:-1]
    return "_".join(parts)


def find_snapshot(directory: str | Path = "data/snapshots") -> Path:
    """Most recently modified snapshot zip in ``directory``."""
    directory = Path(directory)
    zips = sorted(directory.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(
            f"no snapshot zip in {directory}. Create one with:\n"
            "    python scripts/fetch_bloomberg_snapshot.py --offline-demo"
        )
    return zips[0]
