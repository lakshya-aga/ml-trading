#!/usr/bin/env python3
"""Turn a folder of intraday OHLCV files into a snapshot the notebooks can read.

Free minute data arrives in whatever shape the source felt like — a Kaggle dump,
a broker API export, a yfinance pull, a vendor sample. This normalises any of
them into the same zip the Bloomberg script produces, so the rest of the project
does not care where the data came from.

    # one CSV per symbol, named RELIANCE.csv, INFY.csv, ...
    python scripts/build_snapshot_from_bars.py data/raw/minute --out data/snapshots

    # one long file with a symbol column
    python scripts/build_snapshot_from_bars.py all_minute_data.csv --long --out data/snapshots

    # straight from Yahoo (needs network and yfinance)
    python scripts/build_snapshot_from_bars.py --yahoo RELIANCE,INFY,TCS --interval 5m

Column names are auto-detected across the common spellings — ``date``/
``datetime``/``timestamp``, ``o/h/l/c/v``, ``open``/``high``/..., Kaggle's
``Date``+``Time`` pair. Anything it cannot resolve is reported rather than
guessed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_snapshot")

IST = "Asia/Kolkata"

#: Column spellings seen across Kaggle dumps, broker exports and vendor samples.
_ALIASES = {
    "date": "date",
    "dt": "date",
    "day": "date",
    "time": "time",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "timestamp": "timestamp",
    "o": "open",
    "open": "open",
    "h": "high",
    "high": "high",
    "l": "low",
    "low": "low",
    "c": "close",
    "close": "close",
    "last": "close",
    "ltp": "close",
    "v": "volume",
    "volume": "volume",
    "vol": "volume",
    "qty": "volume",
    "oi": "open_interest",
    "open_interest": "open_interest",
    "symbol": "symbol",
    "ticker": "symbol",
    "name": "symbol",
    "instrument": "symbol",
}

REQUIRED = ("open", "high", "low", "close", "volume")


def normalise(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map arbitrary column names onto the canonical intraday-bar schema."""
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        renamed[column] = _ALIASES.get(key, key)
    frame = frame.rename(columns=renamed)

    # A separate date and time column is the usual Kaggle layout.
    if "timestamp" not in frame.columns:
        if {"date", "time"} <= set(frame.columns):
            frame["timestamp"] = pd.to_datetime(
                frame["date"].astype(str) + " " + frame["time"].astype(str),
                errors="coerce",
            )
        elif "date" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
        else:
            raise ValueError(
                f"{source}: no timestamp column found. Looked for "
                f"timestamp/datetime/date(+time). Columns present: {sorted(frame.columns)}"
            )
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Columns present: {sorted(frame.columns)}"
        )

    frame = frame.dropna(subset=["timestamp"])
    for column in REQUIRED:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close"])

    if frame["timestamp"].dt.tz is None:
        # Indian intraday files are IST wall-clock time; NSE has no DST, so this
        # is unambiguous.
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(IST)
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(IST)

    frame = frame.sort_values("timestamp")
    frame = frame[~frame["timestamp"].duplicated(keep="last")]
    return frame


def to_bar_file(frame: pd.DataFrame) -> pd.DataFrame:
    """Shape a normalised frame into the snapshot's intraday-bar layout."""
    out = pd.DataFrame(
        {
            "date_time": frame["timestamp"],
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "volume": frame["volume"],
        }
    )
    out["num_events"] = np.nan
    # Turnover at the typical price. Approximate, but it is what rupee-value
    # bars are thresholded on, so it needs to exist.
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    out["value"] = typical * out["volume"]
    return out.reset_index(drop=True)


def to_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample intraday bars to daily, matching the snapshot's daily schema."""
    indexed = frame.set_index("timestamp")
    grouped = indexed.resample("1D")
    daily = pd.DataFrame(
        {
            "px_open": grouped["open"].first(),
            "px_high": grouped["high"].max(),
            "px_low": grouped["low"].min(),
            "px_last": grouped["close"].last(),
            "px_volume": grouped["volume"].sum(),
        }
    ).dropna(subset=["px_last"])

    typical = (indexed["high"] + indexed["low"] + indexed["close"]) / 3.0
    turnover = (typical * indexed["volume"]).resample("1D").sum()
    daily["turnover"] = turnover.reindex(daily.index)
    volume = daily["px_volume"].replace(0, np.nan)
    daily["eqy_weighted_avg_px"] = (daily["turnover"] / volume).fillna(daily["px_last"])
    daily["cur_mkt_cap"] = np.nan

    daily.index = daily.index.tz_localize(None).normalize()
    daily.index.name = "date"
    return daily


def infer_interval(frame: pd.DataFrame) -> str:
    """Modal spacing between bars, as a human-readable string."""
    gaps = frame["timestamp"].diff().dropna()
    if gaps.empty:
        return "unknown"
    modal = gaps.mode()
    if modal.empty:
        return "unknown"
    seconds = int(modal.iloc[0].total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def load_sources(
    path: Path | None,
    long_format: bool,
    yahoo: list[str] | None,
    interval: str,
    start: str | None,
    end: str | None,
) -> dict[str, pd.DataFrame]:
    """Read every input into ``{symbol: normalised frame}``."""
    if yahoo:
        from afml_india.data.free_sources import yahoo_intraday_many  # noqa: PLC0415

        fetched = yahoo_intraday_many(yahoo, interval=interval, start=start, end=end)
        out = {}
        for symbol, frame in fetched.items():
            reset = frame.reset_index().rename(columns={"timestamp": "timestamp"})
            out[symbol] = normalise(reset, f"yahoo:{symbol}")
        return out

    if path is None:
        raise ValueError("supply an input path, or --yahoo SYMBOLS")

    if path.is_file():
        raw = pd.read_csv(path) if path.suffix != ".parquet" else pd.read_parquet(path)
        if not long_format:
            return {path.stem.upper(): normalise(raw, path.name)}

        # Split by symbol BEFORE normalising. Normalising the whole file first
        # would deduplicate timestamps across symbols — every symbol shares the
        # same minute grid, so that silently discards all but one of them.
        renamed = {}
        for column in raw.columns:
            key = str(column).strip().lower().replace(" ", "_")
            renamed[column] = _ALIASES.get(key, key)
        raw = raw.rename(columns=renamed)
        if "symbol" not in raw.columns:
            raise ValueError(
                f"{path.name}: --long needs a symbol column (symbol/ticker/name/"
                f"instrument). Columns present: {sorted(raw.columns)}"
            )

        out: dict[str, pd.DataFrame] = {}
        for symbol, group in raw.groupby("symbol"):
            name = str(symbol).strip().upper()
            try:
                out[name] = normalise(group.drop(columns=["symbol"]), f"{path.name}:{name}")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping %s: %s", name, exc)
        if not out:
            raise ValueError(f"{path.name}: no symbol produced usable rows")
        return out

    if not path.is_dir():
        raise NotADirectoryError(path)

    out: dict[str, pd.DataFrame] = {}
    files = sorted([*path.glob("*.csv"), *path.glob("*.parquet"), *path.glob("*.txt")])
    if not files:
        raise FileNotFoundError(f"no .csv/.parquet/.txt files in {path}")

    for file in files:
        try:
            raw = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
            out[file.stem.upper()] = normalise(raw, file.name)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            LOG.warning("Skipping %s: %s", file.name, exc)
    if not out:
        raise ValueError(f"no usable files in {path}")
    return out


def build(
    sources: dict[str, pd.DataFrame],
    out_dir: Path,
    label: str,
    keep_tree: bool,
    provenance: str,
) -> Path:
    """Write the snapshot tree and compress it."""
    root = out_dir / label
    if root.exists():
        shutil.rmtree(root)
    (root / "intraday_bars").mkdir(parents=True)
    (root / "daily").mkdir(parents=True)

    report_rows, intervals = [], []
    for symbol, frame in sources.items():
        safe = symbol.replace("/", "-").replace(" ", "_")
        bars = to_bar_file(frame)
        bars.to_csv(root / "intraday_bars" / f"{safe}.csv", index=False)
        to_daily(frame).to_csv(root / "daily" / f"{safe}.csv")

        interval = infer_interval(frame)
        intervals.append(interval)
        report_rows.append(
            {
                "ticker": symbol,
                "kind": "intraday_bar",
                "day": "",
                "rows": len(bars),
                "status": "ok",
            }
        )
        LOG.info(
            "%-14s %6d bars at %-6s  %s -> %s",
            symbol,
            len(bars),
            interval,
            frame["timestamp"].iloc[0].date(),
            frame["timestamp"].iloc[-1].date(),
        )

    pd.DataFrame(
        {"ticker": list(sources), "weight": np.nan, "as_of": pd.Timestamp.today().normalize()}
    ).to_csv(root / "index_members.csv", index=False)
    pd.DataFrame(report_rows).to_csv(root / "fetch_report.csv", index=False)

    spans = [(f["timestamp"].min(), f["timestamp"].max()) for f in sources.values()]
    manifest = {
        "mode": "imported-bars",
        "synthetic": False,
        "status": "complete",
        "index": provenance,
        "as_of": str(pd.Timestamp.today().date()),
        "members": len(sources),
        "point_in_time": False,
        "intraday_bar_interval": pd.Series(intervals).mode().iloc[0] if intervals else "unknown",
        "tick_window": {
            "start": str(min(s for s, _ in spans).date()),
            "end": str(max(e for _, e in spans).date()),
            "sessions": int(max(f["timestamp"].dt.normalize().nunique() for f in sources.values())),
            "note": (
                "Intraday BARS, not ticks. Use Snapshot.pseudo_ticks() to feed the "
                "bar builders; within-bar path information is not present."
            ),
        },
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    zip_path = out_dir / f"{label}.zip"
    files = sorted(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in files:
            zf.write(file, arcname=str(file.relative_to(root)))
    LOG.info("Wrote %s (%.1f MB, %d files)", zip_path, zip_path.stat().st_size / 1e6, len(files))

    if not keep_tree:
        shutil.rmtree(root)
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Directory of per-symbol files, or a single file",
    )
    parser.add_argument(
        "--long", action="store_true", help="Input is one long file with a symbol column"
    )
    parser.add_argument(
        "--yahoo",
        default=None,
        metavar="SYMBOLS",
        help="Comma-separated NSE symbols to fetch from Yahoo instead",
    )
    parser.add_argument("--interval", default="5m", help="Interval for --yahoo (default: 5m)")
    parser.add_argument("--start", default=None, help="Start date for --yahoo")
    parser.add_argument("--end", default=None, help="End date for --yahoo")
    parser.add_argument("--out", dest="out_dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--label", default=None, help="Snapshot name (default: derived)")
    parser.add_argument("--keep-tree", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    yahoo = [s.strip().upper() for s in args.yahoo.split(",")] if args.yahoo else None
    try:
        sources = load_sources(args.path, args.long, yahoo, args.interval, args.start, args.end)
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s", exc)
        if args.verbose:
            raise
        return 1

    provenance = f"yahoo {args.interval}" if yahoo else f"imported from {args.path}"
    label = args.label or (
        f"yahoo_{args.interval}_{dt.date.today():%Y%m%d}"
        if yahoo
        else f"imported_{dt.date.today():%Y%m%d}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = build(sources, args.out_dir, label, args.keep_tree, provenance)

    print(f"\nSnapshot written to {zip_path}")
    print("\nRead it with:")
    print("    from afml_india.research import Snapshot")
    print(f"    snap = Snapshot('{zip_path}')")
    print("    ticks = snap.pseudo_ticks(snap.tickers[0])   # bars, reshaped -- not a tape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
