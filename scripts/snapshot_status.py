#!/usr/bin/env python3
"""Inspect a Bloomberg snapshot while it is still being pulled.

The fetch script writes each ticker-day to disk the moment it arrives, so a pull
in progress is readable from another terminal — you do not have to wait for the
zip. This reports what has landed so far, estimates how much is left, and lets
you load partial data.

    # progress on the in-flight pull
    python scripts/snapshot_status.py

    # watch it, refreshing every 30s
    python scripts/snapshot_status.py --watch 30

    # per-ticker detail
    python scripts/snapshot_status.py --detail

    # load what has landed so far into pandas
    python scripts/snapshot_status.py --peek RIL_IN

Works on an in-progress directory, a finished directory, or a zip.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def find_target(directory: Path) -> Path:
    """Newest snapshot under ``directory`` — an in-progress tree wins over a zip.

    A live pull is almost always what you want to look at, and its directory is
    newer than any zip sitting beside it from a previous run.
    """
    if directory.is_file():
        return directory

    # Pointed straight at a snapshot tree rather than the folder holding them.
    if (directory / "index_members.csv").exists() or (directory / "manifest.json").exists():
        return directory

    candidates: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if path.is_dir() and (path / "index_members.csv").exists():
            candidates.append((path.stat().st_mtime, path))
        elif path.suffix == ".zip":
            candidates.append((path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"no snapshot found in {directory}. A pull writes to "
            f"{directory}/<index>_<asof>/ while it runs."
        )
    return max(candidates)[1]


def _directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def collect(target: Path) -> dict:
    """Gather counts from a snapshot directory or zip without loading the data."""
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as zf:
            names = zf.namelist()
            sizes = {info.filename: info.file_size for info in zf.infolist()}
            members_bytes = zf.read("index_members.csv") if "index_members.csv" in names else None
            manifest = json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
        total_bytes = sum(sizes.values())
        newest = datetime.fromtimestamp(target.stat().st_mtime)
    else:
        names = [str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()]
        members_path = target / "index_members.csv"
        members_bytes = members_path.read_bytes() if members_path.exists() else None
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        total_bytes = _directory_size(target)
        files = [p for p in target.rglob("*") if p.is_file()]
        newest = datetime.fromtimestamp(max(f.stat().st_mtime for f in files)) if files else None

    import io

    members = pd.read_csv(io.BytesIO(members_bytes)) if members_bytes else pd.DataFrame()

    trades = [n for n in names if n.startswith("ticks/trades/")]
    quotes = [n for n in names if n.startswith("ticks/quotes/")]
    daily = [n for n in names if n.startswith("daily/")]
    bars = [n for n in names if n.startswith("intraday_bars/")]

    def tickers_in(paths: list[str], dated: bool) -> set[str]:
        out = set()
        for name in paths:
            stem = Path(name).stem
            out.add(stem.rsplit("_", 1)[0] if dated else stem)
        return out

    return {
        "target": target,
        "is_zip": target.suffix == ".zip",
        "manifest": manifest,
        "members": members,
        "n_members": len(members),
        "daily_files": daily,
        "trade_files": trades,
        "quote_files": quotes,
        "bar_files": bars,
        "daily_tickers": tickers_in(daily, dated=False),
        "trade_tickers": tickers_in(trades, dated=True),
        "quote_tickers": tickers_in(quotes, dated=True),
        "total_bytes": total_bytes,
        "newest_file": newest,
        "names": names,
    }


def _bar(fraction: float, width: int = 34) -> str:
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(state: dict, started: datetime | None = None) -> str:
    """Human-readable progress summary."""
    lines: list[str] = []
    target = state["target"]
    manifest = state["manifest"]

    status = manifest.get("status")
    if state["is_zip"]:
        label = "COMPLETE (zipped)"
    elif status == "complete":
        label = "COMPLETE (not yet zipped)"
    elif status == "in_progress":
        label = "IN PROGRESS"
    else:
        label = "IN PROGRESS (no manifest yet)"

    lines.append(f"Snapshot : {target}")
    lines.append(f"Status   : {label}")
    if manifest.get("index"):
        lines.append(f"Index    : {manifest['index']}   as of {manifest.get('as_of', '?')}")

    n_members = state["n_members"]
    lines.append(f"Universe : {n_members} tickers")

    window = manifest.get("tick_window") or {}
    sessions = window.get("sessions")
    if sessions:
        lines.append(
            f"Ticks    : {window.get('start')} to {window.get('end')} ({sessions} sessions)"
        )
    lines.append("")

    # Daily history
    if n_members:
        done = len(state["daily_tickers"])
        lines.append(f"daily    {_bar(done / n_members)} {done:>4}/{n_members} tickers")

    # Tick data: expected files is tickers x sessions
    for kind, key in (("trades", "trade_tickers"), ("quotes", "quote_tickers")):
        files = state[f"{kind[:-1]}_files" if kind != "quotes" else "quote_files"]
        if not files and not sessions:
            continue
        expected = (n_members * sessions) if (n_members and sessions) else None
        touched = len(state[key])
        if expected:
            lines.append(
                f"{kind:<8} {_bar(len(files) / expected)} {len(files):>4}/{expected} files"
                f"   ({touched}/{n_members} tickers started)"
            )
        else:
            lines.append(f"{kind:<8} {len(files):>4} files ({touched} tickers)")

    if state["bar_files"]:
        lines.append(f"bars     {len(state['bar_files']):>4} files")

    lines.append("")
    lines.append(f"On disk  : {state['total_bytes'] / 1e9:.2f} GB")
    if state["newest_file"]:
        age = datetime.now() - state["newest_file"]
        stale = (
            "  <-- nothing new for a while; is the pull still running?"
            if age > timedelta(minutes=10)
            else ""
        )
        lines.append(
            f"Last file: {state['newest_file']:%H:%M:%S} ({int(age.total_seconds())}s ago){stale}"
        )

    # Throughput and ETA, from the manifest's start time when available.
    start_text = manifest.get("started_at")
    start = None
    if start_text:
        try:
            start = datetime.fromisoformat(start_text)
        except ValueError:
            start = None
    start = start or started

    total_files = len(state["trade_files"]) + len(state["quote_files"])
    if start and total_files > 0 and sessions and n_members:
        elapsed = (datetime.now() - start).total_seconds()
        expected_total = n_members * sessions * 2  # trades and quotes
        rate = total_files / max(elapsed, 1)
        remaining = max(expected_total - total_files, 0)
        lines.append(f"Elapsed  : {timedelta(seconds=int(elapsed))}")
        lines.append(f"Rate     : {rate * 60:.1f} files/min")
        if rate > 0 and remaining:
            lines.append(f"ETA      : {timedelta(seconds=int(remaining / rate))} remaining")

    # Request-level outcomes, if the report has been flushed.
    report = read_report(state)
    if report is not None and not report.empty:
        lines.append("")
        lines.append("Request outcomes so far:")
        counts = report["status"].str.split(":").str[0].value_counts()
        for status_name, count in counts.items():
            lines.append(f"  {status_name:<10} {count}")
        empties = report[report["status"] == "empty"]
        if len(empties) and len(empties) / len(report) > 0.3:
            lines.append(
                "  NOTE: a large share came back empty — check entitlements and the "
                "140-day tick window with --probe."
            )
    return "\n".join(lines)


def read_report(state: dict) -> pd.DataFrame | None:
    """The incremental fetch report, if the pull has flushed one."""
    target = state["target"]
    import io

    if state["is_zip"]:
        with zipfile.ZipFile(target) as zf:
            if "fetch_report.csv" not in zf.namelist():
                return None
            return pd.read_csv(io.BytesIO(zf.read("fetch_report.csv")))
    path = target / "fetch_report.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def render_detail(state: dict) -> str:
    """Per-ticker file counts and sizes."""
    target = state["target"]
    rows = []
    members = state["members"]
    tickers = (
        [_stem(t) for t in members["ticker"]]
        if "ticker" in members
        else sorted(state["daily_tickers"])
    )

    for ticker in tickers:
        trades = [n for n in state["trade_files"] if Path(n).stem.rsplit("_", 1)[0] == ticker]
        quotes = [n for n in state["quote_files"] if Path(n).stem.rsplit("_", 1)[0] == ticker]
        size = 0
        if not state["is_zip"]:
            for name in trades + quotes:
                path = target / name
                if path.exists():
                    size += path.stat().st_size
        rows.append(
            {
                "ticker": ticker,
                "daily": ticker in state["daily_tickers"],
                "trade_days": len(trades),
                "quote_days": len(quotes),
                "size_mb": round(size / 1e6, 1),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.to_string(index=False) if not frame.empty else "(nothing yet)"


def _stem(ticker: str) -> str:
    parts = str(ticker).replace("_", " ").split()
    if parts and parts[-1] in ("Equity", "Index", "Curncy", "Comdty", "Corp", "Govt"):
        parts = parts[:-1]
    return "_".join(parts)


def peek(state: dict, ticker: str) -> str:
    """Load and summarise what has landed for one ticker."""
    target = state["target"]
    stem = _stem(ticker)
    out: list[str] = [f"=== {stem} ==="]

    daily_path = target / "daily" / f"{stem}.csv"
    if not state["is_zip"] and daily_path.exists():
        daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date")
        out.append(
            f"\ndaily: {len(daily)} rows, {daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d}"
        )
        out.append(daily.tail(3).to_string())

    for kind in ("trades", "quotes"):
        files = sorted(
            n
            for n in state[f"{kind[:-1]}_files" if kind != "quotes" else "quote_files"]
            if Path(n).stem.rsplit("_", 1)[0] == stem
        )
        if not files:
            continue
        frames = [pd.read_csv(target / name) for name in files]
        combined = pd.concat(frames, ignore_index=True)
        combined["date_time"] = pd.to_datetime(combined["date_time"], format="mixed", utc=True)
        combined["date_time"] = combined["date_time"].dt.tz_convert("Asia/Kolkata")
        out.append(
            f"\n{kind}: {len(combined):,} rows across {len(files)} sessions, "
            f"{combined['date_time'].min()} to {combined['date_time'].max()}"
        )
        out.append(combined.head(3).to_string(index=False))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/snapshots"),
        help="Snapshot directory, zip, or the folder holding them",
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        default=None,
        help="Refresh every N seconds until interrupted",
    )
    parser.add_argument("--detail", action="store_true", help="Per-ticker breakdown")
    parser.add_argument(
        "--peek",
        metavar="TICKER",
        default=None,
        help="Load and summarise what has landed for one ticker",
    )
    parser.add_argument(
        "--report", action="store_true", help="Print the full request-outcome report"
    )
    args = parser.parse_args(argv)

    try:
        target = find_target(args.path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    started = datetime.now()

    def once() -> None:
        state = collect(target)
        print(render(state, started))
        if args.detail:
            print("\nPer-ticker:")
            print(render_detail(state))
        if args.peek:
            print()
            print(peek(state, args.peek))
        if args.report:
            report = read_report(state)
            print("\nFull report:")
            print(report.to_string(index=False) if report is not None else "(not flushed yet)")

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                print(f"(refreshing every {args.watch}s — Ctrl-C to stop)\n")
                once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped")
    else:
        once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
