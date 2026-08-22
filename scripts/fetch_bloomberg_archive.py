#!/usr/bin/env python3
"""Archive everything worth keeping from a Bloomberg terminal you are about to lose.

Intraday history is capped at ~140 days and is handled by
``fetch_bloomberg_snapshot.py``. This script archives the things that reach back
decades, ordered by how irreplaceable they are once terminal access ends:

1.  **Point-in-time index membership**, month-end, N years back
    (``INDX_MWEIGHT_HIST``). The one dataset on this list that is effectively
    unobtainable for free later, and the thing that makes every future backtest
    survivorship-free.
2.  **Daily OHLCV + turnover for the union of every historical member** —
    including delisted and demoted names, which free sources will not have.
3.  The same daily series **unadjusted**, plus **dividend/split history**
    (``DVD_HIST_ALL``), so any future data source can be validated and any
    adjustment convention re-derived forever.
4.  **Index and macro series** daily: NIFTY, Bank Nifty, India VIX, the 10-year
    G-sec yield, USDINR — small, permanently useful.
5.  A **reference snapshot** per ticker: ISIN (the join key to every other
    vendor), name, sector, exchange.

Everything is written incrementally, so an interrupted run keeps what it got,
and the tree is zipped at the end for Drive.

Usage, on the terminal machine:

    # 1. Verify the macro tickers resolve on YOUR terminal first (seconds):
    python scripts/fetch_bloomberg_archive.py --check

    # 2. The full archive (typically well under an hour):
    python scripts/fetch_bloomberg_archive.py --years 20

    # Options: --index "NIFTY Index"   --months-step 1   --skip-unadjusted
    #          --extra-tickers file.txt (one Bloomberg ticker per line)

The Bloomberg tickers in ``MACRO_TICKERS`` are best-effort from memory — this
script was written without terminal access. ``--check`` exists precisely so you
correct them in two minutes instead of discovering a typo after access is gone.
Add the NIFTY total-return index if you can confirm its ticker on the terminal;
it is the right benchmark and hard to find free.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("bbg_archive")

# Reuse the session plumbing from the snapshot script without packaging games.
_spec = importlib.util.spec_from_file_location(
    "fetch_bloomberg_snapshot", Path(__file__).parent / "fetch_bloomberg_snapshot.py"
)
_snap = importlib.util.module_from_spec(_spec)
sys.modules["fetch_bloomberg_snapshot"] = _snap
_spec.loader.exec_module(_snap)

BloombergSession = _snap.BloombergSession
fetch_index_members = _snap.fetch_index_members
qualify_ticker = _snap.qualify_ticker
zip_directory = _snap.zip_directory
_element_value = _snap._element_value
_safe_name = _snap._safe_name

DAILY_FIELDS = (
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME",
    "TURNOVER",
    "EQY_WEIGHTED_AVG_PX",
    "CUR_MKT_CAP",
)

#: Daily PX_LAST series worth keeping forever. VERIFY WITH --check: these are
#: best-effort tickers written without terminal access.
MACRO_TICKERS = {
    "NIFTY Index": "NSE Nifty 50",
    "NSEBANK Index": "Nifty Bank",
    "INVIXN Index": "India VIX",
    "GIND10YR Index": "India 10y G-sec yield (risk-free for Sharpe)",
    "USDINR Curncy": "USD/INR",
    # Add the NIFTY total-return index here once confirmed on the terminal.
}

REFERENCE_FIELDS = (
    "NAME",
    "ID_ISIN",
    "GICS_SECTOR_NAME",
    "GICS_INDUSTRY_GROUP_NAME",
    "EQY_PRIM_EXCH",
    "CRNCY",
)


# --------------------------------------------------------------------------- #
# 1. Membership history
# --------------------------------------------------------------------------- #


def fetch_membership_history(
    session: BloombergSession,
    index_ticker: str,
    dates: list[dt.date],
    out_path: Path,
) -> pd.DataFrame:
    """Month-end point-in-time composition, appended to disk as it arrives."""
    frames = []
    out_path.write_text("as_of,ticker,weight\n")
    for n, day in enumerate(dates, 1):
        try:
            members = fetch_index_members(session, index_ticker, day)
        except Exception as exc:  # noqa: BLE001 - one bad month must not kill the run
            LOG.warning("membership %s failed: %s", day, exc)
            continue
        members = members[["ticker", "weight"]].copy()
        members.insert(0, "as_of", day)
        with out_path.open("a") as handle:
            members.to_csv(handle, header=False, index=False)
        frames.append(members)
        if n % 12 == 0:
            LOG.info("membership: %d/%d dates done", n, len(dates))
    if not frames:
        raise RuntimeError(
            f"no membership data for {index_ticker} on any date — check the "
            "INDX_MWEIGHT_HIST entitlement"
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2/3. Daily history with an adjustment toggle
# --------------------------------------------------------------------------- #


def fetch_daily(
    session: BloombergSession,
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    out_dir: Path,
    adjust: bool,
    chunk_size: int = 25,
) -> list[str]:
    """Daily bars for ``tickers``, one CSV per name, adjusted or raw.

    Adjustment flags are set explicitly BOTH ways: terminals carry personal
    DPDF defaults, and an archive whose adjustment state depends on whoever was
    logged in is not an archive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        LOG.info(
            "daily (%s) %d-%d of %d",
            "adjusted" if adjust else "raw",
            i + 1,
            min(i + chunk_size, len(tickers)),
            len(tickers),
        )
        request = session.create_request("HistoricalDataRequest")
        for ticker in chunk:
            request.getElement("securities").appendValue(ticker)
        for field in DAILY_FIELDS:
            request.getElement("fields").appendValue(field)
        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
        request.set("adjustmentSplit", adjust)
        request.set("adjustmentAbnormal", adjust)
        request.set("adjustmentNormal", adjust)

        try:
            messages = session.send(request)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("chunk starting %s failed entirely: %s", chunk[0], exc)
            continue

        for msg in messages:
            if not msg.hasElement("securityData"):
                continue
            data = msg.getElement("securityData")
            ticker = data.getElementAsString("security")
            if data.hasElement("securityError"):
                LOG.warning(
                    "%s: %s", ticker, data.getElement("securityError").getElementAsString("message")
                )
                continue
            rows = []
            for point in data.getElement("fieldData").values():
                row = {"date": _element_value(point, "date")}
                for field in DAILY_FIELDS:
                    row[field.lower()] = _element_value(point, field)
                rows.append(row)
            if not rows:
                LOG.warning("%s: no rows", ticker)
                continue
            frame = pd.DataFrame(rows)
            frame["date"] = pd.to_datetime(frame["date"])
            frame.set_index("date").sort_index().to_csv(out_dir / f"{_safe_name(ticker)}.csv")
            written.append(ticker)
    return written


# --------------------------------------------------------------------------- #
# 3b. Dividend / split history
# --------------------------------------------------------------------------- #


def fetch_dividends(session: BloombergSession, tickers: list[str], out_path: Path) -> int:
    """``DVD_HIST_ALL`` per name, one long CSV. Includes splits and bonuses."""
    out_path.write_text("")
    header_written = False
    count = 0
    for n, ticker in enumerate(tickers, 1):
        request = session.create_request("ReferenceDataRequest")
        request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue("DVD_HIST_ALL")
        try:
            messages = session.send(request)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s dividends failed: %s", ticker, exc)
            continue
        rows = []
        for msg in messages:
            if not msg.hasElement("securityData"):
                continue
            for security in msg.getElement("securityData").values():
                if security.hasElement("securityError"):
                    continue
                field_data = security.getElement("fieldData")
                if not field_data.hasElement("DVD_HIST_ALL"):
                    continue
                for entry in field_data.getElement("DVD_HIST_ALL").values():
                    row = {"ticker": ticker}
                    for k in range(entry.numElements()):
                        element = entry.getElement(k)
                        row[str(element.name())] = (
                            element.getValue() if not element.isNull() else None
                        )
                    rows.append(row)
        if rows:
            frame = pd.DataFrame(rows)
            frame.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True
            count += len(rows)
        if n % 25 == 0:
            LOG.info("dividends: %d/%d tickers done", n, len(tickers))
    return count


# --------------------------------------------------------------------------- #
# 5. Reference snapshot
# --------------------------------------------------------------------------- #


def fetch_reference(session: BloombergSession, tickers: list[str], out_path: Path) -> pd.DataFrame:
    """ISIN, name, sector, exchange per ticker — the join keys to other vendors."""
    rows = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
        request = session.create_request("ReferenceDataRequest")
        for ticker in chunk:
            request.getElement("securities").appendValue(ticker)
        for field in REFERENCE_FIELDS:
            request.getElement("fields").appendValue(field)
        try:
            messages = session.send(request)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("reference chunk failed: %s", exc)
            continue
        for msg in messages:
            if not msg.hasElement("securityData"):
                continue
            for security in msg.getElement("securityData").values():
                ticker = security.getElementAsString("security")
                row = {"ticker": ticker}
                if not security.hasElement("securityError"):
                    field_data = security.getElement("fieldData")
                    for field in REFERENCE_FIELDS:
                        row[field.lower()] = _element_value(field_data, field)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_check(session: BloombergSession, extra: list[str]) -> int:
    """Resolve every macro ticker so typos die now, not after access ends."""
    print(f"{'ticker':<22}{'resolves':<10}description")
    failures = 0
    for ticker, description in {**MACRO_TICKERS, **{t: "(extra)" for t in extra}}.items():
        request = session.create_request("ReferenceDataRequest")
        request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue("PX_LAST")
        ok = False
        try:
            for msg in session.send(request):
                if msg.hasElement("securityData"):
                    for security in msg.getElement("securityData").values():
                        ok = not security.hasElement("securityError")
        except Exception:  # noqa: BLE001
            ok = False
        failures += not ok
        print(f"{ticker:<22}{'yes' if ok else 'NO':<10}{description}")
    if failures:
        print(f"\n{failures} ticker(s) failed — fix MACRO_TICKERS before the real run.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", default="NIFTY Index")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument(
        "--months-step",
        type=int,
        default=1,
        help="Membership sampling: 1 = monthly (default), 3 = quarterly",
    )
    parser.add_argument(
        "--extra-tickers",
        type=Path,
        default=None,
        help="File of extra Bloomberg tickers to archive, one per line",
    )
    parser.add_argument("--skip-unadjusted", action="store_true")
    parser.add_argument("--skip-dividends", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Only verify the macro tickers resolve, then stop"
    )
    parser.add_argument("--out", type=Path, default=Path("data/archive"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    extra = []
    if args.extra_tickers and args.extra_tickers.exists():
        extra = [
            qualify_ticker(line.strip())
            for line in args.extra_tickers.read_text().splitlines()
            if line.strip()
        ]

    today = dt.date.today()
    start = today.replace(year=today.year - args.years)
    label = f"archive_{args.index.split()[0].lower()}_{today:%Y%m%d}"
    root = args.out / label
    root.mkdir(parents=True, exist_ok=True)

    with BloombergSession(args.host, args.port) as session:
        if args.check:
            return run_check(session, extra)

        # 1. Membership — the irreplaceable one, so it goes first.
        month_ends = pd.date_range(start, today, freq=f"{args.months_step}ME")
        membership = fetch_membership_history(
            session, args.index, [d.date() for d in month_ends], root / "membership.csv"
        )
        union = sorted(membership["ticker"].unique())
        LOG.info(
            "union universe: %d tickers ever in %s over %d years",
            len(union),
            args.index,
            args.years,
        )

        universe = union + [t for t in extra if t not in union]

        # 5 early: reference snapshot is cheap and needed to interpret the rest.
        fetch_reference(session, universe, root / "reference.csv")

        # 4. Macro/index series.
        fetch_daily(session, list(MACRO_TICKERS), start, today, root / "macro", adjust=True)

        # 2. Adjusted daily for the whole union universe.
        ok_adjusted = fetch_daily(
            session, universe, start, today, root / "daily_adjusted", adjust=True
        )

        # 3. Raw daily + corporate actions.
        ok_raw = []
        if not args.skip_unadjusted:
            ok_raw = fetch_daily(
                session, universe, start, today, root / "daily_unadjusted", adjust=False
            )
        dividend_rows = 0
        if not args.skip_dividends:
            dividend_rows = fetch_dividends(session, universe, root / "dividends.csv")

        manifest = {
            "mode": "bloomberg-archive",
            "index": args.index,
            "years": args.years,
            "membership_dates": len(month_ends),
            "union_tickers": len(union),
            "daily_adjusted_ok": len(ok_adjusted),
            "daily_unadjusted_ok": len(ok_raw),
            "dividend_rows": dividend_rows,
            "macro_tickers": list(MACRO_TICKERS),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "failed_daily": sorted(set(universe) - set(ok_adjusted)),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    zip_path = args.out / f"{label}.zip"
    zip_directory(root, zip_path)
    print(f"\nArchive written to {zip_path}")
    print("Upload this to Drive alongside the tick snapshot. Failed tickers (renamed")
    print("or unresolvable) are listed under 'failed_daily' in manifest.json — map any")
    print("that matter by hand while you still have <HELP> HELP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
