# Getting the data

Everything the notebooks need comes from one script,
`scripts/fetch_bloomberg_snapshot.py`, which writes a single zip. Run it on a
Bloomberg-connected machine, put the zip on Drive, and pull it down wherever the
research actually runs.

---

## What the project uses

| Layer | Source | Window | Used by |
|---|---|---|---|
| Point-in-time index membership | `INDX_MWEIGHT_HIST` | the as-of date | universe for every notebook |
| Daily OHLCV, split/dividend adjusted | `HistoricalDataRequest` | as-of date → today | notebooks 03 (monthly means) |
| Trade ticks | `IntradayTickRequest`, `TRADE` | last ~140 days | notebooks 01, 02, 04 (bars) |
| Quote ticks | `IntradayTickRequest`, `BID`/`ASK` | last ~140 days | microstructure features |
| Intraday bars *(fallback)* | `IntradayBarRequest` | last ~140 days | when ticks are not entitled |

### The 140-day limit is real and unavoidable — confirmed on this terminal

Bloomberg retains roughly **140 calendar days** of intraday tick and bar history.
Tick data from ten years ago cannot be retrieved over `blpapi` at any price — the
limit is server-side, not a script restriction.

This is not a documentation claim we are taking on trust. `bloomberg_test.ipynb` in
this repo runs an `IntradayTickRequest` against `N50FUTPR Index` for
2020-08-21 to 2020-08-30 on a live terminal, and the response is:

```
IntradayTickResponse = {
    tickData = {
        eidData[] = { }
        tickData[] = { }
    }
}
```

Empty. No error, no permission failure — the data simply is not retained.

Find your own terminal's actual boundary rather than assuming 140:

```bash
python scripts/fetch_bloomberg_snapshot.py --probe --max-members 5
```

That bisects for the oldest session that still returns ticks, per security, and
writes `data/snapshots/<label>_tick_probe.csv`.

### Indices have no trade tape

The same test notebook requests ticks for `NIFTY Index` and gets nothing back. That is
expected and is not an entitlement problem: an index is a computed level, so there are
no trades in it. Tick data exists for its **constituents** (`RIL IN Equity`) and for
its **futures** (`N50FUTPR Index`).

The `--probe` output distinguishes the two cases — a security with no trade tape is
reported as `has_trade_tape=False` rather than as a retention limit — so you find out
in seconds instead of at the end of a long pull.

So the snapshot deliberately mixes horizons:

- the **member list** is point-in-time as of `--asof` (ten years back by default),
- the **daily bars** span that date to today,
- the **tick data** covers the last `--tick-days` sessions *for those same
  point-in-time members*.

The manifest records this, and the script will still issue a request for an older
window if you force one with `--tick-start`, so a refusal is visible rather than
assumed.

---

## Which tickers

The universe is **resolved from Bloomberg**, not hard-coded. Check what you will get
before pulling anything:

```bash
python scripts/fetch_bloomberg_snapshot.py --list-members --asof 2016-08-22
```

That prints the members and writes `data/snapshots/nifty_index_20160822_members.csv`.
Nothing is downloaded.

A current-composition reference list lives at
`data/reference/nifty50_bloomberg_tickers.csv` (50 names, with NSE symbol, Bloomberg
ticker and sector). Use it as a **fallback** if `INDX_MWEIGHT_HIST` is not entitled on
your terminal:

```bash
python scripts/fetch_bloomberg_snapshot.py \
    --tickers-file data/reference/nifty50_bloomberg_tickers.csv
```

Two caveats on that file, repeated from `data/reference/README.md` because they matter:
it is **today's** membership, so using it as a historical universe is
survivorship-biased; and its Bloomberg tickers are best-effort, not machine-verified.
Bloomberg tickers follow no rule from NSE symbols — `INFY` is `INFO IN`, `HDFCBANK` is
`HDFCB IN`, `M&M` is `MM IN` — so verify them on the terminal.

---

## Running the pull

### Prerequisites

- Bloomberg Terminal or B-PIPE running and logged in, reachable on `localhost:8194`
- `blpapi`:

```bash
pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
pip install pandas numpy
```

### Probe first

Before anything else, find out what this terminal will actually return:

```bash
python scripts/fetch_bloomberg_snapshot.py --probe --asof 2016-08-22 --max-members 5
```

Output is one row per security: whether it has a trade tape, the oldest session that
returned ticks, and the implied retention in days. Set `--tick-days` from that number
rather than from the documented 140.

### Watching a pull in progress

A 40-ticker pull takes hours. You do not have to wait for the zip — the script
writes each ticker-day to disk the moment it arrives, so the tree at
`data/snapshots/<index>_<asof>/` is readable from another terminal while the pull
runs:

```bash
python scripts/snapshot_status.py                 # progress, throughput, ETA
python scripts/snapshot_status.py --watch 30      # refresh every 30s
python scripts/snapshot_status.py --detail        # per-ticker file counts
python scripts/snapshot_status.py --peek RIL_IN   # load what has landed for one name
python scripts/snapshot_status.py --report        # every request and its outcome
```

Or read the partial data directly — `Snapshot` accepts a directory, not just a zip:

```python
from afml_india.research import Snapshot, build_bars
snap = Snapshot("data/snapshots/nifty_index_20160822")   # no .zip: the live tree
bars = build_bars(snap.ticks("RIL_IN", "trades"), kind="dollar", bars_per_day=50)
```

**Do not re-run the fetch script into the same `--out` while a pull is running.**
It refuses by default now, but the reason matters: the second process would
otherwise delete the tree the first is writing into. Use `--resume` to continue an
interrupted pull (it skips ticker-days already on disk), `--overwrite` to start
again deliberately, or `--out` to write elsewhere.

### Then start small

Confirm entitlements on three names and five sessions before committing to the full pull:

```bash
python scripts/fetch_bloomberg_snapshot.py \
    --index "NIFTY Index" --asof 2016-08-22 \
    --max-members 3 --tick-days 5 \
    --out data/snapshots --keep-tree -v
```

Then read `fetch_report.csv` in the output. Every request is logged with a status of
`ok`, `empty`, or `error: …`, so a permissions gap shows up as a row rather than as
silently missing data.

### The full pull

```bash
python scripts/fetch_bloomberg_snapshot.py \
    --index "NIFTY Index" \
    --asof 2016-08-22 \
    --tick-days 120 \
    --out data/snapshots
```

Rough expectations for 50 names over 120 sessions:

- **runtime** — several hours. Requests are one ticker-day at a time, which keeps every
  response small and makes a failure cost one day rather than the whole pull.
- **size** — 20–60 GB uncompressed for trades and quotes on liquid large caps; the zip
  compresses CSV tick data to roughly a third of that.

If that is too large, the useful knobs in order are `--tick-days`, `--max-members`, and
dropping quotes (bars only need trades — quotes matter for spread and microstructure
features).

### How far back can I actually go?

Short answer: **ticks and one-minute bars, ~140 days; daily, decades. There is no
intraday product on the Desktop API that reaches further back** —
`IntradayBarRequest` hits the same cliff as `IntradayTickRequest`.

For the full picture — the limit per request type, and where deep Indian tick
history is actually sold, including the qualification that Bloomberg *does* sell
it but not through this API — see **[DATA_SOURCES.md](DATA_SOURCES.md)**.

### If tick data is not entitled

One-minute bars carry the same 140-day retention and much wider entitlement. They still
resample into respectable rupee-value bars; you lose the true trade tape, not the method:

```bash
python scripts/fetch_bloomberg_snapshot.py \
    --index "NIFTY Index" --tick-days 120 --intraday-bars 1 --skip-ticks
```

### Other useful invocations

```bash
# Daily history only -- fast, small, enough for notebook 03
python scripts/fetch_bloomberg_snapshot.py --skip-ticks

# A specific tick window
python scripts/fetch_bloomberg_snapshot.py --tick-start 2026-05-01 --tick-end 2026-08-20

# No terminal? A synthetic snapshot with the identical schema, so the notebooks run
python scripts/fetch_bloomberg_snapshot.py --offline-demo --members 5 --tick-days 90
```

---

## What lands in the zip

```
nifty_index_20160822.zip
├── manifest.json                          provenance: index, as-of, tick window, synthetic flag
├── index_members.csv                      ticker, weight, as_of
├── fetch_report.csv                       ticker, kind, day, rows, status  -- read this first
├── daily/
│   └── RIL_IN.csv                         date, px_open, px_high, px_low, px_last,
│                                          px_volume, turnover, eqy_weighted_avg_px, cur_mkt_cap
├── ticks/
│   ├── trades/RIL_IN_20260817.csv         date_time, type, price, volume,
│   └── quotes/RIL_IN_20260817.csv         condition_codes, exchange_code
└── intraday_bars/                         only with --intraday-bars
    └── RIL_IN.csv                         date_time, open, high, low, close, volume, num_events, value
```

Tick timestamps are converted from Bloomberg's UTC to **IST**, so they line up with the
09:15–15:30 session directly.

---

## Storing it on Drive and using it later

The zip is self-describing — nothing outside it is needed to read it.

```bash
# after the pull
ls -lh data/snapshots/*.zip      # upload this file to Drive
```

To use it in a later session, put it anywhere and point the loader at it:

```python
from afml_india.research import Snapshot

snap = Snapshot("/path/to/nifty_index_20160822.zip")
print(snap)                       # index, as-of, member count, synthetic flag
print(snap.manifest)              # full provenance including the tick window
```

Or drop it in `data/snapshots/` and let the notebooks find it themselves:

```python
from afml_india.research import Snapshot, find_snapshot
snap = Snapshot(find_snapshot())  # most recently modified zip in data/snapshots/
```

`data/snapshots/` is gitignored, so a large zip will not be committed by accident.

### Reading from Google Drive

In Colab, mount and point straight at it — `Snapshot` reads inside the zip without
unpacking:

```python
from google.colab import drive
drive.mount("/content/drive")

from afml_india.research import Snapshot
snap = Snapshot("/content/drive/MyDrive/ml-trading/nifty_index_20160822.zip")
```

Locally, `rclone copy` or the Drive desktop client both work; there is nothing
Drive-specific in the code.

---

## Verifying a snapshot before you trust it

```python
from afml_india.research import Snapshot

snap = Snapshot("nifty_index_20160822.zip")
print(snap.manifest["tick_window"])          # which sessions the ticks actually cover
print(snap.members.head())                   # point-in-time composition
print(snap.report.groupby("status").size())  # how many requests came back empty
print(snap.tickers)                          # what is actually present
print(snap.tick_days(snap.tickers[0]))       # sessions available for one name
```

`snap.report` is the one to read first. A large `empty` count means entitlements or the
140-day window bit, not that the market was quiet.

---

## Reproducing the exact notebook inputs

```bash
# 1. Confirm the universe
python scripts/fetch_bloomberg_snapshot.py --list-members --asof 2016-08-22

# 2. Find this terminal's real tick retention
python scripts/fetch_bloomberg_snapshot.py --probe --asof 2016-08-22 --max-members 5

# 3. Trial run
python scripts/fetch_bloomberg_snapshot.py --asof 2016-08-22 --max-members 3 --tick-days 5 -v

# 4. Full pull, with --tick-days set from the probe
python scripts/fetch_bloomberg_snapshot.py --asof 2016-08-22 --tick-days 120

# 5. Run the notebooks against it
jupyter lab notebooks/
```

The notebooks call `find_snapshot()`, so they pick up whatever zip is newest in
`data/snapshots/` — no edits needed when you swap the synthetic snapshot for a real one.
