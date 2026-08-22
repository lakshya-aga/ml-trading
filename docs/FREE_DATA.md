# Free intraday data for Indian markets

> **Verification note.** I could not test any of these sources from the
> environment this was written in — outbound access to every market-data host
> was blocked by egress policy. The **ingestion path is tested end to end**
> (`scripts/build_snapshot_from_bars.py`, and the pipeline running on imported
> minute data), but the source-by-source claims below are from knowledge, not
> from a live check. Treat depths and limits as starting hypotheses and verify
> on your own machine — every section says how.

---

## Short answer

**Yahoo Finance, right now, no account:** one-minute data for the last 30 days,
5/15/30-minute for 60 days, hourly for two years. Free, no key, works today.

**A free broker API, after a day of setup:** one-minute data going back a year
or more, which is the real answer if you need depth.

**Neither gives you ten years of minute data for free.** That does not exist for
Indian equities from a reputable source. The honest ceiling on free is roughly a
year or two of one-minute data via a broker API.

---

## Tier 1 — Yahoo Finance (zero friction)

No account, no key, one `pip install`.

```bash
pip install yfinance
python -m afml_india.data.free_sources --check RELIANCE
```

That last command measures what Yahoo actually serves for a symbol and prints
it. **Trust its output over the table below** — Yahoo changes these limits
without notice and they differ by exchange.

| Interval | Per request | Total history |
|---|---|---|
| 1m | 7 days | ~30 days |
| 5m / 15m / 30m | 60 days | ~60 days |
| 1h | 730 days | ~730 days |
| 1d | — | decades |

### The trap that makes people think it does not work

Asking for 30 days of one-minute data in a single call returns an **empty
frame**, not an error. Yahoo caps a 1m request at seven days. This is the usual
reason for "Yahoo has no minute data for India" — it does; it just has to be
walked in chunks:

```python
from afml_india.data.free_sources import yahoo_intraday

# Handles the chunking for you
minute = yahoo_intraday("RELIANCE", interval="1m")          # ~30 days
five_min = yahoo_intraday("RELIANCE", interval="5m")        # ~60 days
```

### Quality caveats, in the order they will bite

- **Volume is the weak field.** Yahoo's Indian volume figures are frequently
  wrong or zero on less liquid names. That matters more here than usual, because
  volume and rupee bars are thresholded on it. Sanity-check volume against a
  broker chart before trusting bar counts.
- **Gaps.** Missing minutes are common, especially away from the large caps.
- **Adjustment.** `auto_adjust` behaviour on Indian tickers is inconsistent.
  Verify against a known split — `clean_ohlcv(max_return=0.5)` in this package
  will flag an unadjusted one.
- **Mid- and small-caps are much worse than large-caps.** Test on the name you
  actually care about, not on Reliance.

Good enough to develop the pipeline. Not good enough to publish a result.

---

## Tier 2 — free broker APIs (the real answer for depth)

Several Indian brokers give historical intraday candles free with a demat
account and a developer registration. This is where a year or more of
one-minute data comes from without paying for it.

| Broker | Historical intraday | Notes |
|---|---|---|
| **Upstox** | free with the API | commonly cited as the most generous free historical window |
| **Angel One (SmartAPI)** | free | per-request day caps; loop to go back further |
| **Dhan** | free | |
| **Fyers** | free | |
| **Finvasia (Shoonya)** | free | zero-brokerage account |
| **Zerodha (Kite Connect)** | **paid** | historical data is a paid add-on — do not assume it is free because Zerodha is popular |

**What I am confident about:** these brokers expose historical candle endpoints,
registration is free, and the free tiers are genuinely usable. **What I am not
confident about:** the exact lookback each one grants today, and per-request
limits. Both change, and I could not check.

So verify before you build against one:

1. Open a free account and register a developer app.
2. Pull one symbol at one-minute resolution for the oldest date you care about.
3. Check the response is non-empty and the row count matches a full session
   (375 one-minute bars for NSE equities).
4. Repeat for an illiquid name — that is where free feeds fall apart.

Whatever it returns, dump it to CSV and use the ingestion path below. Nothing
downstream depends on which broker you chose.

---

## Tier 3 — bulk datasets

Kaggle and GitHub host NIFTY / NSE minute datasets, often several years deep.
Free, instant, and no API to wrestle with.

Three things to check before using one for anything but a smoke test:

- **Corporate actions.** Most are unadjusted. A 1:1 bonus reads as a −50%
  return and ruins every volatility estimate downstream.
- **Survivorship.** Almost all cover currently-listed names only, which makes
  any backtest on them optimistic by construction.
- **Provenance.** Frequently no statement of where the data came from or how it
  was cleaned.

Fine for developing code. Not a basis for a result you would defend.

---

## Tier 4 — record it yourself

Any free broker websocket will stream live ticks. Recording costs a small VPS
and nothing else, and a year from now the recorded year is yours at full
resolution with no licence attached.

Useless for the ten years you want today. Worth starting anyway, precisely
because it is the one source that only gets better.

---

## Getting any of it into this project

One script normalises all of the above into the same snapshot the notebooks
read, so the choice of source stays a detail:

```bash
# A folder of per-symbol CSVs (Kaggle layout, broker export, anything)
python scripts/build_snapshot_from_bars.py data/raw/minute --out data/snapshots

# One long file with a symbol column
python scripts/build_snapshot_from_bars.py all_minute.csv --long --out data/snapshots

# Straight from Yahoo
python scripts/build_snapshot_from_bars.py --yahoo RELIANCE,INFY,TCS --interval 5m
```

Column names are auto-detected across the usual spellings — `Date`+`Time`
pairs, `o/h/l/c/v`, `datetime`, `instrument`/`ticker`/`symbol`. Anything
unresolvable is reported rather than guessed.

Then everything works as before:

```python
from afml_india.research import Snapshot, build_all_bars
from afml_india.pipeline import PipelineConfig, TickerPipeline

snap = Snapshot("data/snapshots/imported_20260822.zip")
ticks = snap.pseudo_ticks("RELIANCE")          # bars reshaped -- see below
bars = build_all_bars(ticks, bars_per_day=25)
```

### What `pseudo_ticks` is, and what it is not

It collapses each bar to one synthetic transaction at its typical price carrying
the bar's whole volume. That is an **approximation, not a tape**:

- Within-bar path information is gone — the sequencing that tick and volume bars
  exist to capture in the first place.
- The source resolution is a hard floor. One-minute input cannot produce a bar
  finer than one minute, however low you set the threshold.
- Order-flow features degrade. `cum_buy_volume` from the tick rule is meaningless
  when there is one synthetic trade per bar.

It is still worth doing — resampling minute bars into rupee-value bars recovers
much of the statistical benefit over a fixed clock, which is the main claim of
AFML chapter 2. Verified on imported minute data in this repo: rupee bars from
one-minute input still track session turnover and still improve on a matched
time-bar baseline.

Just do not describe the result as tick-based. It is bar-based, at the
resolution of whatever you fed it.

---

## What I would actually do

1. **Today:** `yahoo_intraday(interval="5m")` across 20 NIFTY names. Sixty days,
   zero setup, enough to run every notebook and find the bugs in your own
   configuration.
2. **This week:** register for one free broker API and pull a year of
   one-minute data for the same names. That is a genuinely usable research
   dataset and it costs nothing.
3. **Only then** decide whether tick resolution is the binding constraint. If a
   signal is invisible in a year of minute bars across 20 names, tick data
   rarely rescues it — and by then you will know exactly what to buy.
