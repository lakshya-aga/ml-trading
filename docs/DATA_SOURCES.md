# Resolution limits, and where deep tick history actually comes from

Two questions this answers:

1. What is the finest resolution `blpapi` will give me, and how far back?
2. If I need ten years of Indian tick data, where do I get it?

---

## 1. What blpapi will actually give you

Three request types matter, and they trade resolution against history:

| Request | Finest resolution | How far back | Use |
|---|---|---|---|
| `IntradayTickRequest` | every trade and quote | **~140 days** | true tick bars |
| `IntradayBarRequest` | **1 minute** | **~140 days** | the practical fallback |
| `HistoricalDataRequest` | **1 day** | full history (decades) | everything older |

So the answer to "what is the max resolution beyond 140 days" is blunt:

> **Daily.** There is no intraday product on the Desktop API that reaches back
> further. `IntradayBarRequest` carries the same retention cliff as tick data —
> going from ticks to one-minute bars buys granularity, not history.

Inside the ~140-day window, one-minute bars are the sensible fallback when tick
data is not entitled. They are far more widely entitled, and they still resample
into respectable rupee-value bars — you lose the true trade tape, not the method.

```bash
python scripts/fetch_bloomberg_snapshot.py --tick-days 120 --intraday-bars 1 --skip-ticks
```

### Measure your own limits rather than trusting the number

The 140 is approximate and terminal-dependent. `--probe` bisects for the oldest
session that actually returns data, separately for ticks and for one-minute
bars:

```bash
python scripts/fetch_bloomberg_snapshot.py --probe --max-members 5
```

```
ticker            has_trade_tape  oldest       retention_days  bars_oldest  bars_retention_days
RIL IN Equity     True            2026-04-05   139             2026-04-05   139
NIFTY Index       False           None         0               None         0
```

`has_trade_tape=False` is a different failure from a retention limit, and the
probe keeps them apart. An index returns nothing because it is a computed level
with no trades in it — you want its constituents or its futures
(`N50FUTPR Index`).

### An important qualification about Bloomberg

"Bloomberg does not have ten-year tick data" is **not true** — the accurate
statement is that the **Desktop API does not serve it**.

Bloomberg does sell deep historical tick through separate products, on separate
contracts, with separate delivery mechanisms — a Data License / B-PIPE
historical tick offering rather than `//blp/refdata` on your terminal. If your
institution already pays for Bloomberg, ask your representative about historical
tick under Data License before buying elsewhere. I can't tell you the pricing,
the exact product name in force today, or your entitlements — that is a
conversation with your rep, and it is the first call worth making.

### What you can still get for the full ten years

`HistoricalDataRequest` returns daily fields going back decades, and several of
them are intraday-derived summaries that carry real information:

| Field | What it gives you |
|---|---|
| `PX_OPEN` / `PX_HIGH` / `PX_LOW` / `PX_LAST` | the daily range — enough for Parkinson and Garman-Klass volatility |
| `PX_VOLUME`, `TURNOVER` | daily rupee turnover, which is what rupee bars are built on |
| `EQY_WEIGHTED_AVG_PX` | the session VWAP |
| `CUR_MKT_CAP` | size, for cross-sectional work |

That is what the current pull already collects, and it is enough to run
notebook 03 (monthly forecasting) over a decade. It is **not** enough for
notebooks 01, 02 and 04 at bar frequency, which is exactly why the snapshot
mixes horizons.

---

For **free** minute data — Yahoo, free broker APIs, bulk datasets — see
**[FREE_DATA.md](FREE_DATA.md)**. This document covers paid depth.

## 2. Where to get deep Indian tick history

Ordered by how likely they are to be the right answer for this project.

### NSE itself — the authoritative source

The exchange sells its own historical data, and for Indian equities it is the
primary source rather than a redistributor. Products have historically included
trade-by-trade capital-market data and order-level ("order log") data going back
many years, licensed per segment and per period.

This is where to start: it is the tape everyone else is reselling, the coverage
of NSE-listed names is by definition complete, and academic and research
licences have historically been cheaper than commercial ones. Contact NSE's data
services arm directly for current products and pricing.

**Caveat worth budgeting for:** exchange tick archives arrive as large
compressed per-day files in the exchange's own format, not as tidy CSV. Expect
to write a parser and expect the raw archive to be measured in terabytes for a
decade of the full market.

### LSEG / Refinitiv Tick History (formerly TRTH)

Global consolidated tick history with very deep coverage, including Indian
venues, delivered via a REST API and bulk extraction rather than a desktop
terminal. This is the usual institutional answer when you need many years of
tick data across markets and want it already normalised.

Expensive, and priced per extraction volume — scope the exact instruments and
window before requesting a quote.

### Indian retail and prop data vendors

Several vendors serve the Indian algo-trading market with historical intraday
data at lower cost and much lower friction than an exchange licence. Names that
come up repeatedly in this space include **GDFL (Global Datafeeds)**,
**TrueData**, and **FirstRate Data**. Typical offerings are one-minute or
tick-level history for NSE cash and F&O, sold per segment per year.

Two things to verify before buying from any of them, because they are the usual
failure modes:

- **Corporate-action adjustment.** Ask explicitly whether prices are
  back-adjusted for splits and bonuses, and whether an unadjusted series is also
  provided. An unadjusted 1:1 bonus shows up as a −50% return and will wreck
  every volatility estimate downstream.
- **Survivorship.** Ask whether delisted and demoted names are included. A
  vendor who only carries currently-listed symbols cannot support an unbiased
  backtest, whatever the resolution.

Ask for a free sample of one liquid and one illiquid name before committing —
the illiquid one is where data quality problems actually show up.

### Broker APIs — mostly not the answer

Zerodha Kite, Upstox, Angel One and others expose historical intraday
candles through their APIs, typically with a limited lookback and rate limits,
and generally not tick data. Useful for recent history and for live work; not a
route to a decade of ticks.

### Building your own going forward

If the research horizon is long, recording your own tape from a live feed costs
almost nothing and compounds. It does not help with the ten years you want
today, but a year from now the recorded year is yours, at full resolution,
with no licence attached.

---

## 3. What I would actually do here

Given the framework already handles all three resolutions, in order of effort:

**Start with daily.** A decade of daily data is already in the snapshot, is free
with your existing terminal, and is enough to run the monthly forecasting work
in notebook 03 over ten years. Get a result there before paying for anything.

**Add the recent 140-day intraday window.** That is enough to develop and debug
the bar-based pipeline in notebooks 01, 02 and 04 — several thousand rupee bars
per name, which is a workable dataset for methodology even if it is thin for
inference.

**Then decide whether depth is the binding constraint.** It often is not. If the
signal is not visible in 120 sessions of tick data across 40 names, ten years of
it is unlikely to rescue it, and the money is better spent on cross-section
(more names) than on history.

**If you do buy, buy the narrowest thing that answers the question.** Two years
of tick data on the twenty most liquid NIFTY names is a fraction of the cost of
a decade of the full market, and for a bar-construction study it is
indistinguishable in value.

---

## 4. Where the framework already meets you

Nothing above requires code changes; the pipeline is resolution-agnostic by
design.

```python
from afml_india.research import Snapshot, build_bars, bars_from_ohlcv

# True ticks, when you have them
bars = build_bars(snap.ticks("RIL_IN", "trades"), kind="dollar", bars_per_day=50)

# One-minute bars from --intraday-bars, treated as a coarse tape
bars = bars_from_ohlcv(minute_bars, kind="value", target_bars_per_day=20)

# Daily data over ten years
bars = bars_from_ohlcv(snap.daily("RIL_IN"), kind="value", target_bars_per_day=1)
```

`bars_from_ohlcv` represents each source bar by its typical price and treats it
as one tick. That is a real approximation and it degrades as the source bars get
coarser — daily bars into "rupee bars" is little more than a relabelled daily
series. It is honest about the trade-off rather than pretending resolution it
does not have.
