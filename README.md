# ml-trading — AFML for Indian equities

Advances in Financial Machine Learning (López de Prado) applied to the NSE/BSE cash
segment, with the market-structure details that make the difference between a method
and a working method: session-aware barriers, the Indian statutory cost stack,
point-in-time index membership, and tick/lot conventions.

The AFML algorithms themselves are **not reimplemented here**. They come from
[`fin-kit`](https://github.com/lakshya-aga/fin-kit) (`mlfinlab`), vendored as a
submodule. This repository supplies the layer around it: data acquisition, Indian
market conventions, feature construction, models, and the glue between the two that is
easy to get subtly wrong.

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/lakshya-aga/ml-trading
cd ml-trading

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"

# No Bloomberg terminal? Generate a synthetic snapshot with the real schema,
# so every notebook runs end to end.
python scripts/fetch_bloomberg_snapshot.py --offline-demo --members 5 --tick-days 90

jupyter lab notebooks/
```

Already cloned without submodules: `git submodule update --init --recursive`.

`afml_india` puts the submodule on `sys.path` automatically, so no separate install of
`mlfinlab` is needed.

---

## Notebooks

Run in order; each builds on the last.

| | Notebook | What it establishes |
|---|---|---|
| 01 | `01_bars.ipynb` | Tick, volume and rupee bars from a trade tape, against a time-bar baseline. Measures the distributional improvement rather than asserting it. |
| 02 | `02_event_sampling_and_fracdiff.ipynb` | CUSUM event sampling, fractional differencing, triple-barrier labels, sample weights. Ends with a leak-checked feature matrix. |
| 03 | `03_lstm_fracdiff_comparator.ipynb` | LSTM held fixed, input representation varied (levels / returns / fracdiff). One-step-ahead forecasts, PnL, Sharpe, per-stock attribution, `baseline + selection − costs`. |
| 04 | `04_per_ticker_framework.ipynb` | Classification with triple-barrier labels and purged walk-forward CV. LSTM against baselines, then swept across the point-in-time universe. |
| **05** | **`05_cross_sectional_significance.ipynb`** | **The core study.** Fracdiff + LSTM on every ticker, identical configuration, out-of-sample event PnL net of Indian costs — then one number: a permutation p-value for the whole cross-section against a skill-free null. Sharpe, turnover and return metrics per ticker; robustness across cost regimes; a built-in calibration on GBM universes. |

All five are committed with outputs, produced from the synthetic snapshot. **Every
number in them is a property of a toy generator** — the notebooks say so where it
matters. Point notebook 05 at the real 140-day pull and its cells become the
project's headline result, whichever way it lands.

---

## Getting real data

See **[docs/DATA.md](docs/DATA.md)**. In short:

```bash
# What will I get?
python scripts/fetch_bloomberg_snapshot.py --list-members --asof 2016-08-22

# What will this terminal actually return? (tick retention, per security)
python scripts/fetch_bloomberg_snapshot.py --probe --max-members 5

# The pull
python scripts/fetch_bloomberg_snapshot.py --asof 2016-08-22 --tick-days 120
```

One zip comes out, self-describing, ready to put on Drive:

```python
from afml_india.research import Snapshot
snap = Snapshot("/path/to/nifty_index_20160822.zip")
```

A pull takes hours; watch it in flight without waiting for the zip:

```bash
python scripts/snapshot_status.py --watch 30
```

### One constraint worth knowing up front

Bloomberg retains roughly **140 days** of intraday tick history. Tick data from ten
years ago is not obtainable over `blpapi` — confirmed on a live terminal in
`bloomberg_test.ipynb`, where an August 2020 request returns an empty `tickData` array.

So a snapshot mixes horizons on purpose: a point-in-time member list from the as-of
date, ten years of daily bars, and recent ticks for those same members. The manifest
records it.

Beyond that window the maximum resolution `blpapi` offers is **daily** —
`IntradayBarRequest` hits the same cliff. [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
covers the limit per request type and where deep Indian tick history is actually sold.

### No Bloomberg? Free intraday data works too

```bash
pip install yfinance
python -m afml_india.data.free_sources --check RELIANCE      # what Yahoo serves
python scripts/build_snapshot_from_bars.py --yahoo RELIANCE,INFY,TCS --interval 5m
```

`build_snapshot_from_bars.py` also ingests any folder of minute CSVs — a Kaggle
dump, a broker export — into the same snapshot the notebooks read, auto-detecting
the usual column spellings. See [docs/FREE_DATA.md](docs/FREE_DATA.md) for what each
free source actually gives you, and for the honest limits of building bars from bars
rather than from a tape.

---

## Using the library

```python
from afml_india.research import *

snap = Snapshot(find_snapshot())
ticks = snap.ticks(snap.most_liquid(1)[0], kind="trades")

bars   = build_bars(ticks, kind="dollar", bars_per_day=50)   # AFML ch. 2
d      = min_ffd_order(bars["close"])["d"]                   # ch. 5
events = sample_events(bars["close"], vol_multiple=1.0)      # ch. 2
labels = triple_barrier_labels(bars["close"], events)        # ch. 3, NSE sessions
```

Or run the whole chain for one ticker and then across the universe:

```python
from afml_india.pipeline import PipelineConfig, TickerPipeline, run_universe
from afml_india.models import LSTMClassifier, random_forest

config = PipelineConfig(bar_kind="dollar", bars_per_day=50, holding_days=2)

result = TickerPipeline(config).run("RIL_IN", make_model=random_forest, snapshot=snap)
print(result.fold_metrics)

table, results = run_universe(snap, make_model=random_forest, config=config)
```

---

## What this adds to fin-kit

Four things fin-kit does not have, each of which is a place the naive version is wrong.

**The NSE calendar is wired into the barriers.** `add_vertical_barrier` counts calendar
days, which silently shortens the holding period across Diwali or a long weekend.
`triple_barrier_labels` counts trading sessions.

**CUSUM runs on log prices.** `cusum_filter` accumulates the first difference of
whatever it is given. Hand it rupee prices and a return-scale threshold is breached
within a bar or two, so effectively every bar becomes an event. `sample_events`
applies the log transform and scales the threshold by trailing volatility.

**Timezones survive the round trip.** Several `mlfinlab` functions push a
`DatetimeIndex` through `.values` and lose the timezone; `get_daily_vol` fails outright
against tz-aware input under pandas 3. `afml_india._tz` contains the conversion in one
place instead of forcing naive timestamps on an IST market.

**Costs are Indian and asymmetric.** STT applies to both legs on delivery and only the
sell on intraday; stamp duty is buy-side; GST compounds on brokerage plus exchange
charges. A round trip is roughly 23 bps on delivery and 4.5 bps intraday — the
difference decides whether a bar-frequency signal is viable at all.

---

## Package layout

```
src/afml_india/
├── data/
│   ├── calendar.py      NSE/BSE sessions, holidays, session-aware barrier shifts
│   ├── costs.py         STT/stamp/exchange/GST by segment, ticks, lots, circuits, impact
│   ├── loaders.py       vendor CSV/Parquet/Yahoo, validation, corporate actions
│   ├── universe.py      dated index membership with survivorship warnings
│   └── snapshot.py      reader for the Bloomberg snapshot zip
├── bars/                OHLCV bars, plus adapters onto fin-kit's tick bars
├── features.py          FeatureConfig recipe and a replay-based look-ahead check
├── models/
│   ├── sequences.py     LSTM windowing; sequence_span for correct purging
│   ├── lstm.py          torch LSTM with sample weights in the loss
│   ├── baselines.py     forest and logistic on the same splits
│   ├── evaluate.py      purged, embargoed walk-forward CV
│   └── forecast.py      one-step and recursive forecasting; exact fracdiff inversion
├── portfolio.py         Roy ratio, signal backtests, attribution, Sharpe variants
├── pipeline.py          TickerPipeline and run_universe
└── research.py          one import for a research session
```

---

## Design notes

**Look-ahead is checked, not asserted.** `assert_no_lookahead` rebuilds the feature
matrix on truncated history at random cut points and verifies two things: that no value
changes, and that the most recent row is still *available*. The second catches centred
rolling windows, which produce identical values for surviving rows and would pass a
value-only check.

**Purging covers the input window too.** An LSTM sample spans backwards as well as
forwards. `sequence_span` reports the full interval so `WalkForwardSplit` purges both
directions.

**Sample weights reach the optimiser.** AFML uniqueness weights only matter if the loss
sees them. `LSTMClassifier.fit` takes `sample_weight` and applies it per sample.

**Effective sample size is reported.** On one ticker at 1:1 barriers, 508 rows carry
roughly 138 independent observations. That number, not the row count, is what the model
has to learn from.

**The significance test is calibrated, and its first version was wrong.** The
cross-sectional permutation test originally shuffled events independently and returned
p = 0.001 on a random walk — overlapping events are not exchangeable, and the iid null
understated its own variance by 2×. The investigation that found this (tape
autocorrelation ≈ 0, plain-sklearn control, clean-GBM control) is documented in
notebook 05, the fix is a circular-shift null that preserves both sequences' serial
structure, and `scripts/calibrate_significance.py` re-checks calibration on skill-free
GBM universes whenever the pipeline changes. The demo generator itself turned out to
carry a small learnable artifact — found by the test, isolated by the calibration.

---

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip the model fits
```

Covers calendar arithmetic across real NSE holidays, the cost model's asymmetries,
bar-threshold behaviour, the fin-kit adapters (including the timezone regression), the
look-ahead guarantee, and the significance layer. The suite leans on negative
controls: a planted centred-window leak the look-ahead check must catch, a shuffled
label the purged CV must fail to recover, a skill-free oracle/noise pair the
permutation test must separate, and block-correlated events on which the iid-shuffle
null must be visibly narrower than the rotation null — the regression that motivated
the default.

---

## Status and honest limits

- **Committed notebook outputs are synthetic.** Structure is real; conclusions are not.
- **The bundled NIFTY 50 list is current-composition** and therefore
  survivorship-biased. Resolve point-in-time membership from Bloomberg for research;
  `data/reference/nifty50_bloomberg_tickers.csv` is a fallback, and its Bloomberg
  tickers are best-effort rather than machine-verified.
- **NSE holidays are compiled from published circulars** and should be verified against
  the exchange notice for settlement-sensitive work. Override with
  `NSECalendar(holidays=...)` or `AFML_NSE_HOLIDAYS`.
- **No live execution layer.** Costs and impact are modelled; nothing here places
  orders.

## Licence

MIT. `fin-kit`/`mlfinlab` carries its own licence — see the submodule.
