# Reference ticker data

## `nifty50_bloomberg_tickers.csv`

NSE symbol to Bloomberg ticker mapping for a **current-composition** NIFTY 50, with
sector labels.

Two warnings before you use it as a universe.

**It is not point-in-time.** This is today's index membership. Applying it to ten-year-old
data is survivorship-biased: it silently omits every name that was in the index in 2016
and has since been demoted or delisted, which are exactly the losers a backtest needs to
see. The download script does not use this file as its universe by default — it resolves
membership from Bloomberg's `INDX_MWEIGHT_HIST` for the as-of date you pass, which is the
authoritative point-in-time source.

**The Bloomberg tickers are best-effort.** They were compiled from published mappings and
are not machine-verified. Bloomberg tickers are not derived from NSE symbols by any rule
(`INFY` is `INFO IN`, `HDFCBANK` is `HDFCB IN`, `M&M` is `MM IN`), and they change on
corporate actions. Verify against the terminal before relying on them.

Use this file for:

- a fallback universe when `INDX_MWEIGHT_HIST` is not entitled on your terminal
  (`--tickers-file data/reference/nifty50_bloomberg_tickers.csv`)
- sector labels to group results
- translating between NSE symbols and Bloomberg tickers in your own code

Resolve the real point-in-time universe with:

```bash
python scripts/fetch_bloomberg_snapshot.py --list-members --asof 2016-08-22
```
