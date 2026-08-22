"""AFML research framework for Indian equities (NSE/BSE).

Layering
--------
``afml_india`` deliberately does *not* reimplement Advances in Financial Machine
Learning. The algorithms live in the ``fin-kit`` submodule under
``external/fin-kit`` (importable as ``mlfinlab``); this package supplies what
fin-kit does not have:

* Indian market conventions — NSE/BSE calendar, tick and lot sizes, circuit
  bands, and the statutory cost stack (STT, stamp duty, exchange charges, GST).
* Data access — Bloomberg point-in-time snapshots, vendor CSV/Parquet, Yahoo.
* Glue — the handful of functions that sit between the two and are easy to get
  wrong, such as CUSUM sampling on log prices and session-aware vertical
  barriers.

Start here::

    from afml_india.research import *

    snap = Snapshot(find_snapshot())
    bars = build_all_bars(snap.ticks(snap.most_liquid(1)[0]))
    events = sample_events(bars["dollar"]["close"], vol_multiple=2.0)
"""

from afml_india.utils.logging import get_logger

__version__ = "0.1.0"
__all__ = ["__version__", "get_logger"]
