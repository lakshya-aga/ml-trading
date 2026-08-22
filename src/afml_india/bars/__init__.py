"""Bar construction: a dependency-light OHLCV path and the fin-kit tick path."""

from afml_india.bars.standard import (
    bars_from_ohlcv,
    dollar_bars,
    suggest_threshold,
    tick_bars,
    time_bars,
    value_bars,
    volume_bars,
)

__all__ = [
    "tick_bars",
    "volume_bars",
    "value_bars",
    "dollar_bars",
    "time_bars",
    "bars_from_ohlcv",
    "suggest_threshold",
]


def __getattr__(name: str):
    """Expose the fin-kit adapters lazily so importing this package never needs mlfinlab."""
    if name in ("build_bars", "build_all_bars", "prepare_tick_frame",
                "suggest_thresholds", "source_timezone"):
        from afml_india.bars import finkit_bars  # noqa: PLC0415

        return getattr(finkit_bars, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
