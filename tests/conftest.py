"""Shared fixtures. Synthetic by construction so the suite needs no data files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20240822)


@pytest.fixture(scope="session")
def tick_frame() -> pd.DataFrame:
    """A synthetic NSE trade tape across three sessions, IST-aware."""
    generator = np.random.default_rng(7)
    frames = []
    level = 1000.0
    # Sessions differ in activity on purpose: a tape with identical volume every
    # day cannot distinguish an activity-driven bar from a calendar one.
    sessions = [
        ("2024-01-02", 3000, 200),
        ("2024-01-03", 6000, 500),
        ("2024-01-04", 4500, 350),
        ("2024-01-05", 9000, 800),
        ("2024-01-08", 2500, 150),
    ]
    for day, n, max_size in sessions:
        opening = pd.Timestamp(day, tz="Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
        offsets = np.sort(generator.uniform(0, 375 * 60, n))
        price = level * np.exp(np.cumsum(generator.normal(0, 3e-4, n)))
        price = np.round(price / 0.05) * 0.05
        level = float(price[-1])
        frames.append(
            pd.DataFrame(
                {
                    "price": price,
                    "volume": generator.integers(1, max_size, n),
                    "type": "TRADE",
                },
                index=opening + pd.to_timedelta(offsets, unit="s"),
            )
        )
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """Daily OHLCV with a realistic high/low envelope."""
    generator = np.random.default_rng(11)
    index = pd.date_range("2022-01-03", periods=500, freq="B", tz="Asia/Kolkata")
    close = 500 * np.exp(np.cumsum(generator.normal(0.0003, 0.014, len(index))))
    spread = np.abs(generator.normal(0, 0.008, len(index)))
    return pd.DataFrame(
        {
            "open": close * (1 + generator.normal(0, 0.003, len(index))),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": generator.lognormal(12, 0.5, len(index)).round(),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def price_series() -> pd.Series:
    """A long random-walk price series for stationarity tests."""
    generator = np.random.default_rng(3)
    index = pd.date_range("2020-01-01", periods=1200, freq="B")
    return pd.Series(
        100 * np.exp(np.cumsum(generator.normal(0.0002, 0.011, len(index)))),
        index=index,
        name="close",
    )
