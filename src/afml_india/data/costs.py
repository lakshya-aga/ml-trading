"""Transaction costs for Indian equity trading.

Indian statutory charges are asymmetric (STT on delivery hits both legs, on
intraday only the sell) and several are levied on turnover rather than on
notional net of slippage, so a flat "10 bps round trip" understates the cost of
a high-turnover intraday strategy and overstates a low-turnover delivery one.
The defaults below reflect the published rate card for the NSE cash segment;
they are configuration, not constants — check them against your broker's
contract note and pass your own :class:`CostModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Segment(str, Enum):
    """Settlement type, which drives which statutory charges apply."""

    DELIVERY = "delivery"
    INTRADAY = "intraday"
    FUTURES = "futures"


@dataclass(frozen=True)
class CostModel:
    """Per-leg cost breakdown for the NSE/BSE cash and futures segments.

    All rates are fractions of turnover unless the field name says otherwise.
    Turnover means price times quantity for the leg being charged.
    """

    segment: Segment = Segment.DELIVERY

    #: Securities Transaction Tax. Delivery: both legs. Intraday/futures: sell only.
    stt_delivery: float = 0.001
    stt_intraday_sell: float = 0.00025
    stt_futures_sell: float = 0.0002

    #: NSE cash-segment transaction charge; futures are cheaper.
    exchange_txn_cash: float = 0.0000297
    exchange_txn_futures: float = 0.0000173

    #: SEBI turnover fee (Rs 10 per crore).
    sebi_turnover: float = 0.000001

    #: Stamp duty, buy side only. 0.015% delivery, 0.003% intraday, 0.002% futures.
    stamp_delivery: float = 0.00015
    stamp_intraday: float = 0.00003
    stamp_futures: float = 0.00002

    #: GST on (brokerage + exchange transaction charge + SEBI fee).
    gst: float = 0.18

    #: Brokerage as a fraction of turnover, capped in rupees per order.
    brokerage_rate: float = 0.0003
    brokerage_cap: float = 20.0

    #: Investor protection fund / clearing charges, folded into one turnover rate.
    clearing: float = 0.0000010

    #: Half-spread paid on aggressive execution, in fractions of price.
    half_spread: float = 0.00025

    #: Market-impact coefficient in the square-root law (see ``impact_bps``).
    impact_coefficient: float = 0.10

    def _exchange_txn(self) -> float:
        return (
            self.exchange_txn_futures if self.segment is Segment.FUTURES else self.exchange_txn_cash
        )

    def _stamp(self) -> float:
        return {
            Segment.DELIVERY: self.stamp_delivery,
            Segment.INTRADAY: self.stamp_intraday,
            Segment.FUTURES: self.stamp_futures,
        }[self.segment]

    def _stt(self, side: int) -> float:
        """STT rate for a leg. ``side`` is +1 for a buy and -1 for a sell."""
        if self.segment is Segment.DELIVERY:
            return self.stt_delivery
        if side > 0:
            return 0.0
        return self.stt_futures_sell if self.segment is Segment.FUTURES else self.stt_intraday_sell

    def brokerage(self, turnover: float | np.ndarray) -> float | np.ndarray:
        """Brokerage on ``turnover``, applying the per-order rupee cap."""
        return np.minimum(
            np.asarray(turnover, dtype=float) * self.brokerage_rate, self.brokerage_cap
        )

    def leg_cost(
        self,
        price: float | np.ndarray,
        quantity: float | np.ndarray,
        side: int,
    ) -> np.ndarray:
        """Rupee cost of one leg, excluding spread and impact.

        Parameters
        ----------
        price, quantity:
            Execution price and absolute share count.
        side:
            ``+1`` for a buy, ``-1`` for a sell.
        """
        turnover = np.abs(np.asarray(price, dtype=float) * np.asarray(quantity, dtype=float))
        brok = self.brokerage(turnover)
        exch = turnover * self._exchange_txn()
        sebi = turnover * self.sebi_turnover
        gst = (brok + exch + sebi) * self.gst
        stt = turnover * self._stt(side)
        stamp = turnover * self._stamp() if side > 0 else 0.0
        clearing = turnover * self.clearing
        return brok + exch + sebi + gst + stt + stamp + clearing

    def round_trip_bps(self, price: float = 1000.0, quantity: float = 100.0) -> float:
        """Round-trip statutory + brokerage cost in basis points of notional.

        Useful as a sanity check on whether a signal's edge survives costs at
        all before running a full backtest.
        """
        notional = price * quantity
        total = self.leg_cost(price, quantity, +1) + self.leg_cost(price, quantity, -1)
        return float(total / notional * 1e4)

    def impact_bps(
        self,
        order_value: float | np.ndarray,
        adv_value: float | np.ndarray,
        volatility: float | np.ndarray,
    ) -> np.ndarray:
        """Square-root market impact in basis points.

        ``impact = coefficient * sigma * sqrt(order_value / adv_value)``, the
        standard concave law. Mid- and small-cap Indian names have thin ADV, so
        this term dominates statutory cost well below institutional size.
        """
        order_value = np.abs(np.asarray(order_value, dtype=float))
        adv_value = np.asarray(adv_value, dtype=float)
        volatility = np.asarray(volatility, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(adv_value > 0, order_value / adv_value, np.nan)
        impact = self.impact_coefficient * volatility * np.sqrt(participation)
        return np.nan_to_num(impact, nan=0.0) * 1e4

    def total_cost(
        self,
        price: float | np.ndarray,
        quantity: float | np.ndarray,
        side: int,
        adv_value: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Rupee cost of one leg including half-spread and optional impact."""
        turnover = np.abs(np.asarray(price, dtype=float) * np.asarray(quantity, dtype=float))
        cost = self.leg_cost(price, quantity, side) + turnover * self.half_spread
        if adv_value is not None and volatility is not None:
            cost = cost + turnover * self.impact_bps(turnover, adv_value, volatility) / 1e4
        return cost


@dataclass(frozen=True)
class InstrumentSpec:
    """Exchange conventions for a single instrument.

    Rounding to the tick and the lot is not cosmetic: an unrounded backtest
    quietly assumes fills at prices the exchange will not accept, and the error
    is largest exactly where it matters, in low-priced small caps.
    """

    symbol: str
    tick_size: float = 0.05
    lot_size: int = 1
    circuit_limit: float | None = 0.20
    segment: Segment = Segment.DELIVERY
    #: Optional per-instrument overrides applied on top of the portfolio model.
    cost_overrides: dict = field(default_factory=dict)

    def round_price(self, price: float | np.ndarray, side: int = 0) -> np.ndarray:
        """Snap ``price`` to the instrument tick.

        ``side`` of ``+1`` rounds up and ``-1`` rounds down, which is the
        conservative choice for a buy and a sell respectively; ``0`` rounds to
        nearest.
        """
        arr = np.asarray(price, dtype=float) / self.tick_size
        if side > 0:
            snapped = np.ceil(arr)
        elif side < 0:
            snapped = np.floor(arr)
        else:
            snapped = np.round(arr)
        return snapped * self.tick_size

    def round_quantity(self, quantity: float | np.ndarray) -> np.ndarray:
        """Truncate ``quantity`` to a whole number of lots, preserving sign."""
        arr = np.asarray(quantity, dtype=float)
        lots = np.trunc(arr / self.lot_size)
        return lots * self.lot_size

    def within_circuit(self, price: float | np.ndarray, reference_close: float) -> np.ndarray:
        """True where ``price`` is inside the daily circuit band.

        A backtest that fills outside the band is booking trades the exchange
        would have frozen — common on gap days in the very names a momentum
        model likes most.
        """
        if self.circuit_limit is None:
            return np.ones_like(np.asarray(price, dtype=float), dtype=bool)
        lo = reference_close * (1.0 - self.circuit_limit)
        hi = reference_close * (1.0 + self.circuit_limit)
        arr = np.asarray(price, dtype=float)
        return (arr >= lo) & (arr <= hi)


def apply_costs(
    trades: pd.DataFrame,
    model: CostModel,
    price_col: str = "price",
    quantity_col: str = "quantity",
    side_col: str = "side",
) -> pd.Series:
    """Vectorised per-trade cost for a trade blotter.

    ``trades`` needs price, absolute quantity and a ``side`` of +1/-1. Buys and
    sells are costed separately because Indian charges are side-dependent.
    """
    for col in (price_col, quantity_col, side_col):
        if col not in trades.columns:
            raise KeyError(f"trades is missing required column {col!r}")
    sides = np.sign(trades[side_col].to_numpy(dtype=float))
    prices = trades[price_col].to_numpy(dtype=float)
    quantities = np.abs(trades[quantity_col].to_numpy(dtype=float))

    out = np.zeros(len(trades), dtype=float)
    for signed in (1, -1):
        mask = sides == signed
        if mask.any():
            out[mask] = model.total_cost(prices[mask], quantities[mask], signed)
    return pd.Series(out, index=trades.index, name="cost")
