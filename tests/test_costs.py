"""Indian transaction-cost model and instrument conventions."""

from __future__ import annotations

import pandas as pd
import pytest

from afml_india.data.costs import CostModel, InstrumentSpec, Segment, apply_costs


def test_delivery_costs_more_than_intraday():
    delivery = CostModel(segment=Segment.DELIVERY).round_trip_bps()
    intraday = CostModel(segment=Segment.INTRADAY).round_trip_bps()
    assert delivery > intraday
    # STT alone is 10 bps a side on delivery, so the round trip must clear 20.
    assert delivery > 20


def test_delivery_stt_is_symmetric_and_intraday_is_sell_only():
    delivery = CostModel(segment=Segment.DELIVERY)
    intraday = CostModel(segment=Segment.INTRADAY)
    assert delivery._stt(+1) == delivery._stt(-1) == pytest.approx(0.001)
    assert intraday._stt(+1) == 0.0
    assert intraday._stt(-1) == pytest.approx(0.00025)


def test_stamp_duty_is_buy_side_only():
    model = CostModel(segment=Segment.DELIVERY)
    buy = model.leg_cost(1000.0, 100, +1)
    sell = model.leg_cost(1000.0, 100, -1)
    assert buy > sell


def test_brokerage_cap_binds_on_large_orders():
    model = CostModel(brokerage_rate=0.0003, brokerage_cap=20.0)
    assert model.brokerage(10_000) == pytest.approx(3.0)  # under the cap
    assert model.brokerage(10_000_000) == pytest.approx(20.0)  # capped


def test_impact_grows_with_the_square_root_of_participation():
    model = CostModel()
    small = model.impact_bps(1e6, 1e9, 0.02)
    large = model.impact_bps(4e6, 1e9, 0.02)
    # Four times the size is twice the impact under the square-root law.
    assert large == pytest.approx(2 * small, rel=1e-9)


def test_impact_is_zero_when_adv_is_unknown():
    assert CostModel().impact_bps(1e6, 0.0, 0.02) == 0.0


def test_tick_rounding_is_conservative_per_side():
    spec = InstrumentSpec("TEST", tick_size=0.05)
    assert spec.round_price(100.03, +1) == pytest.approx(100.05)  # buy rounds up
    assert spec.round_price(100.03, -1) == pytest.approx(100.00)  # sell rounds down
    assert spec.round_price(100.03, 0) == pytest.approx(100.05)


def test_quantity_truncates_to_whole_lots_preserving_sign():
    spec = InstrumentSpec("TEST", lot_size=25)
    assert spec.round_quantity(74) == 50
    assert spec.round_quantity(-74) == -50


def test_circuit_band():
    spec = InstrumentSpec("TEST", circuit_limit=0.10)
    assert bool(spec.within_circuit(105.0, 100.0))
    assert not bool(spec.within_circuit(111.0, 100.0))
    assert bool(InstrumentSpec("TEST", circuit_limit=None).within_circuit(1e6, 100.0))


def test_apply_costs_prices_both_sides():
    trades = pd.DataFrame({"price": [1000.0, 1000.0], "quantity": [100, 100], "side": [1, -1]})
    costs = apply_costs(trades, CostModel(segment=Segment.DELIVERY))
    assert len(costs) == 2
    assert (costs > 0).all()
    assert costs.iloc[0] != costs.iloc[1]  # asymmetric by construction


def test_apply_costs_requires_its_columns():
    with pytest.raises(KeyError, match="quantity"):
        apply_costs(pd.DataFrame({"price": [1.0], "side": [1]}), CostModel())
