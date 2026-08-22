"""Recursive level forecasting, with and without fractional differencing.

This mirrors the framing in the Stock-Portfolio-Optimiser project: resample
prices to a monthly mean, split train/test by date, forecast the *next* value,
then feed the prediction back in and forecast again until the test window is
covered. The original used ARIMA; here the forecaster is an LSTM, so the only
thing that changes between the two arms of the comparison is the **input
representation**.

The fracdiff arm exists to answer one question: does making the input stationary
while retaining memory change what the network can learn? Differencing once
(what ARIMA's ``d=1`` does) throws away the level information; not differencing
at all leaves a unit root that pushes a network towards the degenerate
"predict the last value" solution. Fractional ``d`` sits between the two, and
:class:`FracDiffTransformer` inverts exactly so both arms are scored in the same
price space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afml_india import finkit  # noqa: F401  (side effect: mlfinlab on sys.path)
from afml_india.utils.logging import get_logger

logger = get_logger(__name__)


class FracDiffTransformer:
    """Fixed-width fractional differencing with an exact inverse.

    The FFD filter is ``fd_t = sum_j w[j] * x_{t-W+1+j}`` with ``w[-1] == 1``
    weighting the current observation. Because that leading weight is exactly
    one and every lagged term is known, the level can be recovered without
    approximation::

        x_t = fd_t - sum_{j<W-1} w[j] * x_{t-W+1+j}

    That is what makes a like-for-like comparison possible: the fracdiff model
    forecasts in differenced space, and its prediction is mapped back to a price
    before any error metric is computed.
    """

    def __init__(self, d: float, thresh: float = 1e-4, log: bool = True) -> None:
        from mlfinlab.features.fracdiff import get_weights_ffd  # noqa: PLC0415

        if d < 0:
            raise ValueError("d must be non-negative")
        self.d = float(d)
        self.thresh = thresh
        self.log = log
        # ``lim`` bounds the window; 10,000 is far beyond any weight that
        # survives a sensible threshold.
        self.weights = get_weights_ffd(self.d, thresh, 10_000).ravel()
        self.width = len(self.weights)
        if not np.isclose(self.weights[-1], 1.0):
            raise RuntimeError("unexpected weight convention: w[-1] should be 1.0")

    # ------------------------------------------------------------------ #
    def _to_space(self, prices: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(prices, dtype=float)
        if self.log:
            if (values <= 0).any():
                raise ValueError("log-space fracdiff needs strictly positive prices")
            return np.log(values)
        return values

    def _from_space(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values) if self.log else values

    # ------------------------------------------------------------------ #
    def transform(self, prices: pd.Series) -> pd.Series:
        """Fractionally difference a price series, dropping the warm-up window."""
        values = self._to_space(prices)
        if len(values) < self.width:
            raise ValueError(
                f"need at least {self.width} observations for d={self.d} at "
                f"thresh={self.thresh}, got {len(values)}. Loosen thresh."
            )
        out = np.full(len(values), np.nan)
        for i in range(self.width - 1, len(values)):
            out[i] = float(self.weights @ values[i - self.width + 1 : i + 1])
        return pd.Series(out, index=prices.index, name=f"fd_{self.d}").dropna()

    def invert_next(self, history: pd.Series | np.ndarray, fd_next: float) -> float:
        """Recover the next price from a forecast in differenced space.

        ``history`` is the price series up to and including time ``t``; the
        returned value is the price at ``t + 1`` implied by ``fd_next``.
        """
        values = self._to_space(history)
        if len(values) < self.width - 1:
            raise ValueError(
                f"need {self.width - 1} prior observations to invert, got {len(values)}"
            )
        lagged = values[-(self.width - 1) :] if self.width > 1 else np.array([])
        # w[:-1] applies to the lagged terms; w[-1] == 1 applies to the unknown.
        level = fd_next - float(self.weights[:-1] @ lagged)
        return float(self._from_space(np.array([level]))[0])

    def memory_retained(self, prices: pd.Series) -> float:
        """Correlation between the differenced series and the original levels."""
        differenced = self.transform(prices)
        aligned = pd.Series(self._to_space(prices), index=prices.index).reindex(differenced.index)
        return float(np.corrcoef(aligned.to_numpy(), differenced.to_numpy())[0, 1])


@dataclass
class ForecastConfig:
    """Hyperparameters for :class:`RecursiveLSTMForecaster`."""

    lookback: int = 12
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    max_epochs: int = 400
    patience: int = 40
    batch_size: int = 16
    validation_fraction: float = 0.15
    seed: int = 0
    device: str = "cpu"
    verbose: bool = False


class RecursiveLSTMForecaster:
    """One-step-ahead LSTM regressor, applied recursively over a test window.

    Scaling is fit on the **training window only**. This matters more than it
    looks: the usual min-max-over-the-whole-series setup leaks the test period's
    range into training, and on a trending series that alone can manufacture a
    convincing-looking forecast.
    """

    def __init__(self, config: ForecastConfig | None = None, **overrides) -> None:
        self.config = config or ForecastConfig(**overrides)
        self.model_ = None
        self.mean_: float | None = None
        self.std_: float | None = None
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _windows(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        if len(values) <= lookback:
            raise ValueError(f"need more than {lookback} observations, got {len(values)}")
        X = np.stack([values[i : i + lookback] for i in range(len(values) - lookback)])
        y = values[lookback:]
        return X[:, :, None].astype(np.float32), y.astype(np.float32)

    def _build(self):
        import torch  # noqa: PLC0415

        cfg = self.config
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    1,
                    cfg.hidden_size,
                    cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(cfg.hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net()

    # ------------------------------------------------------------------ #
    def fit(self, series: pd.Series) -> RecursiveLSTMForecaster:
        """Train on a 1D series (prices, or a differenced transform of them)."""
        import torch  # noqa: PLC0415

        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        values = np.asarray(series, dtype=np.float64)
        self.mean_ = float(values.mean())
        self.std_ = float(values.std()) or 1.0
        scaled = (values - self.mean_) / self.std_

        X, y = self._windows(scaled, cfg.lookback)
        n_val = int(len(X) * cfg.validation_fraction)
        n_val = n_val if n_val >= 4 else 0
        split = len(X) - n_val

        device = torch.device(cfg.device)
        model = self._build().to(device)
        criterion = torch.nn.MSELoss()
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        Xt = torch.as_tensor(X[:split], device=device)
        yt = torch.as_tensor(y[:split], device=device)
        if n_val:
            Xv = torch.as_tensor(X[split:], device=device)
            yv = torch.as_tensor(y[split:], device=device)

        best, best_state, bad = np.inf, None, 0
        self.history_ = []
        for epoch in range(cfg.max_epochs):
            model.train()
            order = torch.randperm(len(Xt), device=device)
            total = 0.0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                optimiser.zero_grad()
                loss = criterion(model(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * len(idx)
            train_loss = total / max(len(Xt), 1)

            if n_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(Xv), yv))
            else:
                val_loss = train_loss
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best - 1e-7:
                best, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        if cfg.verbose:
            logger.info("Trained %d epochs, best val MSE %.6f", len(self.history_), best)
        return self

    # ------------------------------------------------------------------ #
    def predict_next(self, window: np.ndarray) -> float:
        """One-step-ahead prediction from the most recent ``lookback`` values."""
        import torch  # noqa: PLC0415

        if self.model_ is None:
            raise RuntimeError("forecaster is not fitted")
        cfg = self.config
        values = np.asarray(window, dtype=np.float64)[-cfg.lookback :]
        if len(values) < cfg.lookback:
            raise ValueError(f"window must hold at least {cfg.lookback} values")
        scaled = (values - self.mean_) / self.std_
        tensor = torch.as_tensor(
            scaled[None, :, None].astype(np.float32), device=torch.device(cfg.device)
        )
        with torch.no_grad():
            out = float(self.model_(tensor))
        return out * self.std_ + self.mean_


def recursive_forecast_levels(
    prices: pd.Series,
    train_end: str | pd.Timestamp,
    config: ForecastConfig | None = None,
    transformer: FracDiffTransformer | None = None,
) -> pd.Series:
    """Recursively forecast prices over the test window, in price space.

    Parameters
    ----------
    prices:
        Full price series; everything after ``train_end`` is the test window.
    transformer:
        ``None`` trains directly on price levels (the baseline arm). Supplying a
        :class:`FracDiffTransformer` trains on the differenced series and
        inverts each prediction back to a price (the fracdiff arm).

    Notes
    -----
    Forecasts are genuinely recursive — each step consumes the model's own
    previous prediction, never the realised price. That is the honest version of
    multi-step forecasting and it is much harder than the one-step-ahead chart
    that most write-ups show.
    """
    prices = prices.dropna().sort_index()
    train_end = pd.Timestamp(train_end)
    if prices.index.tz is not None and train_end.tz is None:
        train_end = train_end.tz_localize(prices.index.tz)

    train = prices[prices.index <= train_end]
    test = prices[prices.index > train_end]
    if test.empty:
        raise ValueError(f"no observations after train_end={train_end}")

    forecaster = RecursiveLSTMForecaster(config)

    if transformer is None:
        forecaster.fit(train)
        history = list(train.to_numpy(dtype=float))
        predictions = []
        for _ in range(len(test)):
            nxt = forecaster.predict_next(np.asarray(history))
            predictions.append(nxt)
            history.append(nxt)
        return pd.Series(predictions, index=test.index, name="forecast")

    differenced = transformer.transform(train)
    forecaster.fit(differenced)

    fd_history = list(differenced.to_numpy(dtype=float))
    price_history = list(train.to_numpy(dtype=float))
    predictions = []
    for _ in range(len(test)):
        fd_next = forecaster.predict_next(np.asarray(fd_history))
        price_next = transformer.invert_next(np.asarray(price_history), fd_next)
        predictions.append(price_next)
        # Feed both spaces forward so the next step is fully recursive.
        fd_history.append(fd_next)
        price_history.append(price_next)
    return pd.Series(predictions, index=test.index, name="forecast")


def walk_forward_forecast(
    prices: pd.Series,
    train_end: str | pd.Timestamp,
    config: ForecastConfig | None = None,
    transformer: FracDiffTransformer | None = None,
    refit_every: int | None = None,
) -> pd.Series:
    """One-step-ahead forecasts across the test window, using realised history.

    At each test step the model sees the **actual** observations up to ``t`` and
    predicts ``t + 1``. This is the setup a strategy actually operates under —
    yesterday's price is known before today's decision — and it is what makes the
    output usable as a trading signal. :func:`recursive_forecast_levels` answers a
    different and much harder question (what will the path look like from here),
    and its errors compound.

    Parameters
    ----------
    transformer:
        ``None`` models price levels directly. A :class:`FracDiffTransformer`
        models the differenced series and inverts each prediction back to a
        price, so both arms are scored in price space.
    refit_every:
        Refit on all data realised so far every ``n`` steps, expanding the
        training window. ``None`` fits once on the training window, which is
        cheaper and avoids confounding the comparison with refit frequency.

    Notes
    -----
    The differenced series is computed over the whole sample, which is *not*
    leakage: fixed-width fracdiff at ``t`` reads only ``x_{t-W+1..t}``. The
    scaler is still fit on the training window alone.
    """
    prices = prices.dropna().sort_index()
    train_end = pd.Timestamp(train_end)
    if prices.index.tz is not None and train_end.tz is None:
        train_end = train_end.tz_localize(prices.index.tz)

    test_index = prices.index[prices.index > train_end]
    if len(test_index) == 0:
        raise ValueError(f"no observations after train_end={train_end}")

    if transformer is None:
        modelled = prices
    else:
        modelled = transformer.transform(prices)

    train_modelled = modelled[modelled.index <= train_end]
    cfg = config or ForecastConfig()
    if len(train_modelled) <= cfg.lookback + 4:
        raise ValueError(
            f"only {len(train_modelled)} training observations after the fracdiff "
            f"warm-up, need more than lookback={cfg.lookback}. Loosen the fracdiff "
            "threshold or shorten the lookback."
        )

    forecaster = RecursiveLSTMForecaster(cfg).fit(train_modelled)

    values = modelled.to_numpy(dtype=float)
    positions = {stamp: i for i, stamp in enumerate(modelled.index)}
    predictions, kept = [], []

    for step, stamp in enumerate(test_index):
        target = positions.get(stamp)
        if target is None or target < cfg.lookback:
            continue  # inside the fracdiff warm-up; nothing to condition on

        if refit_every and step > 0 and step % refit_every == 0:
            forecaster = RecursiveLSTMForecaster(cfg).fit(modelled.iloc[:target])

        # Everything strictly before the target: realised, never predicted.
        window = values[target - cfg.lookback : target]
        prediction = forecaster.predict_next(window)

        if transformer is not None:
            history = prices.loc[: modelled.index[target - 1]]
            prediction = transformer.invert_next(history, prediction)
        predictions.append(prediction)
        kept.append(stamp)

    if not predictions:
        raise ValueError("no test step had enough history to forecast")
    return pd.Series(predictions, index=pd.DatetimeIndex(kept), name="forecast")


def expected_returns(forecast: pd.Series, prices: pd.Series) -> pd.Series:
    """Predicted one-period return implied by a one-step-ahead price forecast.

    ``r_hat_t = forecast_t / price_{t-1} - 1`` — the forecast for ``t`` against
    the last price actually observed before it. This is the quantity a portfolio
    layer consumes; a raw price forecast is not directly comparable across names.
    """
    aligned_prices = prices.reindex(prices.index.union(forecast.index)).sort_index().ffill()
    previous = aligned_prices.shift(1).reindex(forecast.index)
    return (forecast / previous - 1.0).rename("expected_return")


def forecast_errors(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """RMSE, MAE, MAPE and directional accuracy on a common index.

    Directional accuracy is the one that matters for a portfolio: a forecast can
    have a flattering RMSE purely by tracking the level, while getting the sign
    of every change wrong.
    """
    common = actual.index.intersection(predicted.index)
    if len(common) == 0:
        raise ValueError("actual and predicted share no timestamps")
    a = actual.loc[common].to_numpy(dtype=float)
    p = predicted.loc[common].to_numpy(dtype=float)

    error = p - a
    out = {
        "n": float(len(common)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error / np.where(a == 0, np.nan, a))) * 100),
        "bias": float(np.mean(error)),
    }
    if len(common) > 1:
        actual_direction = np.sign(np.diff(a))
        predicted_direction = np.sign(np.diff(p))
        valid = actual_direction != 0
        if not valid.any() or np.all(predicted_direction == 0):
            # A flat forecast expresses no direction at all. Scoring it as 0%
            # correct would read as "always wrong", which is not what happened.
            out["directional_accuracy"] = float("nan")
        else:
            out["directional_accuracy"] = float(
                np.mean(actual_direction[valid] == predicted_direction[valid])
            )
    else:
        out["directional_accuracy"] = float("nan")
    return pd.Series(out)
