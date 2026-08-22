"""LSTM classifier for bar-level financial sequences.

A scikit-learn-shaped wrapper over a small PyTorch LSTM, with the three things
financial ML needs and generic implementations omit: per-sample weights in the
loss (so AFML uniqueness weights actually bite), early stopping on a
chronological tail rather than a random split, and deterministic seeding.

The architecture is deliberately small. With a few thousand overlapping,
low-signal-to-noise bars, a two-layer LSTM with dropout is already at the edge
of what the data supports — capacity is rarely the binding constraint here, and
a larger network mostly buys a more confident overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

_TORCH_HINT = (
    "PyTorch is required for the LSTM model. Install it with:\n"
    "    pip install 'afml-india[deep]'\n"
    "or use a scikit-learn model from afml_india.models.baselines instead."
)


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_TORCH_HINT) from exc


@dataclass
class LSTMConfig:
    """Hyperparameters for :class:`LSTMClassifier`."""

    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.25
    bidirectional: bool = False

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    #: Fraction of the training fold held out, chronologically, for early stopping.
    validation_fraction: float = 0.2
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    verbose: bool = False
    class_weight: str | None = "balanced"
    history: list = field(default_factory=list)


class LSTMClassifier:
    """Binary LSTM classifier with ``fit``/``predict_proba``/``predict``.

    Parameters
    ----------
    config:
        Hyperparameters. The defaults are tuned for a few thousand bar-level
        samples, not for a large dataset.

    Notes
    -----
    Labels are mapped internally: AFML's ``{-1, +1}`` and sklearn's ``{0, 1}``
    are both accepted, and ``classes_`` reports the original values so
    ``predict`` round-trips.
    """

    def __init__(self, config: LSTMConfig | None = None, **overrides) -> None:
        self.config = config or LSTMConfig(**overrides)
        self.model_ = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ #
    def _build(self, n_features: int):
        torch = _torch()
        nn = torch.nn
        cfg = self.config

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    batch_first=True,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                    bidirectional=cfg.bidirectional,
                )
                out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
                self.head = nn.Sequential(
                    nn.LayerNorm(out_size),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(out_size, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                # Only the final timestep matters: it is the event bar.
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net()

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LSTMClassifier:
        """Train on ``(n_samples, lookback, n_features)`` sequences.

        ``sample_weight`` is applied per sample inside the loss. Passing AFML
        uniqueness or return-attribution weights here is the point of the
        wrapper: without it, overlapping labels are counted as if they were
        independent observations.
        """
        torch = _torch()
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (samples, lookback, features), got shape {X.shape}")
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"LSTMClassifier is binary; y has classes {self.classes_}")
        y_binary = (y == self.classes_[1]).astype(np.float32)

        weights = (
            np.ones(len(y), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32).ravel()
        )
        if len(weights) != len(y):
            raise ValueError("sample_weight length does not match y")
        # Normalise so the loss magnitude does not depend on the weighting
        # scheme, which keeps the learning rate transferable.
        weights = weights / max(weights.mean(), 1e-12)

        # Chronological holdout: the last slice of the training fold. A random
        # split would let a neighbouring overlapping sample leak across.
        n_val = int(len(X) * cfg.validation_fraction)
        if n_val < 8:
            n_val = 0
        split = len(X) - n_val
        X_tr, y_tr, w_tr = X[:split], y_binary[:split], weights[:split]
        X_va, y_va, w_va = X[split:], y_binary[split:], weights[split:]

        self.n_features_ = X.shape[-1]
        device = torch.device(cfg.device)
        model = self._build(self.n_features_).to(device)

        pos_weight = None
        if cfg.class_weight == "balanced":
            n_pos = float(y_tr.sum())
            n_neg = float(len(y_tr) - n_pos)
            if n_pos > 0 and n_neg > 0:
                pos_weight = torch.tensor(n_neg / n_pos, device=device)

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        def as_tensor(arr):
            return torch.as_tensor(arr, device=device)

        Xt, yt, wt = as_tensor(X_tr), as_tensor(y_tr), as_tensor(w_tr)
        if n_val:
            Xv, yv, wv = as_tensor(X_va), as_tensor(y_va), as_tensor(w_va)

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history_ = []

        for epoch in range(cfg.max_epochs):
            model.train()
            order = torch.randperm(len(Xt), device=device)
            epoch_loss, seen = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                optimiser.zero_grad()
                logits = model(Xt[idx])
                loss = (criterion(logits, yt[idx]) * wt[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()
                epoch_loss += float(loss) * len(idx)
                seen += len(idx)
            train_loss = epoch_loss / max(seen, 1)

            if n_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float((criterion(model(Xv), yv) * wv).mean())
            else:
                val_loss = train_loss

            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if cfg.verbose:
                logger.info("epoch %3d  train %.4f  val %.4f", epoch, train_loss, val_loss)

            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.debug("Early stopping at epoch %d (best val %.4f)", epoch, best_loss)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities as ``(n_samples, 2)``, ordered by ``classes_``."""
        torch = _torch()
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(X, device=torch.device(self.config.device)))
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class predictions in the original label space."""
        if self.classes_ is None:
            raise RuntimeError("model is not fitted")
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, self.classes_[1], self.classes_[0])

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatibility."""
        return {"config": self.config}

    def set_params(self, **params) -> LSTMClassifier:
        """scikit-learn compatibility."""
        for key, value in params.items():
            if key == "config":
                self.config = value
            else:
                setattr(self.config, key, value)
        return self
