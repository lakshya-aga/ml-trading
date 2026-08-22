"""Bridge to the ``mlfinlab`` AFML implementation vendored at ``external/fin-kit``.

The AFML algorithms themselves — bars, filters, fractional differentiation,
triple-barrier labelling, sample weights, purged CV, feature importance, bet
sizing — already exist in the ``fin-kit`` submodule, so this package does not
reimplement them. It adds the layer fin-kit does not have: Indian market
conventions, data loading, and the glue that gets a snapshot into the shape
those functions expect.

Importing this module makes ``mlfinlab`` importable whether or not fin-kit was
pip-installed, by putting the submodule checkout on ``sys.path``. That means a
fresh clone runs after ``git submodule update --init`` with no install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

from afml_india.utils.logging import get_logger

logger = get_logger(__name__)

#: Candidate locations for the fin-kit checkout, nearest first.
_SEARCH_PATHS = (
    Path(__file__).resolve().parents[2] / "external" / "fin-kit",
    Path.cwd() / "external" / "fin-kit",
    Path.home() / "lakshya-aga" / "fin-kit",
)

_INSTALL_HINT = (
    "mlfinlab (fin-kit) is not importable. From the repository root run:\n"
    "    git submodule update --init --recursive\n"
    "or install it directly:\n"
    "    pip install git+https://github.com/lakshya-aga/fin-kit.git"
)


def finkit_root() -> Path | None:
    """Path to the fin-kit checkout, or ``None`` when it cannot be found."""
    for candidate in _SEARCH_PATHS:
        if (candidate / "mlfinlab" / "__init__.py").exists():
            return candidate
    return None


def ensure_finkit_on_path() -> Path | None:
    """Put the fin-kit checkout on ``sys.path`` if ``mlfinlab`` is not installed.

    Returns the path that was added, or ``None`` when ``mlfinlab`` was already
    importable from the environment.
    """
    try:
        import mlfinlab  # noqa: F401, PLC0415

        return None
    except ImportError:
        pass

    root = finkit_root()
    if root is None:
        raise ImportError(_INSTALL_HINT)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        logger.debug("Added fin-kit checkout to sys.path: %s", root)
    return root


def available() -> bool:
    """True when the AFML functions can be imported right now."""
    try:
        ensure_finkit_on_path()
        import mlfinlab.data_structures  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


ensure_finkit_on_path()
