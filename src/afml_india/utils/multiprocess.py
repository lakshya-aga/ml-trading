"""Parallel apply over a partitioned pandas index (AFML chapter 20).

The book's ``mpPandasObj`` splits the *molecule* — the atoms of work, usually
index labels — into chunks and runs the callback on each chunk. This version
keeps the same contract but defaults to a serial path, because the callbacks in
this package are already vectorised and process startup usually dominates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd


def _linear_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    return np.ceil(parts).astype(int)


def _nested_parts(num_atoms: int, num_threads: int, upper_triangle: bool = False) -> np.ndarray:
    """Partition atoms so each chunk carries equal work for triangular loops."""
    parts: list[float] = [0.0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part**0.5) / 2.0
        parts.append(part)
    parts_arr = np.round(parts).astype(int)
    if upper_triangle:
        parts_arr = np.cumsum(np.diff(parts_arr)[::-1])
        parts_arr = np.append(np.array([0]), parts_arr)
    return parts_arr


def _expand_call(kwargs: dict[str, Any]) -> Any:
    func = kwargs.pop("func")
    return func(**kwargs)


def mp_pandas_obj(
    func: Callable[..., Any],
    pd_obj: tuple[str, Sequence[Any]],
    num_threads: int = 1,
    mp_batches: int = 1,
    lin_mols: bool = True,
    **kwargs: Any,
) -> Any:
    """Apply ``func`` over a partitioned index and concatenate the results.

    Parameters
    ----------
    func:
        Callback accepting the molecule under the keyword named by ``pd_obj[0]``.
    pd_obj:
        ``(argument_name, atoms)`` — the keyword to pass each chunk under, and
        the full sequence of atoms to split.
    num_threads:
        Worker processes. ``1`` (the default) runs in-process, which is both
        faster for small jobs and far easier to debug.
    mp_batches:
        Number of chunks per worker; more batches smooths uneven work.
    lin_mols:
        ``True`` for evenly sized chunks, ``False`` for the triangular
        partitioning used by nested loops.
    """
    arg_name, atoms = pd_obj
    atoms = list(atoms)
    if not atoms:
        return pd.Series(dtype=float)

    n_chunks = max(1, num_threads * mp_batches)
    parts = _linear_parts(len(atoms), n_chunks) if lin_mols else _nested_parts(len(atoms), n_chunks)

    jobs: list[dict[str, Any]] = []
    for i in range(1, len(parts)):
        job = dict(kwargs)
        job[arg_name] = atoms[parts[i - 1] : parts[i]]
        job["func"] = func
        jobs.append(job)

    if num_threads <= 1:
        outputs = [_expand_call(dict(job)) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            outputs = list(pool.map(_expand_call, jobs))

    outputs = [out for out in outputs if out is not None and len(out) > 0]
    if not outputs:
        return pd.Series(dtype=float)
    if isinstance(outputs[0], pd.DataFrame):
        result = pd.concat(outputs, axis=0)
    elif isinstance(outputs[0], pd.Series):
        result = pd.concat(outputs, axis=0)
    else:
        return outputs
    return result.sort_index()
