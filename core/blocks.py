"""Closest-fit ("I-candidate") block search.

QC and CC each own a single dedicated core, so their search is the original 1-D
closest-fit search (Sec. VI-B). DC may use any core in the data pool ``D``, so its
search is 2-D (core x spectrum). Both rank candidate contiguous free runs of size
>= required by closeness of the run length to the requirement, and both are capped at
their respective ``I`` candidate count. The asymmetry is intentional and preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from core.resource_grid import SlotTable


@dataclass(frozen=True)
class SingleCoreBlock:
    """A candidate spectrum block on a fixed (dedicated) core."""

    start: int
    size: int


@dataclass(frozen=True)
class MultiCoreBlock:
    """A candidate spectrum block on a chosen core from the data pool."""

    core: int
    start: int
    size: int


@dataclass(frozen=True)
class _RankedRun:
    core: int
    start: int
    run_length: int


def _free_runs(free_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous free runs as ``(start, length)`` from a boolean free mask."""
    runs: List[Tuple[int, int]] = []
    start = None
    for i, is_free in enumerate(free_mask):
        if is_free and start is None:
            start = i
        elif not is_free and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(free_mask) - start))
    return runs


def _rank_runs(runs: Sequence[_RankedRun], required_fs: int, cap: int) -> List[_RankedRun]:
    """Closest-fit ranking: runs >= required, sorted by |run_length - required| then
    deterministically by (core, start). Capped at ``cap`` entries."""
    eligible = [r for r in runs if r.run_length >= required_fs]
    eligible.sort(key=lambda r: (abs(r.run_length - required_fs), r.core, r.start))
    return eligible[:cap]


def feasible_blocks_single_core(
    link_indices: Sequence[int],
    core: int,
    required_fs: int,
    slot_table: SlotTable,
    cap: int,
) -> Tuple[SingleCoreBlock, ...]:
    """1-D closest-fit search on a single dedicated core (QC or CC)."""
    if required_fs < 1:
        raise ValueError("required_fs must be >= 1")
    free_mask = slot_table.free_mask(link_indices, core)
    runs = [_RankedRun(core=core, start=s, run_length=length) for s, length in _free_runs(free_mask)]
    ranked = _rank_runs(runs, required_fs, cap)
    return tuple(SingleCoreBlock(start=r.start, size=required_fs) for r in ranked)


def feasible_blocks_multi_core(
    link_indices: Sequence[int],
    data_cores: Sequence[int],
    required_fs: int,
    slot_table: SlotTable,
    cap: int,
) -> Tuple[MultiCoreBlock, ...]:
    """2-D closest-fit search across the data-core pool (DC)."""
    if required_fs < 1:
        raise ValueError("required_fs must be >= 1")
    runs: List[_RankedRun] = []
    for core in data_cores:
        free_mask = slot_table.free_mask(link_indices, core)
        runs.extend(
            _RankedRun(core=core, start=s, run_length=length)
            for s, length in _free_runs(free_mask)
        )
    ranked = _rank_runs(runs, required_fs, cap)
    return tuple(MultiCoreBlock(core=r.core, start=r.start, size=required_fs) for r in ranked)
