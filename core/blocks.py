"""Closest-fit ("I-candidate") block search.

QC and CC each own a single dedicated core, so their search is the original 1-D
closest-fit search (Sec. VI-B). DC may use any core in the data pool ``D``, so its
search is 2-D (core x spectrum). Both rank candidate contiguous free runs of size
>= required by closeness of the run length to the requirement, and both are capped at
their respective ``I`` candidate count. The asymmetry is intentional and preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.resource_grid import SlotTable


def _xt_safe_mask(
    free_mask: np.ndarray, link_indices: Sequence[int], core: int, slot_table: SlotTable,
    adjacency: Optional[Mapping[int, Sequence[int]]],
) -> np.ndarray:
    """Narrow ``free_mask`` (already free on ``core``) to slots that are also free on
    every core adjacent to it -- the XT-avoided constraint: a slot occupied on one core
    forbids that same slot on a physically adjacent core, for every other request."""
    if not adjacency:
        return free_mask
    mask = free_mask
    for adjacent_core in adjacency.get(core, ()):
        mask = mask & slot_table.free_mask(link_indices, adjacent_core)
    return mask


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
    adjacency: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[SingleCoreBlock, ...]:
    """1-D closest-fit search on a single dedicated core (QC or CC).

    ``adjacency`` (core -> physically adjacent cores), when given, applies the
    XT-avoided constraint by excluding slots that are occupied on an adjacent core.
    """
    if required_fs < 1:
        raise ValueError("required_fs must be >= 1")
    free_mask = slot_table.free_mask(link_indices, core)
    free_mask = _xt_safe_mask(free_mask, link_indices, core, slot_table, adjacency)
    runs = [_RankedRun(core=core, start=s, run_length=length) for s, length in _free_runs(free_mask)]
    ranked = _rank_runs(runs, required_fs, cap)
    return tuple(SingleCoreBlock(start=r.start, size=required_fs) for r in ranked)


def free_run_length(
    link_indices: Sequence[int],
    core: int,
    start: int,
    slot_table: SlotTable,
    adjacency: Optional[Mapping[int, Sequence[int]]] = None,
) -> int:
    """Length of the contiguous (XT-safe) free run that contains slot ``start`` on
    ``core`` along the route -- the ``L`` a candidate block of size ``F`` is cut from.
    Zero if ``start`` is not free. Used for the closeness-of-fit reward term F / L."""
    mask = _xt_safe_mask(slot_table.free_mask(link_indices, core), link_indices, core, slot_table, adjacency)
    if not mask[start]:
        return 0
    lo = start
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = start
    while hi + 1 < len(mask) and mask[hi + 1]:
        hi += 1
    return hi - lo + 1


def xt_new_footprint(
    link_indices: Sequence[int],
    core: int,
    start: int,
    size: int,
    slot_table: SlotTable,
    adjacency: Optional[Mapping[int, Sequence[int]]] = None,
) -> int:
    """Number of (link, adjacent core, slot) units that occupying ``start:start+size``
    on ``core`` makes NEWLY unusable under the XT-avoided constraint.

    A slot on an adjacent core counts only if it was usable before this allocation:
    free on that core and not already forbidden by one of *its* other occupied
    neighbours. Spectrally aligning a block with neighbours' existing occupancy
    therefore costs nothing extra, while opening a fresh spectral region on a
    many-neighboured core (e.g. a hub core) costs the most."""
    if not adjacency:
        return 0
    count = 0
    for adjacent in adjacency.get(core, ()):
        # Counted per link (not with the route-level AND) so a multi-link route
        # sterilises proportionally more spectrum than a single-link one.
        for link in link_indices:
            per_link = slot_table.free_mask([link], adjacent)
            for other in adjacency.get(adjacent, ()):
                if other != core:
                    per_link = per_link & slot_table.free_mask([link], other)
            count += int(per_link[start:start + size].sum())
    return count


def feasible_blocks_multi_core(
    link_indices: Sequence[int],
    data_cores: Sequence[int],
    required_fs: int,
    slot_table: SlotTable,
    cap: int,
    adjacency: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[MultiCoreBlock, ...]:
    """2-D closest-fit search across the data-core pool (DC). See ``adjacency`` above."""
    if required_fs < 1:
        raise ValueError("required_fs must be >= 1")
    runs: List[_RankedRun] = []
    for core in data_cores:
        free_mask = slot_table.free_mask(link_indices, core)
        free_mask = _xt_safe_mask(free_mask, link_indices, core, slot_table, adjacency)
        runs.extend(
            _RankedRun(core=core, start=s, run_length=length)
            for s, length in _free_runs(free_mask)
        )
    ranked = _rank_runs(runs, required_fs, cap)
    return tuple(MultiCoreBlock(core=r.core, start=r.start, size=required_fs) for r in ranked)
