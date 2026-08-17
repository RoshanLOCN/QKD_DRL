"""Resource-grid module -- live occupancy of every (link, core, FSU).

The cores of every link are partitioned network-wide into ``{core_QC} u {core_CC} u D``
(Eq. 2). The slot table is a boolean array ``free[link, core, fsu]`` (True == free).

Spectrum **continuity** is enforced by intersecting a core's free FSUs across all links
of a route; spectrum **contiguity** is enforced later, in the block search, by taking
contiguous runs of the intersected free mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from configs.config import NetworkConfig


@dataclass(frozen=True)
class CorePartition:
    """Fixed, network-wide partition of cores into QC / CC / data-pool roles."""

    core_qc: int
    core_cc: int
    data_cores: Tuple[int, ...]

    @classmethod
    def from_config(cls, config: NetworkConfig) -> "CorePartition":
        return cls(
            core_qc=config.core_qc_index,
            core_cc=config.core_cc_index,
            data_cores=config.data_core_indices,
        )


class SlotTable:
    """Boolean occupancy grid: ``free[link, core, fsu]`` (True == free)."""

    def __init__(self, num_links: int, cores_per_link: int, fsus_per_core: int) -> None:
        if num_links < 1 or cores_per_link < 1 or fsus_per_core < 1:
            raise ValueError("num_links, cores_per_link, fsus_per_core must all be >= 1")
        self._free = np.ones((num_links, cores_per_link, fsus_per_core), dtype=bool)
        self._fsus = fsus_per_core

    @property
    def fsus_per_core(self) -> int:
        return self._fsus

    def reset(self) -> None:
        self._free.fill(True)

    def free_mask(self, link_indices: Sequence[int], core: int) -> np.ndarray:
        """Free FSUs common to ``core`` across every link of a route (continuity)."""
        if not link_indices:
            raise ValueError("link_indices must be non-empty")
        mask = np.ones(self._fsus, dtype=bool)
        for link in link_indices:
            mask &= self._free[link, core]
        return mask

    def is_block_free(self, link_indices: Sequence[int], core: int, start: int, size: int) -> bool:
        if start < 0 or size < 1 or start + size > self._fsus:
            return False
        for link in link_indices:
            if not self._free[link, core, start:start + size].all():
                return False
        return True

    def occupy(self, link_indices: Sequence[int], core: int, start: int, size: int) -> None:
        if not self.is_block_free(link_indices, core, start, size):
            raise ValueError("cannot occupy a block that is not entirely free")
        for link in link_indices:
            self._free[link, core, start:start + size] = False

    def release(self, link_indices: Sequence[int], core: int, start: int, size: int) -> None:
        for link in link_indices:
            self._free[link, core, start:start + size] = True

    def occupied_fsus_on_core(self, core: int) -> int:
        """Total occupied (link, FSU) units on a single core, summed over links."""
        return int((~self._free[:, core, :]).sum())

    def occupied_fsus_on_cores(self, cores: Sequence[int]) -> int:
        if not cores:
            return 0
        return int((~self._free[:, list(cores), :]).sum())
