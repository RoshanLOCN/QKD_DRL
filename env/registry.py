"""Active-QLR registry.

A served QLR occupies **two routes and three placements** that must be released or
reassigned together. Each record therefore stores both routes, all three (core, block)
placements, and the release / next-update times, plus the originating request (its
demands are needed to re-solve the joint placement at key-update time).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

from core.blocks import MultiCoreBlock, SingleCoreBlock
from core.topology import Route
from request.qlr import QLR


@dataclass(frozen=True)
class ActiveRecord:
    qlr: QLR
    k1_route: Route
    k2_route: Route
    block_qc: SingleCoreBlock
    block_cc: SingleCoreBlock
    block_dc: MultiCoreBlock
    release_time: float
    next_update_time: float

    def with_next_update(self, next_update_time: float) -> "ActiveRecord":
        return replace(self, next_update_time=next_update_time)


class ActiveQLRRegistry:
    """Registry of currently provisioned QLRs, keyed by request id."""

    def __init__(self) -> None:
        self._records: Dict[int, ActiveRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, request_id: int) -> bool:
        return request_id in self._records

    def add(self, record: ActiveRecord) -> None:
        self._records[record.qlr.request_id] = record

    def remove(self, request_id: int) -> None:
        self._records.pop(request_id, None)

    def replace_record(self, record: ActiveRecord) -> None:
        self._records[record.qlr.request_id] = record

    def due_for_release(self, current_time: float) -> List[ActiveRecord]:
        return [r for r in self._records.values() if r.release_time <= current_time]

    def due_for_update(self, current_time: float) -> List[ActiveRecord]:
        return [
            r
            for r in self._records.values()
            if r.next_update_time <= current_time and r.release_time > current_time
        ]

    def clear(self) -> None:
        self._records.clear()
