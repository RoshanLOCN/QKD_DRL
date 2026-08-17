"""Outcome of a joint provisioning attempt for one QLR (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.blocks import MultiCoreBlock, SingleCoreBlock
from core.topology import Route


@dataclass(frozen=True)
class CommitResult:
    """All-or-nothing result of provisioning a QLR.

    On success it records both routes and all three placements (needed for release and
    key-update reassignment) plus the quantities the reward function consumes.
    """

    served: bool
    request_id: int
    k1_route: Optional[Route] = None
    k2_route: Optional[Route] = None
    block_qc: Optional[SingleCoreBlock] = None
    block_cc: Optional[SingleCoreBlock] = None
    block_dc: Optional[MultiCoreBlock] = None
    hops_qc_cc: Optional[int] = None       # H_{k1}
    hops_dc: Optional[int] = None          # H_{k2}
    eta_cc: Optional[float] = None         # eta(m*_{k1,CC})
    eta_dc: Optional[float] = None         # eta(m*_{k2,DC})

    @staticmethod
    def blocked(request_id: int) -> "CommitResult":
        return CommitResult(served=False, request_id=request_id)
