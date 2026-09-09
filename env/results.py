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
    key-update reassignment) plus the quantities the reward function consumes. The
    placement-quality terms (``fit``, ``xt``, ``compactness``) and the per-request
    minima (``hops_min_*``, ``eta_max``) are computed by the environment at commit
    time, from the pre-commit slot table, so the reward can grade the *decision*
    rather than the request.
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
    hops_min_qc_cc: Optional[int] = None   # shortest candidate QC/CC route for this request
    hops_min_dc: Optional[int] = None      # shortest eligible DC route for this request
    eta_max: Optional[float] = None        # best spectral efficiency in the modulation table
    fit: Optional[float] = None            # mean over channels of F_c / L_c, in (0, 1]
    xt: Optional[float] = None             # normalized newly-forbidden adjacent-core spectrum, in [0, 1]
    compactness: Optional[float] = None    # 1 - mean block start / S, in [0, 1]

    @staticmethod
    def blocked(request_id: int) -> "CommitResult":
        return CommitResult(served=False, request_id=request_id)
