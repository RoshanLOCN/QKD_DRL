"""Joint action mask (Step 8.7).

A joint action ``(k1, i_qc, i_cc, k2, i_dc)`` is valid **only** if all five components are
real (not padding) *and* the QC, CC, and DC blocks at those indices all actually exist.
Checking the three channels together (not independently) is what prevents the agent from
being offered a partial combination that cannot be committed as a whole QLR.
"""

from __future__ import annotations

import numpy as np

from configs.config import CandidateConfig, RoutingConfig
from env.candidates import RequestCandidates
from env.state import encode_action


def build_joint_action_mask(
    request: RequestCandidates, routing: RoutingConfig, candidates: CandidateConfig
) -> np.ndarray:
    """Boolean mask over the full joint action space; True == jointly feasible."""
    size = routing.k1 * candidates.i_qc * candidates.i_cc * routing.k2 * candidates.i_dc
    mask = np.zeros(size, dtype=bool)

    for k1 in range(len(request.k1_routes)):
        qc_blocks = request.blocks_qc.get(k1, ())
        cc_blocks = request.blocks_cc.get(k1, ())
        dc_routes = request.eligible_dc.get(k1, ())
        for i_qc in range(len(qc_blocks)):
            for i_cc in range(len(cc_blocks)):
                for k2 in range(len(dc_routes)):
                    dc_blocks = request.blocks_dc.get((k1, k2), ())
                    for i_dc in range(len(dc_blocks)):
                        mask[encode_action((k1, i_qc, i_cc, k2, i_dc), routing, candidates)] = True
    return mask
