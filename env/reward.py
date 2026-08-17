"""Whole-QLR reward (Eq. 12).

Reward is defined at the whole-QLR level: ``+X`` plus bonuses for a short QC/CC route,
a short DC route, and efficient CC/DC modulation when the entire QLR is served; ``-X``
when it is blocked. Rewarding partial success is deliberately impossible here.
"""

from __future__ import annotations

from configs.config import RewardConfig
from env.results import CommitResult


def compute_reward(result: CommitResult, config: RewardConfig) -> float:
    if not result.served:
        return -config.served_base

    if (
        result.hops_qc_cc is None
        or result.hops_dc is None
        or result.eta_cc is None
        or result.eta_dc is None
    ):
        raise ValueError("a served CommitResult must carry hop counts and modulation efficiencies")

    return (
        config.served_base
        + config.beta_qc_cc_hops * (1.0 / result.hops_qc_cc)
        + config.beta_dc_hops * (1.0 / result.hops_dc)
        + config.beta_efficiency * (result.eta_cc + result.eta_dc)
    )
