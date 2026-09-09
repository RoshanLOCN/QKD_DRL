"""Whole-QLR reward -- Eq. 12, extended with placement-quality terms.

Reward is defined at the whole-QLR level: ``+X`` plus bonuses when the entire QLR is
served, ``-X`` when it is blocked. Rewarding partial success is deliberately impossible.

Original Eq. 12 (``relative_terms=False``, placement betas 0)::

    R = X + beta/H_k1 + beta'/H_k2 + beta''*(eta_CC + eta_DC)

Extended form (``relative_terms=True``) -- every term is a quality of the *decision*
normalized to [0, 1], so the betas are directly comparable::

    R = X + beta *(H_min,k1 / H_k1)          shortest candidate route chosen -> 1
          + beta'*(H_min,k2 / H_k2)
          + beta''*(eta_CC + eta_DC)/(2*eta_max)
          + beta_fit * Fit                    closeness of fit F/L, mean over channels
          - beta_xt  * XT                     newly sterilised adjacent-core spectrum
          + beta_cmp * Cmp                    spectral compactness (low start index)

Why the placement terms exist: under the original reward the block/core choices (3 of
the 5 action dimensions) have no effect on the immediate reward, and their only
consequence -- future blocking -- lies beyond a gamma=0.95 credit horizon. The learner
therefore could not improve on First-Fit's closest-fit rule; these terms make the
placement decision directly gradable. FF and RF never read the reward, so the
comparison across methods is unaffected.
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

    if config.relative_terms:
        if result.hops_min_qc_cc is None or result.hops_min_dc is None or not result.eta_max:
            raise ValueError("relative_terms requires hops_min_* and eta_max on the CommitResult")
        hop_qc_cc = result.hops_min_qc_cc / result.hops_qc_cc
        hop_dc = result.hops_min_dc / result.hops_dc
        efficiency = (result.eta_cc + result.eta_dc) / (2.0 * result.eta_max)
    else:
        hop_qc_cc = 1.0 / result.hops_qc_cc
        hop_dc = 1.0 / result.hops_dc
        efficiency = result.eta_cc + result.eta_dc

    reward = (
        config.served_base
        + config.beta_qc_cc_hops * hop_qc_cc
        + config.beta_dc_hops * hop_dc
        + config.beta_efficiency * efficiency
    )

    placement_betas = (config.beta_fit, config.beta_xt, config.beta_compact)
    if any(b != 0.0 for b in placement_betas):
        if result.fit is None or result.xt is None or result.compactness is None:
            raise ValueError("placement betas require fit/xt/compactness on the CommitResult")
        reward += (
            config.beta_fit * result.fit
            - config.beta_xt * result.xt
            + config.beta_compact * result.compactness
        )
    return reward
