"""Performance metrics: Blocking Probability, Resource Utilization, Discounted Return.

RU is reported both in aggregate and broken out per channel role, because the QC/CC
dedicated cores (one core's capacity per link) and the DC pool (``|D|`` cores per link)
have structurally different utilization dynamics (Eq. 14).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.resource_grid import CorePartition, SlotTable


@dataclass(frozen=True)
class ResourceUtilization:
    qc: float
    cc: float
    dc: float
    aggregate: float


def resource_utilization(
    slot_table: SlotTable, partition: CorePartition, num_links: int
) -> ResourceUtilization:
    fsus = slot_table.fsus_per_core
    total_single = num_links * fsus
    total_dc = num_links * len(partition.data_cores) * fsus

    used_qc = slot_table.occupied_fsus_on_core(partition.core_qc)
    used_cc = slot_table.occupied_fsus_on_core(partition.core_cc)
    used_dc = slot_table.occupied_fsus_on_cores(partition.data_cores)

    total_all = total_single + total_single + total_dc
    return ResourceUtilization(
        qc=used_qc / total_single if total_single else 0.0,
        cc=used_cc / total_single if total_single else 0.0,
        dc=used_dc / total_dc if total_dc else 0.0,
        aggregate=(used_qc + used_cc + used_dc) / total_all if total_all else 0.0,
    )


@dataclass
class EpisodeMetrics:
    """Running counters for one episode.

    ``gamma`` must match the training gamma (pass it in explicitly) -- it previously
    defaulted to a hardcoded 0.99 regardless of what the agent was actually trained with,
    which silently decoupled the logged G_t from the real discount horizon whenever
    ``gamma`` was tuned to anything else.
    """

    gamma: float = 0.99
    served: int = 0
    total: int = 0
    reward_sum: float = 0.0
    discounted_return: float = 0.0
    _gamma_multiplier: float = 1.0

    def record(self, served: bool, reward: float) -> None:
        self.total += 1
        self.reward_sum += reward

        # Iteratively calculates G_t = \sum \gamma^j R(s_{t+j}, a)
        self.discounted_return += self._gamma_multiplier * reward
        self._gamma_multiplier *= self.gamma

        if served:
            self.served += 1

    @property
    def blocking_probability(self) -> float:
        return 1.0 - self.served / self.total if self.total else 0.0

    @property
    def average_reward(self) -> float:
        return self.reward_sum / self.total if self.total else 0.0

    @property
    def discounted_cumulative_reward(self) -> float:
        """Exposes the formal objective metric (Gt)."""
        return self.discounted_return
