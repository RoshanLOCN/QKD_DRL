"""Configuration objects for the SS-EON QKD DRL framework.

Design rule (explicitly requested): **config-only, no defaults**. Every value that
the specification documents give only as an example ("e.g. ...") lives here as a
required field with **no default**, so the code never assumes a value on its own.
Each dataclass validates its own invariants at construction time, so an invalid
configuration cannot be built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class NetworkConfig:
    """SS-EON network model ``G(V, E, C, S, M)`` -- the numeric/spatial part.

    ``V`` and ``E`` (and per-link distances) come from the external topology file;
    the remaining structural parameters are supplied here.
    """

    cores_per_link: int              # C
    fsus_per_core: int               # S
    fsu_width_ghz: float             # Delta f
    core_qc_index: int               # dedicated QC core index (the set {core_QC})
    core_cc_index: int               # dedicated CC core index (the set {core_CC})

    def __post_init__(self) -> None:
        if self.cores_per_link < 3:
            raise ValueError(
                "cores_per_link (C) must be >= 3: one core_QC, one core_CC, and at "
                "least one data core in D = C - 2"
            )
        if self.fsus_per_core < 1:
            raise ValueError("fsus_per_core (S) must be >= 1")
        if self.fsu_width_ghz <= 0:
            raise ValueError("fsu_width_ghz (Delta f) must be > 0")
        for name, idx in (("core_qc_index", self.core_qc_index), ("core_cc_index", self.core_cc_index)):
            if not (0 <= idx < self.cores_per_link):
                raise ValueError(f"{name}={idx} out of range [0, {self.cores_per_link})")
        if self.core_qc_index == self.core_cc_index:
            raise ValueError("core_qc_index and core_cc_index must be distinct")

    @property
    def data_core_indices(self) -> Tuple[int, ...]:
        """The data-core pool ``D`` = all cores except core_QC and core_CC."""
        reserved = {self.core_qc_index, self.core_cc_index}
        return tuple(c for c in range(self.cores_per_link) if c not in reserved)


@dataclass(frozen=True)
class ModulationFormat:
    """One entry of the distance-adaptive modulation table (CC/DC only)."""

    name: str
    spectral_efficiency: float       # eta, bits/s/Hz
    max_reach_km: float

    def __post_init__(self) -> None:
        if self.spectral_efficiency <= 0:
            raise ValueError(f"modulation {self.name!r}: spectral_efficiency must be > 0")
        if self.max_reach_km <= 0:
            raise ValueError(f"modulation {self.name!r}: max_reach_km must be > 0")


@dataclass(frozen=True)
class ModulationConfig:
    """Distance-adaptive modulation table and guard band (used for CC and DC only)."""

    formats: Tuple[ModulationFormat, ...]
    guard_band_fsus: int             # G

    def __post_init__(self) -> None:
        if not self.formats:
            raise ValueError("modulation table must contain at least one format")
        if self.guard_band_fsus < 0:
            raise ValueError("guard_band_fsus (G) must be >= 0")


@dataclass(frozen=True)
class QuantumConfig:
    """Quantum-channel parameters that are protocol-fixed, not modulation-derived."""

    f_qc_min: int                    # F^{QC}_min fixed footprint in FSUs
    r_qkd_km: float                  # QKD reach limit for QC/CC candidate routes

    def __post_init__(self) -> None:
        if self.f_qc_min < 1:
            raise ValueError("f_qc_min must be >= 1")
        if self.r_qkd_km <= 0:
            raise ValueError("r_qkd_km must be > 0")


@dataclass(frozen=True)
class RoutingConfig:
    """Candidate-route counts."""

    k1: int                          # shared QC/CC candidate routes
    k2: int                          # cap on DC routes link-disjoint from a given k1

    def __post_init__(self) -> None:
        if self.k1 < 1:
            raise ValueError("k1 must be >= 1")
        if self.k2 < 1:
            raise ValueError("k2 must be >= 1")


@dataclass(frozen=True)
class CandidateConfig:
    """Per-channel closest-fit candidate-block counts (the ``I`` values)."""

    i_qc: int
    i_cc: int
    i_dc: int

    def __post_init__(self) -> None:
        for name, value in (("i_qc", self.i_qc), ("i_cc", self.i_cc), ("i_dc", self.i_dc)):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class TrafficConfig:
    """Traffic generator parameters (Poisson arrivals, exponential holding times)."""

    arrival_rate: float                     # Poisson lambda
    mean_holding_time: float                # exponential mean
    key_update_period: float                # T
    b_qc: float                             # fixed, protocol-determined QC demand
    b_cc_classes: Tuple[float, ...]         # discrete CC bandwidth classes (Gbps)
    b_dc_classes: Tuple[float, ...]         # discrete DC bandwidth classes (Gbps)
    seed: int

    def __post_init__(self) -> None:
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")
        if self.mean_holding_time <= 0:
            raise ValueError("mean_holding_time must be > 0")
        if self.key_update_period <= 0:
            raise ValueError("key_update_period (T) must be > 0")
        if self.b_qc <= 0:
            raise ValueError("b_q c must be > 0")
        if not self.b_cc_classes:
            raise ValueError("b_cc_classes must be non-empty")
        if not self.b_dc_classes:
            raise ValueError("b_dc_classes must be non-empty")
        if any(b <= 0 for b in self.b_cc_classes):
            raise ValueError("all b_cc_classes must be > 0")
        if any(b <= 0 for b in self.b_dc_classes):
            raise ValueError("all b_dc_classes must be > 0")


@dataclass(frozen=True)
class RewardConfig:
    """Whole-QLR reward parameters (Eq. 12)."""

    served_base: float               # X
    beta_qc_cc_hops: float           # beta   (bonus on 1 / H_k1)
    beta_dc_hops: float              # beta'  (bonus on 1 / H_k2)
    beta_efficiency: float           # beta'' (bonus on eta_CC + eta_DC)


@dataclass(frozen=True)
class LearningConfig:
    """Shared learning hyperparameters for the trainable agents (PPO and DQN)."""

    gamma: float
    learning_rate: float
    hidden_sizes: Tuple[int, ...]
    buffer_size: int                 # N: experiences per on-policy update
    epochs_per_update: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not self.hidden_sizes or any(h < 1 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be non-empty positive integers")
        if self.buffer_size < 1:
            raise ValueError("buffer_size (N) must be >= 1")
        if self.epochs_per_update < 1:
            raise ValueError("epochs_per_update must be >= 1")


@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific parameters on top of the shared learning config."""

    learning: LearningConfig
    clip_epsilon: float
    value_loss_coef: float
    gae_lambda: float                # GAE(lambda) mixing factor
    entropy_coef: float              # exploration pressure on the actor's objective
    minibatch_size: int              # SGD minibatch size within each on-policy update
    max_grad_norm: float             # gradient clip norm, applied to actor and critic separately
    normalize_advantages: bool       # whether to standardize advantages before the policy loss
    target_kl: Optional[float]       # early-stop an update's epochs once mean approx-KL exceeds this; None disables

    def __post_init__(self) -> None:
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be > 0")
        if self.value_loss_coef < 0:
            raise ValueError("value_loss_coef must be >= 0")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 when set")


@dataclass(frozen=True)
class DQNConfig:
    """DQN-specific parameters on top of the shared learning config."""

    learning: LearningConfig


@dataclass(frozen=True)
class ExplorationConfig:
    """Epsilon-greedy exploration schedule (decayed after each update)."""

    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float             # multiplicative factor applied per update

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0):
            raise ValueError("require 0 <= epsilon_min <= epsilon_start <= 1")
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError("epsilon_decay must be in (0, 1]")


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop control (Algorithm 2)."""

    num_episodes: int
    requests_per_episode: int
    checkpoint_path: str
    checkpoint_bp_window: int        # episodes averaged for best-BP checkpointing (see 4.5)

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        if self.requests_per_episode < 1:
            raise ValueError("requests_per_episode must be >= 1")
        if self.checkpoint_bp_window < 1:
            raise ValueError("checkpoint_bp_window must be >= 1")


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation sweep control."""

    arrival_rates: Tuple[float, ...]
    requests_per_run: int

    def __post_init__(self) -> None:
        if not self.arrival_rates:
            raise ValueError("arrival_rates must be non-empty")
        if any(r <= 0 for r in self.arrival_rates):
            raise ValueError("all arrival_rates must be > 0")
        if self.requests_per_run < 1:
            raise ValueError("requests_per_run must be >= 1")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration aggregating every sub-config. No defaults anywhere."""

    topology_path: str
    sentinel_value: float            # padding sentinel for the state vector
    network: NetworkConfig
    modulation: ModulationConfig
    quantum: QuantumConfig
    routing: RoutingConfig
    candidates: CandidateConfig
    traffic: TrafficConfig
    reward: RewardConfig
    ppo: PPOConfig
    dqn: DQNConfig
    exploration: ExplorationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @property
    def action_space_size(self) -> int:
        """``|A| = K1 * I_QC * I_CC * K2 * I_DC`` (Eq. 11)."""
        return (
            self.routing.k1
            * self.candidates.i_qc
            * self.candidates.i_cc
            * self.routing.k2
            * self.candidates.i_dc
        )
