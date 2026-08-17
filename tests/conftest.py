"""Shared pytest fixtures. The config here is TEST data (small, synthetic) -- it is not a
claim about Tokyo12's real parameters, only enough to exercise the logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.config import (
    CandidateConfig,
    DQNConfig,
    EvaluationConfig,
    ExplorationConfig,
    LearningConfig,
    ModulationConfig,
    ModulationFormat,
    NetworkConfig,
    PPOConfig,
    QuantumConfig,
    RewardConfig,
    RoutingConfig,
    SimulationConfig,
    TrafficConfig,
    TrainingConfig,
)
from core.topology import load_topology

FIXTURE = Path(__file__).parent / "fixtures" / "mini_topology.json"


@pytest.fixture
def topology_path() -> str:
    return str(FIXTURE)


@pytest.fixture
def topology():
    return load_topology(str(FIXTURE))


def _modulation() -> ModulationConfig:
    return ModulationConfig(
        formats=(
            ModulationFormat("BPSK", 1.0, 4000.0),
            ModulationFormat("QPSK", 2.0, 2000.0),
            ModulationFormat("8QAM", 3.0, 1000.0),
            ModulationFormat("16QAM", 4.0, 500.0),
        ),
        guard_band_fsus=1,
    )


@pytest.fixture
def base_config(topology_path: str, tmp_path) -> SimulationConfig:
    learning = LearningConfig(
        gamma=0.95, learning_rate=1e-3, hidden_sizes=(32,), buffer_size=8, epochs_per_update=2
    )
    return SimulationConfig(
        topology_path=topology_path,
        sentinel_value=-1.0,
        network=NetworkConfig(
            cores_per_link=4, fsus_per_core=16, fsu_width_ghz=12.5, core_qc_index=0, core_cc_index=1
        ),
        modulation=_modulation(),
        quantum=QuantumConfig(f_qc_min=2, r_qkd_km=300.0),
        routing=RoutingConfig(k1=2, k2=2),
        candidates=CandidateConfig(i_qc=2, i_cc=2, i_dc=2),
        traffic=TrafficConfig(
            arrival_rate=1.0,
            mean_holding_time=5.0,
            key_update_period=3.0,
            b_qc=1.0,
            b_cc_classes=(10.0,),
            b_dc_classes=(40.0,),
            seed=123,
        ),
        reward=RewardConfig(served_base=1.0, beta_qc_cc_hops=0.1, beta_dc_hops=0.1, beta_efficiency=0.01),
        ppo=PPOConfig(learning=learning, clip_epsilon=0.2, value_loss_coef=0.5),
        dqn=DQNConfig(learning=learning),
        exploration=ExplorationConfig(epsilon_start=0.5, epsilon_min=0.01, epsilon_decay=0.9),
        training=TrainingConfig(
            num_episodes=2, requests_per_episode=10, checkpoint_path=str(tmp_path / "ckpt")
        ),
        evaluation=EvaluationConfig(arrival_rates=(1.0, 2.0), requests_per_run=10),
    )
