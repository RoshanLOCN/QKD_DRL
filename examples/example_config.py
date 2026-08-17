"""EXAMPLE configuration -- the caller-supplied values live here, not in the library.

Every number below is an example the *user* provides; the library itself ships no
defaults. The values mirror the paper's Section X examples so the system is runnable
out of the box; edit freely for your own study. Run with, e.g.::

    python -m training.simulator train --config examples/example_config.py --agent PPO
    python -m training.simulator evaluate --config examples/example_config.py

Learning hyperparameters below follow the fix order in docs/training_diagnosis.md
section 5/6: buffer_size, learning_rate, gamma and hidden_sizes are resized, and the
previously-missing PPO fields (entropy_coef, gae_lambda, minibatch_size, max_grad_norm,
normalize_advantages, target_kl) are now present. This alone does not fix blocking
probability -- that required the credit-assignment fix in agents/ppo.py and
env/environment.py (diagnosis section 2); these hyperparameters are what let that fix
actually train well once the signal is there.
"""

from __future__ import annotations

import os

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

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_config() -> SimulationConfig:
    learning = LearningConfig(
        gamma=0.99,                 # horizon 100 requests >> ~30-arrival mean connection lifetime
        learning_rate=3e-4,
        hidden_sizes=(256, 256),
        buffer_size=2048,           # N: was 64 -- 2% coverage of the 3125-action space
        epochs_per_update=4,
    )
    return SimulationConfig(
        topology_path=os.path.join(_HERE, "sample_topology.json"),
        sentinel_value=-1.0,
        network=NetworkConfig(
            cores_per_link=7,
            fsus_per_core=320,
            fsu_width_ghz=12.5,
            core_qc_index=0,
            core_cc_index=1,
        ),
        modulation=ModulationConfig(
            formats=(
                ModulationFormat("BPSK", 1.0, 4000.0),
                ModulationFormat("QPSK", 2.0, 2000.0),
                ModulationFormat("8QAM", 3.0, 1000.0),
                ModulationFormat("16QAM", 4.0, 500.0),
            ),
            guard_band_fsus=1,
        ),
        quantum=QuantumConfig(
            f_qc_min=2,
            r_qkd_km=1500.0,        # kept high so Tokyo12 links are valid
        ),
        routing=RoutingConfig(k1=5, k2=5),
        candidates=CandidateConfig(i_qc=5, i_cc=5, i_dc=5),
        traffic=TrafficConfig(
            # Raised into a real load regime: at arrival_rate=5.0/holding=6.0 (30 Erlangs)
            # measured BP was 0% on this topology -- no blocking signal at all to learn
            # from. holding_time raised to 15.0 (connections held longer); training
            # arrival_rate set to 20.0 per explicit instruction (300 Erlangs training
            # load) -- the evaluation sweep below is the one that starts at 500 Erlangs.
            arrival_rate=20.0,
            mean_holding_time=15.0,
            key_update_period=20.0,
            b_qc=1.0,
            b_cc_classes=(10.0, 20.0),
            b_dc_classes=(100.0, 200.0, 400.0),
            seed=2024,
        ),
        reward=RewardConfig(
            served_base=1.0, beta_qc_cc_hops=0.2, beta_dc_hops=0.2, beta_efficiency=0.05
        ),
        ppo=PPOConfig(
            learning=learning,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            gae_lambda=0.95,
            entropy_coef=0.01,
            minibatch_size=256,
            max_grad_norm=0.5,
            normalize_advantages=True,
            target_kl=0.02,
        ),
        dqn=DQNConfig(learning=learning),
        exploration=ExplorationConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995),
        training=TrainingConfig(
            num_episodes=5000,
            requests_per_episode=3000,      # was 1500 -- halves the per-episode BP noise floor
            checkpoint_path=os.path.join(_HERE, "checkpoints", "best_model"),
            checkpoint_bp_window=20,        # average over >=20 episodes before checkpointing (4.5)
        ),
        evaluation=EvaluationConfig(
            # Erlang load = arrival_rate * mean_holding_time (15.0). Starts at ~500
            # Erlangs per explicit instruction: 34.0*15=510 up to 200.0*15=3000.
            arrival_rates=(34.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0, 200.0),
            requests_per_run=3000,
        ),
    )
