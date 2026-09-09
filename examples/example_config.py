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
        gamma=0.95,                 # explicit instruction: horizon 20 requests for the 5000-episode run
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
            # Reduced from 320: arrival_rate is fixed at 20.0 (not to be raised further,
            # per explicit instruction), and at 320 slots that load produced 0% blocking
            # -- no signal at all. Retuned to 300 (was 110) after enabling xt_avoided
            # below: this network's 7-core layout has a hub core (6) adjacent to every
            # other core, so XT-avoidance alone pushed BP from ~4% to ~36% at 110;
            # 300 restores ~3.2% BP at arrival_rate=20 with XT-avoidance active.
            fsus_per_core=300,
            fsu_width_ghz=12.5,
            core_qc_index=0,
            core_cc_index=1,
            # XT-avoided allocation (Xav-RQCD): a slot occupied on one core forbids
            # that same slot on any physically adjacent core, for other requests.
            # Set False to reproduce the paper's "relaxed XT constraint" comparison.
            xt_avoided=True,
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
            # arrival_rate is fixed at 20.0 per explicit instruction (not to be raised
            # further); holding_time raised to 15.0. Blocking signal at this fixed rate
            # comes from the reduced fsus_per_core above, not from arrival_rate itself.
            arrival_rate=20.0,
            mean_holding_time=15.0,
            key_update_period=20.0,
            b_qc=1.0,
            b_cc_classes=(10.0, 20.0),
            b_dc_classes=(100.0, 200.0, 400.0),
            seed=2024,
        ),
        reward=RewardConfig(
            served_base=1.0,
            # Extended Eq. 12: every term below is a decision quality in [0, 1].
            # Ablation (original Eq. 12): relative_terms=False, beta_efficiency=0.05,
            # beta_fit=beta_xt=beta_compact=0.0.
            beta_qc_cc_hops=0.2,        # H_min/H_k1 -- full bonus for the shortest candidate
            beta_dc_hops=0.2,           # H_min/H_k2
            beta_efficiency=0.2,        # (eta_CC+eta_DC)/(2 eta_max); was 0.05 on raw eta (0..8)
            relative_terms=True,
            beta_fit=0.2,               # closest-fit block placement
            beta_xt=0.2,                # penalty for sterilising adjacent-core spectrum (hub core 6)
            beta_compact=0.05,          # tie-breaker: pack toward low FSU indices
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
        dqn=DQNConfig(
            learning=learning,          # same gamma / lr / hidden sizes / buffer as PPO
            replay_capacity=100_000,    # ~33 episodes of transitions
            minibatch_size=256,
            target_sync_interval=500,   # gradient steps (~16 buffer fills)
            max_grad_norm=1.0,
        ),
        exploration=ExplorationConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995),
        training=TrainingConfig(
            # Reduced from 5000: in the Sept-2026 mixed-load run the last checkpoint
            # improvement came at episode 3215 and everything after ~2000 gained only
            # 0.0005 normalized BP, so 3500 covers the useful learning with margin.
            num_episodes=3500,
            requests_per_episode=3000,      # was 1500 -- halves the per-episode BP noise floor
            checkpoint_path=os.path.join(_HERE, "checkpoints", "best_model"),
            checkpoint_bp_window=20,        # average over >=20 episodes before checkpointing (4.5)
            # Run A (mixed-load): each episode trains at a load drawn from this pool, so
            # the policy actually sees the congested regimes it is evaluated on (10..70
            # sweep). Run B (fixed-load baseline for comparison): set this to (20.0,).
            # The draw is seeded from traffic.seed, so runs stay fully reproducible.
            train_arrival_rates=(20.0, 25.0, 30.0, 35.0, 40.0, 50.0),
        ),
        evaluation=EvaluationConfig(
            # Rescaled for fsus_per_core=110 (was sized for the original 320 slots, which
            # made this range mostly saturated). Spans 0% BP up through ~42% at rate=70,
            # bracketing the fixed training rate (20.0) so its performance is visible
            # in-sweep alongside the higher-load points.
            arrival_rates=(10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0),
            requests_per_run=3000,
        ),
    )
