"""EXAMPLE configuration -- the caller-supplied values live here, not in the library.

Every number below is an example the *user* provides; the library itself ships no
defaults. The values mirror the paper's Section X examples so the system is runnable
out of the box; edit freely for your own study. Run with, e.g.::

    python -m sseon_qkd_drl.training.simulator train --config sseon_qkd_drl/examples/example_config.py --agent ppo
    python -m sseon_qkd_drl.training.simulator evaluate --config sseon_qkd_drl/examples/example_config.py
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
        gamma=0.95,                 # Increased to 0.99 for better long-term reward stability
        learning_rate=1e-5,         # Lowered to 3e-4 for gradual, stable learning
        hidden_sizes=(128, 128),
        buffer_size=64,             
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
            r_qkd_km=1500.0,        # Kept high so Tokyo12 links are valid
        ),
        routing=RoutingConfig(k1=5, k2=5),
        candidates=CandidateConfig(i_qc=5, i_cc=5, i_dc=5),
        traffic=TrafficConfig(
            arrival_rate=5.0,       # Set to 5.0 to create moderate, learnable contention
            mean_holding_time=6.0,  
            key_update_period=20.0, 
            b_qc=1.0,
            b_cc_classes=(10.0, 20.0),        
            b_dc_classes=(100.0, 200.0, 400.0),
            seed=2024,
        ),
        reward=RewardConfig(
            served_base=1.0, beta_qc_cc_hops=0.2, beta_dc_hops=0.2, beta_efficiency=0.05
        ),
        ppo=PPOConfig(learning=learning, clip_epsilon=0.2, value_loss_coef=0.5),
        dqn=DQNConfig(learning=learning),
        exploration=ExplorationConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995),
        training=TrainingConfig(
            num_episodes=5000,
            requests_per_episode=1500,
            checkpoint_path=os.path.join(_HERE, "checkpoints", "best_model"),
        ),
        evaluation=EvaluationConfig(arrival_rates=(20.0, 40.0, 60.0, 80.0, 100.0), requests_per_run=1500),
    )









# """EXAMPLE configuration -- the caller-supplied values live here, not in the library.

# Every number below is an example the *user* provides; the library itself ships no
# defaults. The values mirror the paper's Section X examples so the system is runnable
# out of the box; edit freely for your own study. Run with, e.g.::

#     python -m sseon_qkd_drl.training.simulator train --config sseon_qkd_drl/examples/example_config.py --agent ppo
#     python -m sseon_qkd_drl.training.simulator evaluate --config sseon_qkd_drl/examples/example_config.py
# """

# from __future__ import annotations

# import os

# from configs.config import (
#     CandidateConfig,
#     DQNConfig,
#     EvaluationConfig,
#     ExplorationConfig,
#     LearningConfig,
#     ModulationConfig,
#     ModulationFormat,
#     NetworkConfig,
#     PPOConfig,
#     QuantumConfig,
#     RewardConfig,
#     RoutingConfig,
#     SimulationConfig,
#     TrafficConfig,
#     TrainingConfig,
# )

# _HERE = os.path.dirname(os.path.abspath(__file__))


# def build_config() -> SimulationConfig:
#     learning = LearningConfig(
#         gamma=0.95,                 # Section X
#         learning_rate=1e-3,         # Section X
#         hidden_sizes=(128, 128),
#         buffer_size=64,             # N
#         epochs_per_update=4,
#     )
#     return SimulationConfig(
#         topology_path=os.path.join(_HERE, "sample_topology.json"),
#         sentinel_value=-1.0,
#         network=NetworkConfig(
#             cores_per_link=5,       
#             fsus_per_core=320,      # S = 320 (Section X)
#             fsu_width_ghz=12.5,     # Delta f = 12.5 GHz (Section X)
#             core_qc_index=0,
#             core_cc_index=1,
#         ),
#         modulation=ModulationConfig(
#             formats=(               # distance-adaptive table, consistent with [4] (Section X)
#                 ModulationFormat("BPSK", 1.0, 4000.0),
#                 ModulationFormat("QPSK", 2.0, 2000.0),
#                 ModulationFormat("8QAM", 3.0, 1000.0),
#                 ModulationFormat("16QAM", 4.0, 500.0),
#             ),
#             guard_band_fsus=1,      # G
#         ),
#         quantum=QuantumConfig(
#             f_qc_min=2,             # fixed QC footprint
#             r_qkd_km=150.0,         # ~100-150 km QKD reach (Section X)
#         ),
#         routing=RoutingConfig(k1=3, k2=3),
#         candidates=CandidateConfig(i_qc=3, i_cc=3, i_dc=3),
#         traffic=TrafficConfig(
#             arrival_rate=8.0,
#             mean_holding_time=10.0,
#             key_update_period=20.0,  # T
#             b_qc=1.0,
#             b_cc_classes=(10.0, 20.0),        # CC classes set lower (control channel)
#             b_dc_classes=(100.0, 200.0, 400.0),
#             seed=2024,
#         ),
#         reward=RewardConfig(
#             served_base=1.0, beta_qc_cc_hops=0.2, beta_dc_hops=0.2, beta_efficiency=0.05
#         ),
#         ppo=PPOConfig(learning=learning, clip_epsilon=0.2, value_loss_coef=0.5),
#         dqn=DQNConfig(learning=learning),
#         exploration=ExplorationConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995),
#         training=TrainingConfig(
#             num_episodes=50,
#             requests_per_episode=500,
#             checkpoint_path=os.path.join(_HERE, "checkpoints", "best_model"),
#         ),
#         evaluation=EvaluationConfig(arrival_rates=(4.0, 8.0, 12.0, 16.0), requests_per_run=1000),
#     )
