# SS-EON QKD DRL

A faithful implementation of **DRL-based joint Routing, Core, and Spectrum Assignment
(RRCSA) for QKD-secured Spectrally-Spatially Elastic Optical Networks (SS-EON)**, from
the three specification documents (IEEE paper, QKD-ON→SS-EON migration note, algorithmic
flow guide).

Every Quantum Lightpath Request (QLR) is provisioned as **three coordinated channels** —
Quantum (QC), Classical Control (CC), Data (DC) — resolved **jointly** in a single
five-part action `(k1, i_QC, i_CC, k2, i_DC)`, all-or-nothing. QC and CC each keep a
dedicated core (1-D closest-fit search); DC draws from a shared data-core pool `𝒟`
(2-D core×spectrum closest-fit search). A PPO agent is the primary learner; DQN, First-Fit
and Random-Fit are baselines.

## Two ground rules baked into this codebase

1. **No embedded numbers.** Every parameter the papers give only as an example
   (`C, S, Δf`, the modulation table, `R_QKD`, `K1/K2/I*`, reward constants,
   hyperparameters, traffic classes, …) is a **required** config field with **no default**.
   Nothing runs until you supply a complete `SimulationConfig`.
2. **No embedded topology.** The network graph is loaded from an external JSON file —
   no node/link/distance data is invented in code. Supply your own Tokyo12 dataset.

## Install

```bash
cd sseon_qkd_drl
pip install -e .          # networkx, numpy, torch
pip install -e ".[test]"  # + pytest
```

## Topology file format

```json
{
  "name": "Tokyo12",
  "nodes": ["n0", "n1", "..."],
  "links": [
    {"source": "n0", "target": "n1", "distance_km": 50.0}
  ]
}
```

Links are undirected; `distance_km` must be > 0. See `examples/sample_topology.json`
(an **example**, not the real Tokyo12 data).

## Configuration

You supply a Python file that exposes `build_config() -> SimulationConfig`. See
`examples/example_config.py` for a complete, runnable example (its values mirror the
paper's Section X examples and are yours to edit).

## Run

```bash
# Train PPO (or --agent DQN), with best-BP checkpointing
python -m sseon_qkd_drl.training.simulator train \
    --config sseon_qkd_drl/examples/example_config.py --agent PPO

# Evaluate an arrival-rate sweep for PPO / DQN / FF / RF
python -m sseon_qkd_drl.training.simulator evaluate \
    --config sseon_qkd_drl/examples/example_config.py --methods PPO FF RF
```

(For `evaluate` with `PPO`/`DQN`, train that agent first so its checkpoint exists.)

## Tests

```bash
pytest
```

Tests use a small synthetic fixture topology (`tests/fixtures/mini_topology.json`) — test
data only, not a claim about Tokyo12. Torch-dependent tests are skipped automatically if
torch is not installed.

## Module map (spec artifact → code)

| Spec artifact | Code |
|---|---|
| Network model `G(V,E,C,S,M)`, core partition (Eq. 2) | `core/topology.py`, `core/resource_grid.py`, `configs/config.py` |
| QLR tuple (Eq. 3) | `request/qlr.py` |
| `F^{QC}=F_min` (Eq. 4), `F^{CC},F^{DC}` (Eqs. 5–6) | `core/modulation.py` |
| 1-D QC/CC & 2-D DC closest-fit (Sec. VI-B) | `core/blocks.py` |
| Nested state (Eq. 10), action encode/decode (Eq. 11) | `env/state.py` |
| Joint action mask | `env/mask.py` |
| Reward (Eq. 12) | `env/reward.py` |
| Algorithm 1 (atomic joint commit, safety re-check) | `env/environment.py` |
| Algorithm 2 (training loop, best-BP checkpoint) | `training/simulator.py` |
| PPO / DQN / FF / RF | `agents/` |
| Release + key-update reassignment | `env/environment.py`, `env/registry.py` |
| BP / RU (aggregate + per-role) / AR (Eqs. 13–14) | `training/metrics.py` |

## Decisions the documents left open

- **Key-update reassignment infeasibility** (Step 8.1): the docs re-run the full joint
  placement at `t_upd` but do not say what happens if no placement is feasible at refresh
  time. This implementation frees the old placements, attempts reassignment, and drops the
  QLR if none is feasible; reassignment neither stores experience nor counts toward BP
  (keeping the pseudocode's "one experience per arrival" density). See the comment in
  `env/environment.py:release_and_reassign`.
- **Standard PPO/DQN mechanics** not numerically fixed by the docs (critic-baseline
  advantage, clipped-surrogate form, discounted-return targets) are textbook; every *number*
  still comes from config.

## Out of scope (paper's Future Work, Sec. XV)

Crosstalk-aware constraints, physical-layer-grounded QC modeling, and `𝒟`-pool
defragmentation are intentionally **not** implemented.
