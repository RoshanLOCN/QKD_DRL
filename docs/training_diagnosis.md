# PPO Training Diagnosis — SS-EON QKD DRL

**Date:** 2026-08-17
**Log analysed:** `train_qkd.txt` — 5000 episodes, PPO, λ=5.0, μ=6.0 (30 Erlangs), 1500 requests/episode
**Code analysed:** `training/simulator.py`, `agents/ppo.py`, `agents/dqn.py`, `env/environment.py`, `env/reward.py`, `training/metrics.py`, `configs/config.py`, `examples/example_config.py`

---

## TL;DR

1. **The training is not unstable — it is flat.** The episode-to-episode wobble in BP is 99.2% explained by the binomial sampling noise of 1500 Bernoulli trials. There is no divergence, no reward collapse, no oscillation.
2. **The root cause of the flatness: no negative reward ever reaches the learner.** Every blocking event occurs *before* the agent is consulted, so the experience buffer contains only positive rewards. The agent is not optimizing blocking probability at all.
3. **Several hyperparameters are badly mis-sized** (notably `buffer_size=64` against a 3125-action space, and `learning_rate=1e-5`), and several standard PPO hyperparameters are absent from the config entirely.
4. Retuning hyperparameters alone will not move BP. The credit-assignment defect in (2) must be fixed first.

---

## 1. Evidence: the training is flat, not unstable

### 1.1 The observed variance is pure measurement noise

| Quantity | Value |
|---|---|
| Observed BP std across 5000 episodes | 0.00313 |
| Binomial noise floor, √(p(1−p)/n) with p=0.0152, n=1500 | 0.00316 |
| **Ratio (observed / floor)** | **0.992** |

With BP ≈ 1.52% over 1500 requests, each episode measures roughly 22.8 blocked events. The spread you see between consecutive episodes is what you would get from a *fixed, frozen* policy. There is essentially zero excess variance attributable to the learning process.

**Consequence:** "instability" is not a tunable property here. The metric cannot be made smoother by changing hyperparameters — only by averaging more samples (more requests per episode, or a rolling mean across episodes).

### 1.2 No improvement over 5000 episodes

| Episode window | BP mean | BP std | BP min | G_t mean | G_t std |
|---|---|---|---|---|---|
| 1–500 | 0.0149 | 0.0032 | 0.0047 | 134.115 | 3.341 |
| 501–1000 | 0.0152 | 0.0031 | 0.0073 | 134.755 | 3.064 |
| 1001–1500 | 0.0151 | 0.0031 | 0.0073 | 134.172 | 2.570 |
| 1501–2000 | 0.0151 | 0.0031 | 0.0067 | 136.576 | 3.595 |
| 2001–2500 | 0.0154 | 0.0032 | 0.0080 | 138.286 | 2.991 |
| 2501–3000 | 0.0152 | 0.0030 | 0.0073 | 136.182 | 2.520 |
| 3001–3500 | 0.0152 | 0.0032 | 0.0047 | 136.785 | 2.814 |
| 3501–4000 | 0.0153 | 0.0032 | 0.0067 | 136.596 | 2.466 |
| 4001–4500 | 0.0154 | 0.0030 | 0.0080 | 136.544 | 2.470 |
| 4501–5000 | 0.0153 | 0.0033 | 0.0060 | 136.069 | 2.652 |

- **BP linear trend: +0.00008 per 1000 episodes** — i.e. flat, if anything marginally worse.
- **G_t linear trend: +0.473 per 1000 episodes** — a real but tiny rise (134.1 → 136.1, ≈ +1.5%).
- **corr(BP, G_t) = −0.227** — the reward the agent maximizes is only weakly related to the metric you care about.

Overall: BP mean 0.0152, std 0.0031, min 0.0047, max 0.0280. G_t mean 136.008, std 3.125.

### 1.3 The checkpoint is a noise artifact

```
Training complete. Overall Best BP: 0.0047
```

`New Best BP` fired only 4 times, at episodes **1, 4, 5, and 18**. The best value (0.0047) was reached at **episode 18** and never beaten in the following 4982 episodes.

0.0047 sits **3.4σ below the mean** of 0.0152 given a per-episode std of 0.00313. It is a lucky draw, not a better policy. `training/simulator.py:150` selects the checkpoint on a single noisy episode, so the saved `best_model.ppo.pt` is whichever episode got the most favourable arrival sequence — with no relationship to policy quality.

### 1.4 Statistical power of the current setup

To resolve even a 10% relative BP improvement (0.0152 → 0.0137, a difference of 0.0015) at 2σ confidence, you need to average over

```
k > (2 × 0.00313 / 0.0015)²  ≈  18 episodes
```

A single-episode BP value carries almost no information. Any plot must use a rolling mean of ≥50 episodes.

---

## 2. Root cause: the learner never sees a blocking event

### 2.1 The code path

`env/environment.py:135-137`:

```python
if not mask.any():
    return ProvisionOutcome(result=CommitResult.blocked(qlr.request_id), action_taken=False)
```

`training/simulator.py:86`:

```python
if buffer is not None and agent.trainable and outcome.action_taken:
```

An experience is buffered **only when `action_taken` is True**. Separately, `env/environment.py:176-185` re-checks the three chosen blocks against the slot table and commits atomically; because the mask was built from the same candidate set against the same unchanged state within one synchronous call, **a masked-feasible action always commits successfully**.

Therefore: `action_taken == True` ⟹ `served == True`. Always.

### 2.2 Measured across the full load range

Instrumented with `RandomFitAgent`, 1500 requests, seed 2024:

| λ | Erlangs | BP | blocked | **blocked with `action_taken=True`** |
|---|---|---|---|---|
| 5 | 30 | 0.0120 | 18 | **0** |
| 20 | 120 | 0.0120 | 18 | **0** |
| 40 | 240 | 0.0193 | 29 | **0** |
| 60 | 360 | 0.0253 | 38 | **0** |
| 100 | 600 | 0.0440 | 66 | **0** |

**100% of blocking is pre-action at every load from 30 to 600 Erlangs.** This is structural, not a property of the light training load.

### 2.3 What the agent actually optimizes

Measured reward distribution over the buffered (i.e. served-only) experiences:

| Quantity | Value |
|---|---|
| Reward range in the buffer | `[1.15, 1.70]` |
| Mean served reward | 1.381 |
| Spread among served rewards | 0.485 |
| Number of −1 rewards ever buffered | **0** |

From `env/reward.py`, a served QLR yields

```
1.0 + 0.2·(1/H_qc_cc) + 0.2·(1/H_dc) + 0.05·(η_cc + η_dc)
```

Since the `served_base = 1.0` term is constant across every buffered sample, the critic absorbs it and the only learnable advantage signal is the 0.485-wide shaping term: **shorter routes and higher-order modulation**. Nothing in the gradient refers to blocking.

**This explains the log exactly:** G_t rises slightly (the agent *is* learning to pick shorter routes and better modulation — the only signal available), while BP stays flat (blocking is absent from the objective as far as the gradient is concerned), and the two correlate only weakly (−0.227).

### 2.4 Why this cannot be fixed by tuning

No value of `learning_rate`, `gamma`, `buffer_size`, or `entropy_coef` can teach a policy to avoid an outcome that never appears in its training data. **Section 3 must be read as secondary to this section.**

### 2.5 The principled fix

A blocked arrival with an empty mask has no action to attribute the −X to, so it cannot become a policy-gradient sample directly. But blocking is *caused* by earlier placement decisions, so the penalty must propagate backward through the return.

Keep the arrival trajectory **contiguous**:

- compute discounted returns over **all** arrivals, including empty-mask ones, so a −X flows back into the returns of the placement decisions that caused the congestion;
- restrict the **policy loss** to the action-bearing steps only.

Currently the empty-mask rewards are computed (`training/simulator.py:83`) and counted toward BP, then discarded.

This also requires γ and the rollout length to cover the consequence horizon — see 3.1 and 4.2.

---

## 3. Hyperparameter audit

### 3.1 Sizing facts that drive the verdicts

| Quantity | Value | Source |
|---|---|---|
| Action space `|A| = k1·i_qc·i_cc·k2·i_dc` | 5·5·5·5·5 = **3125** | `configs/config.py:277` |
| State dimension | **512** | `env.state_size` |
| Feasible actions per decision (measured) | mean **1943.5**, median 2150, min 25, max 3125 | mask density 62.2% |
| Actor parameters | **481,920** (of which **400,000** in the 128→3125 output layer) | (512·128)+(128·128)+(128·3125) |
| PPO updates per episode | 1500 / 64 = **23.4** | |
| Total updates over the run | **117,187** | |
| γ=0.95 effective horizon | **20 requests** | 1/(1−γ) |
| Mean connection lifetime | μ·λ = **30 arrivals** | |
| Uniform entropy over 1943 actions | ln(1943) = **7.57 nats** | |

### 3.2 Hyperparameters present in the config

| Parameter | Value | Verdict |
|---|---|---|
| `buffer_size` (N) | **64** | **Worst offender.** 64 samples against a 3125-action space is 2% coverage per update. Standard PPO rollouts are 2048–8192. |
| `learning_rate` | **1e-5** | **~30× too low.** The comment at `examples/example_config.py:39` says "Lowered to 3e-4" but the value is 1e-5. Glacial for a 482k-parameter network. |
| `gamma` | **0.95** | **Too low.** Horizon 20 requests vs 30-arrival mean connection lifetime — the consequence of a packing decision has decayed to nothing before it materialises. The comment at `example_config.py:38` says "Increased to 0.99"; the value is 0.95. |
| `hidden_sizes` | **(128, 128)** | **Too small.** 3125 action logits produced from a 128-d bottleneck. Use ≥(256,256), or factor the joint action head into per-dimension heads. |
| `epochs_per_update` | 4 | Value is reasonable, but it is 4 **full-batch** passes over 64 samples with no minibatching → each rollout is overfitted. |
| `epsilon_start` | **1.0** | **Should not exist for PPO.** See 4.3. |
| `epsilon_min` | **0.05** | **Should not exist for PPO.** Permanently corrupts 5% of samples. |
| `epsilon_decay` | **0.995/update** | Reaches the 0.05 floor after 598 updates ≈ **episode 25.5** of 5000. |
| `requests_per_episode` | 1500 | Too few to resolve BP improvements (noise floor 0.0032). |
| `clip_epsilon` | 0.2 | ✅ Correct, standard. |
| `value_loss_coef` | 0.5 | ✅ Correct, standard. |
| `num_episodes` | 5000 | ✅ Fine — 117k updates is ample. |
| `sentinel_value` | −1.0 | Distinguishable from real features (which are ≥0), but see 4.1 on scaling. |

### 3.3 Hyperparameters missing from the config entirely

`PPOConfig` (`configs/config.py:189`) contains only `learning`, `clip_epsilon`, `value_loss_coef`. The following standard PPO knobs have no representation:

| Missing | Consequence |
|---|---|
| `entropy_coef` | No exploration pressure in the objective. `loss = policy_loss + 0.5·value_loss` only (`agents/ppo.py:116`). Nothing resists premature determinism over a 1943-action distribution. |
| `gae_lambda` | No GAE at all. Advantage is raw Monte-Carlo return minus V (`agents/ppo.py:103`) — maximum variance. |
| `minibatch_size` | Full-batch updates only. |
| `max_grad_norm` | Hardcoded to 0.5 at `agents/ppo.py:122-123`, outside config. |
| advantage normalization | Absent. Raw advantages reach magnitude ≈29 (γ=0.95, r≈1.45 → max return 29). |
| `target_kl` / early stopping | Absent — no guard against a destructive update. |
| LR schedule / annealing | Absent. |

`DQNConfig` (`configs/config.py:203`) contains **only** `learning`. Missing: `target_update_freq`, `replay_capacity`, `batch_size`, `train_freq`. See 4.6.

---

## 4. Additional defects found

### 4.1 State inputs are unnormalized

Measured over 400 real states:

| Quantity | Value |
|---|---|
| Range | `[−1, 315]` |
| Mean / std | 20.224 / 48.723 |
| Features with max > 1.0 | **512 of 512** |
| Features with max > 10 | 327 |
| Features with max > 100 | 125 |
| Per-feature std | min 0.365, max **93.58** |
| Constant features | 0 |

Feeding raw magnitudes spanning two orders of magnitude into a ReLU MLP means the large-scale occupancy features dominate the first layer, and at `lr=1e-5` the network has little chance to rescale them. **Fix:** running mean/std normalization of observations, or divide occupancy counts by `fsus_per_core` in the encoder.

### 4.2 Returns are computed per-buffer with no bootstrap, on a non-episodic process

`agents/ppo.py:27-34` — `discounted_returns` initialises `running = 0.0` and walks backward from the end of the buffer. Arrivals never terminate: this is a *continuing* task, not an episodic one.

Consequences:
- `returns[63]` equals `rewards[63]` alone — the value target for the last transition of every rollout is systematically truncated.
- Every target depends on where the arbitrary 64-step boundary happened to fall.
- The horizon is capped at 64 steps regardless of γ.

**Fix:** bootstrap the tail with `V(s_T)`, and use GAE(λ) rather than raw MC returns.

### 4.3 ε-greedy is bolted onto PPO with the wrong `log_prob`

`agents/ppo.py:74-82`:

```python
if explore and self._rng.random() < self._epsilon:
    feasible = np.flatnonzero(mask)
    action = int(self._rng.choice(feasible))      # behaviour = uniform over feasible
elif explore:
    action = int(dist.sample().item())
...
log_prob = float(dist.log_prob(...).item())        # recorded = log π(a), the TARGET policy
```

When the ε branch fires, the action comes from a uniform distribution but the recorded `log_prob` is that of the policy network. PPO's ratio `exp(new_log_prob − old_log_prob)` is then not an importance ratio between the behaviour and target policies at all.

- For the first ~26 episodes ε ≈ 1.0, so essentially **all** early training data is mislabeled off-policy data.
- Afterwards ε floors at 0.05, permanently corrupting 5% of samples — and these are precisely the samples with the most extreme (wrong) ratios, which clipping then handles incorrectly.

**Fix:** delete the ε-greedy path. `dist.sample()` *is* PPO's exploration; use `entropy_coef` to control it.

### 4.4 Metric γ does not match training γ, and is hardcoded

`training/metrics.py:50`:

```python
gamma: float = 0.99
```

`EpisodeMetrics()` is constructed with no arguments (`training/simulator.py:77`), so G_t uses **γ=0.99** while training uses **γ=0.95**. The value never comes from config.

Additional consequence: Σ 0.99^j over 1500 steps = **100.0** exactly, so

```
G_t ≈ 100 × (mean reward per request) = 100 × 1.345 = 134.5
```

against an observed mean of 136.0 — G_t is simply a rescaled mean reward. And since 0.99^460 ≈ 0.01, **99% of G_t is determined by the first ~460 requests** of each episode, i.e. the low-occupancy transient right after `ctx.env.reset()`. This is why G_t barely correlates with BP.

### 4.5 Best-BP checkpointing selects on noise

`training/simulator.py:150`. See 1.3. **Fix:** run a deterministic greedy evaluation on a fixed seed, averaged over ≥20 episodes, and checkpoint on that.

### 4.6 The DQN baseline is not a DQN

`agents/dqn.py:56-76` regresses Q(s,a) onto the **Monte-Carlo discounted returns** of a 64-step on-policy buffer (it imports `discounted_returns` from `agents/ppo.py`). Absent:

- no target network,
- no Bellman bootstrap (`r + γ·max_a' Q(s',a')`),
- no replay memory (the on-policy buffer is cleared after each update),
- no gradient clipping (PPO has it; DQN does not).

This is Monte-Carlo Q-regression, not DQN. As a paper baseline it is not comparable to a published DQN result.

### 4.7 Train/evaluate load mismatch

Training runs at λ=5.0 → **30 Erlangs** (`example_config.py:71`). Evaluation sweeps λ ∈ {20, 40, 60, 80, 100} → **120–600 Erlangs** (`example_config.py:89`). That is 4–20× the training load; the state distribution at evaluation has little overlap with training.

### 4.8 The task is too easy to generate gradient

At 30 Erlangs, BP = 1.5% — 98.5% of decisions succeed regardless of what the agent picks. Note also from 2.2 that even at 600 Erlangs BP only reaches 4.4%, so reaching a 5–20% blocking regime requires either λ > 100 or a smaller resource grid (`fsus_per_core`, `cores_per_link`).

### 4.9 Minor: wasted autograd graph on every decision

`agents/ppo.py:72`:

```python
value = float(self._critic(state_t).item())
```

This sits **outside** the `with torch.no_grad()` block above it. `.item()` detaches, so there is no correctness bug, but a graph is built and discarded on every one of ~7.5M decisions (1500 × 5000, plus reassignments).

### 4.10 No PPO diagnostics are logged

The log contains only BP and G_t. Absent: policy loss, value loss, entropy, approximate KL, clip fraction, explained variance, current ε. Without these it is impossible to distinguish "no gradient signal" from "policy collapsed" from "updates too large" — which is why the flat curve read as instability.

---

## 5. Recommended configuration

```python
# LearningConfig
gamma                 = 0.99      # horizon 100 >> 30-arrival connection lifetime
learning_rate         = 3e-4      # + linear anneal to 0 over training
hidden_sizes          = (256, 256)
buffer_size           = 2048      # rollout length; was 64
epochs_per_update     = 4

# PPOConfig — new fields required
clip_epsilon          = 0.2       # unchanged
value_loss_coef       = 0.5       # unchanged
entropy_coef          = 0.01      # NEW
gae_lambda            = 0.95      # NEW
minibatch_size        = 256       # NEW
max_grad_norm         = 0.5       # NEW (move out of agents/ppo.py:122)
normalize_advantages  = True      # NEW
target_kl             = 0.02      # NEW (optional early stop)

# ExplorationConfig
epsilon_start         = 0.0       # delete the ε-greedy path for PPO entirely
```

Plus:

- normalize observations (running mean/std, or scale occupancy by `fsus_per_core`);
- read the metric γ from config instead of hardcoding 0.99 in `training/metrics.py:50`;
- train at a load where BP is 5–15%, and close the gap to the evaluation sweep;
- checkpoint on a fixed-seed greedy eval averaged over ≥20 episodes;
- log a 100-episode rolling BP mean alongside entropy, approx-KL and clip fraction.

---

## 6. Suggested fix order

| # | Change | Why first |
|---|---|---|
| 1 | Contiguous-trajectory returns so blocked arrivals enter the return chain (2.5) | Without this nothing else can affect BP |
| 2 | Bootstrap + GAE (4.2) | Correctness bug; also required for #1 to propagate |
| 3 | Delete ε-greedy (4.3) | Correctness bug; free to fix |
| 4 | Observation normalization (4.1) | Cheap, large effect |
| 5 | Resize `buffer_size`, `learning_rate`, `gamma`, `hidden_sizes` (3.2) | Now they can matter |
| 6 | `entropy_coef`, advantage normalization, minibatching (3.3) | Stabilises the larger updates from #5 |
| 7 | Eval-based checkpointing (4.5) + diagnostics logging (4.10) | So the next run is interpretable |
| 8 | Raise training load / align with eval sweep (4.7, 4.8) | Needs #1–#6 in place to show benefit |
| 9 | Rewrite the DQN baseline properly (4.6) | Independent of the PPO path |

After #1–#7, a ~300-episode run is enough to confirm the BP curve moves before committing to another 5000.

---

## 7. Reproducing the measurements

All figures in this document come from the log plus the two probes below.

**Log statistics** (windowed means, trends, noise floor):

```bash
python3 - <<'EOF'
import re, math, statistics as st
bp, g = [], []
for l in open('train_qkd.txt'):
    m = re.match(r'Episode \d+/\d+ \| BP: ([\d.]+) \| G_t: ([-\d.]+)', l)
    if m: bp.append(float(m.group(1))); g.append(float(m.group(2)))
p = st.mean(bp)
print('episodes           :', len(bp))
print('BP mean/std        : %.4f / %.5f' % (p, st.stdev(bp)))
print('binomial noise floor: %.5f' % math.sqrt(p*(1-p)/1500))
print('ratio              : %.3f' % (st.stdev(bp)/math.sqrt(p*(1-p)/1500)))
print('first500 / last500 : %.4f / %.4f' % (st.mean(bp[:500]), st.mean(bp[-500:])))
EOF
```

**Pre-action blocking across loads** (the finding in 2.2) — run from the project root:

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
from examples.example_config import build_config
from core.topology import load_topology
from core.routing import precompute_routes
from core.resource_grid import SlotTable
from env.environment import JointRRCSAEnvironment
from request.traffic import TrafficGenerator
from agents.heuristics import RandomFitAgent

c = build_config()
t = load_topology(c.topology_path)
ca = precompute_routes(t, c.routing.k1, c.routing.k2, c.quantum.r_qkd_km)
print('%6s %8s %8s %9s %20s' % ('lambda','Erlang','BP','blocked','blocked_with_action'))
for rate in (5.0, 20.0, 40.0, 60.0, 100.0):
    st_ = SlotTable(t.num_links, c.network.cores_per_link, c.network.fsus_per_core)
    env = JointRRCSAEnvironment(c, t, ca, st_)
    tr = TrafficGenerator(c.traffic, t); tr.reset(seed=2024); tr.configure_arrival_rate(rate)
    ag = RandomFitAgent(2024)
    served = tot = withact = 0
    for _ in range(1500):
        env.release_and_reassign(tr.advance_time(), ag, True)
        out = env.provision(tr.generate_request(), ag, True); tot += 1
        if out.result.served: served += 1
        elif out.action_taken: withact += 1
    print('%6.1f %8.1f %8.4f %9d %20d'
          % (rate, rate*c.traffic.mean_holding_time, 1-served/tot, tot-served, withact))
EOF
```

Expect `blocked_with_action = 0` on every row. If any row is non-zero, the conclusion in section 2 needs revisiting.

---

## 8. Claims worth verifying independently

The diagnosis rests on three checks that are cheap to confirm:

1. **`blocked_with_action == 0`** (2.2) — that a masked-feasible action always commits. Cross-read `env/environment.py:137` against `training/simulator.py:86`. If a committed action can fail, the "no negative rewards in the buffer" conclusion weakens.
2. **The BP noise floor** (1.1) — observed 0.00313 vs binomial 0.00316. This is what establishes that the wobble is not instability.
3. **The config comment/value mismatches** (`example_config.py:38-39`) and the **hardcoded `gamma=0.99`** (`training/metrics.py:50`) — direct reads, no inference involved.
