"""PPO agent -- the primary learning agent (Section VIII-D).

Actor-critic over the masked joint action space. Exploration is the policy's own
categorical sampling (no bolted-on epsilon-greedy -- see ``select``). Updates use the
clipped policy-gradient objective on the actor and value regression on the critic, with
GAE(lambda) advantages computed over the *full* arrival trajectory -- including blocked
arrivals, which have a state and a critic value but no action -- so a blocking penalty
propagates back into the placement decisions that caused it. The policy loss and its
importance ratio are restricted to the action-bearing steps only, since a blocked step has
no action to attribute a policy gradient to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import ActionSelection, Agent
from agents.networks import build_mlp, masked_logits
from configs.config import ExplorationConfig, PPOConfig
from training.experience import Experience


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    """G_t = r_t + gamma * G_{t+1} over the buffer order. Kept for DQN's Monte-Carlo
    Q-regression and for direct testing; PPO itself now uses ``_gae`` below."""
    returns: List[float] = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _gae(rewards: Sequence[float], values: Sequence[float], gamma: float, lam: float):
    """GAE(lambda) advantages and bootstrapped returns over a contiguous trajectory that
    may mix action-bearing and blocked (no-action) steps -- both kinds carry a reward and
    a critic value, which is all GAE needs. There is no true terminal state (arrivals are
    a continuing process), so the tail bootstraps with the buffer's own last value instead
    of 0; this also avoids the previous Monte-Carlo return's truncation bias at the
    arbitrary buffer-size boundary."""
    t_count = len(rewards)
    advantages = [0.0] * t_count
    gae = 0.0
    for t in range(t_count - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < t_count else values[t]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class RunningNormalizer:
    """Welford running mean/std for observation normalization.

    Raw state magnitudes span roughly [-1, 300+] (occupancy counts and FSU indices next to
    a -1 padding sentinel); feeding that directly into a ReLU MLP lets the large-scale
    features dominate the first layer. Updated only while exploring (training); frozen at
    evaluation so a fixed policy's behaviour doesn't drift with eval-time statistics.
    """

    def __init__(self, size: int, eps: float = 1e-4) -> None:
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        new_count = self.count + 1.0
        delta = x - self.mean
        new_mean = self.mean + delta / new_count
        new_var = (self.var * self.count + delta * (x - new_mean)) / new_count
        self.mean, self.var, self.count = new_mean, new_var, new_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent(Agent):
    trainable = True

    def __init__(
        self,
        config: PPOConfig,
        exploration: ExplorationConfig,
        state_size: int,
        action_size: int,
        seed: int,
        device: Optional[torch.device] = None,
    ) -> None:
        torch.manual_seed(seed)
        self._device = device if device is not None else torch.device("cpu")
        self._config = config
        self._learning = config.learning
        self._actor = build_mlp(state_size, self._learning.hidden_sizes, action_size).to(self._device)
        self._critic = build_mlp(state_size, self._learning.hidden_sizes, 1).to(self._device)
        self._optimizer = torch.optim.Adam(
            list(self._actor.parameters()) + list(self._critic.parameters()),
            lr=self._learning.learning_rate,
        )
        self._rng = np.random.default_rng(seed)
        self._normalizer = RunningNormalizer(state_size)
        # exploration is accepted for interface parity with DQNAgent (simulator.py builds
        # both agents uniformly) but unused: PPO's exploration is the policy distribution
        # itself, not an epsilon-greedy schedule -- see the note in ``select``.
        self.last_update_stats: Dict[str, float] = {}

    @property
    def gamma(self) -> float:
        return self._learning.gamma

    def _normalized_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        normalized = self._normalizer.normalize(state)
        return torch.from_numpy(normalized.astype(np.float32)).to(self._device)

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        if explore:
            self._normalizer.update(state)
        state_t = self._normalized_state_tensor(state)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self._device)

        with torch.no_grad():
            raw_logits = self._actor(state_t)
            logits = masked_logits(raw_logits, mask_t)
            dist = Categorical(logits=logits)
            value = float(self._critic(state_t).item())

            # dist.sample() *is* PPO's exploration. An epsilon-greedy branch here would
            # draw the action from a uniform behaviour policy while still recording the
            # target policy's log_prob, corrupting the importance ratio PPO's clipped
            # objective relies on -- see training_diagnosis.md section 4.3.
            if explore:
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(logits).item())

            log_prob = float(dist.log_prob(torch.tensor(action, device=self._device)).item())

        return ActionSelection(action=action, log_prob=log_prob, value=value)

    def estimate_value(self, state: np.ndarray) -> float:
        """Critic value at a blocked (no-action) state, for GAE bootstrapping."""
        state_t = self._normalized_state_tensor(state)
        with torch.no_grad():
            return float(self._critic(state_t).item())

    def update(self, experiences: Sequence[Experience]) -> float:
        if not experiences:
            return 0.0

        rewards = [e.reward for e in experiences]
        values = [float(e.value) for e in experiences]
        advantages, returns = _gae(rewards, values, self._learning.gamma, self._config.gae_lambda)

        states_np = np.stack([self._normalizer.normalize(e.state) for e in experiences]).astype(np.float32)
        all_states = torch.from_numpy(states_np).to(self._device)
        all_returns = torch.tensor(returns, dtype=torch.float32, device=self._device)
        n_value = all_states.shape[0]
        value_mb = min(self._config.minibatch_size, n_value)

        action_rows = [i for i, e in enumerate(experiences) if e.has_action]
        n_policy = len(action_rows)
        if n_policy:
            action_states = all_states[action_rows]
            action_masks = torch.from_numpy(
                np.stack([experiences[i].mask for i in action_rows]).astype(bool)
            ).to(self._device)
            action_actions = torch.tensor(
                [experiences[i].action for i in action_rows], dtype=torch.long, device=self._device
            )
            action_old_log_probs = torch.tensor(
                [float(experiences[i].log_prob) for i in action_rows], dtype=torch.float32, device=self._device
            )
            action_advantages = torch.tensor(
                [advantages[i] for i in action_rows], dtype=torch.float32, device=self._device
            )
            if self._config.normalize_advantages and n_policy > 1:
                action_advantages = (
                    action_advantages - action_advantages.mean()
                ) / (action_advantages.std() + 1e-8)
            policy_mb = min(self._config.minibatch_size, n_policy)

        epoch_stats: List[Dict[str, float]] = []
        stop_early = False

        for _epoch in range(self._learning.epochs_per_update):
            value_losses = []
            perm = self._rng.permutation(n_value)
            for start in range(0, n_value, value_mb):
                idx = torch.from_numpy(perm[start:start + value_mb]).to(self._device)
                pred = self._critic(all_states[idx]).squeeze(-1)
                value_loss = nn.functional.mse_loss(pred, all_returns[idx])

                self._optimizer.zero_grad()
                (self._config.value_loss_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=self._config.max_grad_norm)
                self._optimizer.step()
                value_losses.append(float(value_loss.item()))

            policy_losses, entropies, kls, clip_fracs = [], [], [], []
            if n_policy:
                perm_p = self._rng.permutation(n_policy)
                for start in range(0, n_policy, policy_mb):
                    idx = torch.from_numpy(perm_p[start:start + policy_mb]).to(self._device)
                    logits = masked_logits(self._actor(action_states[idx]), action_masks[idx])
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(action_actions[idx])
                    ratio = torch.exp(new_log_probs - action_old_log_probs[idx])
                    mb_advantages = action_advantages[idx]
                    surrogate_1 = ratio * mb_advantages
                    surrogate_2 = torch.clamp(
                        ratio, 1.0 - self._config.clip_epsilon, 1.0 + self._config.clip_epsilon
                    ) * mb_advantages
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                    # Manual entropy avoids NaN from 0 * (-inf) in dist.entropy() at
                    # masked-out (infeasible) actions, since Categorical.entropy()
                    # multiplies probs by the (unmasked -> -inf) logits internally.
                    probs = dist.probs
                    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).mean()

                    loss = policy_loss - self._config.entropy_coef * entropy

                    self._optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._actor.parameters(), max_norm=self._config.max_grad_norm)
                    self._optimizer.step()

                    with torch.no_grad():
                        approx_kl = float((action_old_log_probs[idx] - new_log_probs).mean().item())
                        clip_frac = float(
                            (torch.abs(ratio - 1.0) > self._config.clip_epsilon).float().mean().item()
                        )
                    policy_losses.append(float(policy_loss.item()))
                    entropies.append(float(entropy.item()))
                    kls.append(approx_kl)
                    clip_fracs.append(clip_frac)

                mean_kl = sum(kls) / len(kls)
                if self._config.target_kl is not None and mean_kl > self._config.target_kl:
                    stop_early = True

            epoch_stats.append(
                {
                    "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
                    "policy_loss": sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
                    "entropy": sum(entropies) / len(entropies) if entropies else 0.0,
                    "approx_kl": sum(kls) / len(kls) if kls else 0.0,
                    "clip_fraction": sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0,
                }
            )
            if stop_early:
                break

        final = epoch_stats[-1]
        total_loss = final["policy_loss"] + self._config.value_loss_coef * final["value_loss"]
        self.last_update_stats = {
            **final,
            "loss": total_loss,
            "epochs_run": float(len(epoch_stats)),
            "buffer_size": float(len(experiences)),
            "action_fraction": n_policy / len(experiences) if experiences else 0.0,
        }
        return total_loss

    def state_dict(self):
        return {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()}

    def load_state_dict(self, state) -> None:
        self._actor.load_state_dict(state["actor"])
        self._critic.load_state_dict(state["critic"])
