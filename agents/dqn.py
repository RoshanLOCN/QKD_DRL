"""DQN baseline -- Mnih et al., Nature 2015 (the paper's reference [38]).

Same masked joint action space, state, reward and on-policy collection buffer as PPO.
Where the two differ is the learner:

* Q-learning targets ``r + gamma^d * max_a' Q_target(s', a')`` over the feasible
  actions of the next decision state, with a periodically synced target network.
* Transitions are stored in a persistent replay memory and minibatches are sampled
  uniformly from it (the on-policy buffer only decides *when* an update happens).
* epsilon-greedy exploration over feasible actions; greedy argmax at evaluation.

Blocked arrivals carry no action, so they are folded into the preceding decision as
extra discounted reward (a semi-MDP step): the transition that led to them earns
``r_t + sum_m gamma^m * (-X)`` and bootstraps from the next decision state with
``gamma^(m+1)``. That is what lets the blocking penalty reach the Q-function at all --
the previous implementation regressed Q-values to Monte-Carlo returns of served
requests only and never observed a -X.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from agents.base import ActionSelection, Agent
from agents.networks import build_mlp, masked_logits
from agents.ppo import RunningNormalizer
from configs.config import DQNConfig, ExplorationConfig
from training.experience import Experience


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float          # decision reward plus discounted blocked penalties that followed
    discount: float        # gamma^(1 + number of folded blocked arrivals)
    next_state: np.ndarray
    next_mask: np.ndarray


def build_transitions(experiences: Sequence[Experience], gamma: float) -> List[Transition]:
    """Pair each action-bearing experience with the next decision state, folding any
    blocked arrivals in between into the reward. The last decision has no successor
    in this batch and is dropped."""
    transitions: List[Transition] = []
    pending: Optional[Experience] = None
    reward = 0.0
    discount = 1.0
    for exp in experiences:
        if exp.has_action:
            if pending is not None:
                transitions.append(
                    Transition(
                        state=pending.state, action=int(pending.action), reward=reward,
                        discount=discount, next_state=exp.state, next_mask=np.asarray(exp.mask, dtype=bool),
                    )
                )
            pending, reward, discount = exp, float(exp.reward), gamma
        elif pending is not None:
            reward += discount * float(exp.reward)
            discount *= gamma
    return transitions


class DQNAgent(Agent):
    trainable = True

    def __init__(
        self,
        config: DQNConfig,
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
        self._q_net = build_mlp(state_size, self._learning.hidden_sizes, action_size).to(self._device)
        self._target_net = copy.deepcopy(self._q_net).to(self._device)
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=self._learning.learning_rate)
        self._epsilon = exploration.epsilon_start
        self._exploration = exploration
        self._rng = np.random.default_rng(seed)
        self._normalizer = RunningNormalizer(state_size)
        self._raw_inputs = False  # legacy checkpoints were trained on unnormalized states
        self._replay: Deque[Transition] = deque(maxlen=config.replay_capacity)
        self._grad_steps = 0
        self.last_update_stats: Dict[str, float] = {}

    @property
    def gamma(self) -> float:
        return self._learning.gamma

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        x = np.asarray(state, dtype=np.float64) if self._raw_inputs else self._normalizer.normalize(state)
        return torch.from_numpy(x.astype(np.float32)).to(self._device)

    def _batch_tensor(self, states: Sequence[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(states).astype(np.float64)
        if not self._raw_inputs:
            stacked = (stacked - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        return torch.from_numpy(stacked.astype(np.float32)).to(self._device)

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        if explore and not self._raw_inputs:
            self._normalizer.update(state)
        if explore and self._rng.random() < self._epsilon:
            return ActionSelection(action=int(self._rng.choice(np.flatnonzero(mask))))
        mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self._device)
        with torch.no_grad():
            q_values = masked_logits(self._q_net(self._state_tensor(state)), mask_t)
        return ActionSelection(action=int(torch.argmax(q_values).item()))

    def estimate_value(self, state: np.ndarray) -> float:
        """State value max_a Q(s, a) at a blocked (no-action) arrival. Returning a value
        (rather than the base-class None) is what makes the simulator buffer blocked
        arrivals for this agent, so their -X reaches ``build_transitions``."""
        with torch.no_grad():
            return float(self._q_net(self._state_tensor(state)).max().item())

    def update(self, experiences: Sequence[Experience]) -> float:
        new = build_transitions(experiences, self.gamma)
        self._replay.extend(new)
        if not new or len(self._replay) < self._config.minibatch_size:
            return 0.0

        batch_size = self._config.minibatch_size
        steps = self._learning.epochs_per_update * math.ceil(len(new) / batch_size)
        replay = list(self._replay)
        last_loss = 0.0
        q_sum = 0.0
        for _ in range(steps):
            idx = self._rng.choice(len(replay), size=batch_size, replace=len(replay) < batch_size)
            batch = [replay[i] for i in idx]
            states = self._batch_tensor([t.state for t in batch])
            next_states = self._batch_tensor([t.next_state for t in batch])
            actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self._device)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self._device)
            discounts = torch.tensor([t.discount for t in batch], dtype=torch.float32, device=self._device)
            next_masks = torch.from_numpy(np.stack([t.next_mask for t in batch])).to(self._device)

            with torch.no_grad():
                next_q = masked_logits(self._target_net(next_states), next_masks).max(dim=1).values
                # A next state with no feasible action bootstraps from zero, not -inf.
                next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
                targets = rewards + discounts * next_q

            taken = self._q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(taken, targets)
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._q_net.parameters(), self._config.max_grad_norm)
            self._optimizer.step()

            self._grad_steps += 1
            if self._grad_steps % self._config.target_sync_interval == 0:
                self._target_net.load_state_dict(self._q_net.state_dict())
            last_loss = float(loss.item())
            q_sum += float(taken.mean().item())

        self.last_update_stats = {
            "td_loss": last_loss,
            "mean_q": q_sum / steps,
            "epsilon": self._epsilon,
            "replay_size": float(len(self._replay)),
            "blocked_frac": 1.0 - sum(e.has_action for e in experiences) / max(1, len(experiences)),
        }
        return last_loss

    def decay_exploration(self) -> None:
        self._epsilon = max(self._exploration.epsilon_min, self._epsilon * self._exploration.epsilon_decay)

    def state_dict(self):
        return {
            "q_net": self._q_net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "normalizer_mean": torch.from_numpy(self._normalizer.mean.copy()),
            "normalizer_var": torch.from_numpy(self._normalizer.var.copy()),
            "normalizer_count": torch.tensor(self._normalizer.count, dtype=torch.float64),
        }

    def load_state_dict(self, state) -> None:
        self._q_net.load_state_dict(state["q_net"])
        self._target_net.load_state_dict(state.get("target_net", state["q_net"]))
        if "normalizer_mean" in state:
            self._normalizer.mean = state["normalizer_mean"].cpu().numpy().astype(np.float64)
            self._normalizer.var = state["normalizer_var"].cpu().numpy().astype(np.float64)
            self._normalizer.count = float(state["normalizer_count"].item())
            self._raw_inputs = False
        else:
            # Pre-rewrite checkpoint: the network was trained on raw states, so keep
            # feeding it raw states rather than statistics it has never seen.
            self._raw_inputs = True
            print("[DQNAgent] Legacy checkpoint (Monte-Carlo Q regression, no normalizer); evaluating on raw inputs.")
