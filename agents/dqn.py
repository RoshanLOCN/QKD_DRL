"""DQN baseline (Section VIII, Section X).

Same masked joint action space as PPO; a Q-network is regressed toward the discounted
returns of the on-policy buffer (value regression over the joint action space), and
actions are chosen greedily w.r.t. masked Q-values, with epsilon-greedy exploration.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from agents.base import ActionSelection, Agent
from agents.networks import build_mlp, masked_logits
from agents.ppo import discounted_returns
from configs.config import DQNConfig, ExplorationConfig
from training.experience import Experience


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
        self._learning = config.learning
        self._q_net = build_mlp(state_size, self._learning.hidden_sizes, action_size).to(self._device)
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=self._learning.learning_rate)
        self._epsilon = exploration.epsilon_start
        self._exploration = exploration
        self._rng = np.random.default_rng(seed)

    @property
    def gamma(self) -> float:
        return self._learning.gamma

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self._device)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self._device)
        with torch.no_grad():
            q_values = masked_logits(self._q_net(state_t), mask_t)
            if explore and self._rng.random() < self._epsilon:
                feasible = np.flatnonzero(mask)
                action = int(self._rng.choice(feasible))
            else:
                action = int(torch.argmax(q_values).item())
        return ActionSelection(action=action)

    def update(self, experiences: Sequence[Experience]) -> float:
        if not experiences:
            return 0.0
        states = torch.from_numpy(np.stack([e.state for e in experiences]).astype(np.float32)).to(self._device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self._device)
        returns = torch.tensor(
            discounted_returns([e.reward for e in experiences], self._learning.gamma),
            dtype=torch.float32,
            device=self._device,
        )

        last_loss = 0.0
        for _ in range(self._learning.epochs_per_update):
            q_values = self._q_net(states)
            taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(taken, returns)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            last_loss = float(loss.item())
        return last_loss

    def decay_exploration(self) -> None:
        self._epsilon = max(self._exploration.epsilon_min, self._epsilon * self._exploration.epsilon_decay)

    def state_dict(self):
        return {"q_net": self._q_net.state_dict()}

    def load_state_dict(self, state) -> None:
        self._q_net.load_state_dict(state["q_net"])
