

"""PPO agent -- the primary learning agent (Section VIII-D).

Actor-critic over the masked joint action space. Exploration follows the documents'
epsilon-greedy schedule on top of the policy distribution; updates use the clipped
policy-gradient objective on the actor and value regression on the critic, over discounted
returns computed from the on-policy buffer. Every numeric hyperparameter comes from config.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import ActionSelection, Agent
from agents.networks import build_mlp, masked_logits
from configs.config import ExplorationConfig, PPOConfig
from env import mask
from training.experience import Experience


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    """G_t = r_t + gamma * G_{t+1} over the buffer order."""
    returns: List[float] = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


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
        self._epsilon = exploration.epsilon_start
        self._exploration = exploration
        self._rng = np.random.default_rng(seed)

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self._device)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).to(self._device)

        with torch.no_grad():
            raw_logits = self._actor(state_t)
            logits = masked_logits(raw_logits, mask_t)
            dist = Categorical(logits=logits)

        value = float(self._critic(state_t).item())
        
        if explore and self._rng.random() < self._epsilon:
            feasible = np.flatnonzero(mask)
            action = int(self._rng.choice(feasible))
        elif explore:
            action = int(dist.sample().item())
        else:
            action = int(torch.argmax(logits).item())
            
        log_prob = float(dist.log_prob(torch.tensor(action, device=self._device)).item())
        return ActionSelection(action=action, log_prob=log_prob, value=value)

    def update(self, experiences: Sequence[Experience]) -> float:
        if not experiences:
            return 0.0
            
        states = torch.from_numpy(np.stack([e.state for e in experiences]).astype(np.float32)).to(self._device)
        masks = torch.from_numpy(np.stack([e.mask for e in experiences]).astype(bool)).to(self._device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=self._device)
        old_log_probs = torch.tensor(
            [float(e.log_prob) for e in experiences], dtype=torch.float32, device=self._device
        )
        old_values = torch.tensor(
            [float(e.value) for e in experiences], dtype=torch.float32, device=self._device
        )
        returns = torch.tensor(
            discounted_returns([e.reward for e in experiences], self._learning.gamma),
            dtype=torch.float32,
            device=self._device,
        )
        advantages = returns - old_values

        last_loss = 0.0
        for _ in range(self._learning.epochs_per_update):
            logits = masked_logits(self._actor(states), masks)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(ratio, 1.0 - self._config.clip_epsilon, 1.0 + self._config.clip_epsilon) * advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            values = self._critic(states).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns)
            loss = policy_loss + self._config.value_loss_coef * value_loss

            self._optimizer.zero_grad()
            loss.backward()
            
            # --- ADD THESE TWO LINES TO FIX THE CRASH ---
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=0.5)
            # --------------------------------------------
            
            self._optimizer.step()
            last_loss = float(loss.item())
            
        return last_loss

    def decay_exploration(self) -> None:
        self._epsilon = max(self._exploration.epsilon_min, self._epsilon * self._exploration.epsilon_decay)

    def state_dict(self):
        return {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()}

    def load_state_dict(self, state) -> None:
        self._actor.load_state_dict(state["actor"])
        self._critic.load_state_dict(state["critic"])







































# """PPO agent -- the primary learning agent (Section VIII-D).

# Actor-critic over the masked joint action space. Exploration follows the documents'
# epsilon-greedy schedule on top of the policy distribution; updates use the clipped
# policy-gradient objective on the actor and value regression on the critic, over discounted
# returns computed from the on-policy buffer. Every numeric hyperparameter comes from config.
# """

# from __future__ import annotations

# from typing import List, Sequence

# import numpy as np
# import torch
# from torch import nn
# from torch.distributions import Categorical

# from agents.base import ActionSelection, Agent
# from agents.networks import build_mlp, masked_logits
# from configs.config import ExplorationConfig, PPOConfig
# from env import mask
# from training.experience import Experience


# def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
#     """G_t = r_t + gamma * G_{t+1} over the buffer order."""
#     returns: List[float] = [0.0] * len(rewards)
#     running = 0.0
#     for t in range(len(rewards) - 1, -1, -1):
#         running = rewards[t] + gamma * running
#         returns[t] = running
#     return returns


# class PPOAgent(Agent):
#     trainable = True

#     def __init__(
#         self,
#         config: PPOConfig,
#         exploration: ExplorationConfig,
#         state_size: int,
#         action_size: int,
#         seed: int,
#     ) -> None:
#         torch.manual_seed(seed)
#         self._config = config
#         self._learning = config.learning
#         self._actor = build_mlp(state_size, self._learning.hidden_sizes, action_size)
#         self._critic = build_mlp(state_size, self._learning.hidden_sizes, 1)
#         self._optimizer = torch.optim.Adam(
#             list(self._actor.parameters()) + list(self._critic.parameters()),
#             lr=self._learning.learning_rate,
#         )
#         self._epsilon = exploration.epsilon_start
#         self._exploration = exploration
#         self._rng = np.random.default_rng(seed)

#     def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
#         # state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
#         # mask_t = torch.from_numpy(np.asarray(mask, dtype=bool))
#         # with torch.no_grad():
#         #     # logits = masked_logits(self._actor(state_t), mask_t)
#         #     # dist = Categorical(logits=logits)
#         #     raw_logits = self._actor(state_t)

#         #     print("Valid actions:", int(mask_t.sum()))
#         #     print("Total actions:", mask_t.numel())

#         #     if mask_t.sum() == 0:
#         #         raise RuntimeError("No feasible actions generated by environment!")

#         #     logits = masked_logits(raw_logits, mask_t)
#         #     dist = Categorical(logits=logits)
#             # print("State NaN:", torch.isnan(state_t).any().item())
#             # print("State Inf:", torch.isinf(state_t).any().item())

#             # print("Raw logits NaN:", torch.isnan(raw_logits).any().item())
#             # print("Raw logits Inf:", torch.isinf(raw_logits).any().item())

#             # print("Valid actions:", mask_t.sum().item(), "/", mask_t.numel())

#             # logits = masked_logits(raw_logits, mask_t)

#             # print("Masked logits NaN:", torch.isnan(logits).any().item())
#             # print("Masked logits Inf:", torch.isinf(logits).any().item())
#             # dist = Categorical(logits=logits)

#         state_t = torch.from_numpy(np.asarray(state, dtype=np.float32))
#         mask_t = torch.from_numpy(np.asarray(mask, dtype=bool))

#         with torch.no_grad():
#             print("\n--------------------")
#             # print("State NaN:", torch.isnan(state_t).any().item())
#             print("State Inf:", torch.isinf(state_t).any().item())
#             print("State min:", float(state_t.min()))
#             print("State max:", float(state_t.max()))

#             raw_logits = self._actor(state_t)

#             print("Raw logits NaN:", torch.isnan(raw_logits).any().item())
#             print("Raw logits Inf:", torch.isinf(raw_logits).any().item())
#             print("Raw logits min:", float(raw_logits.min()))
#             print("Raw logits max:", float(raw_logits.max()))

#             print("Valid actions:", int(mask_t.sum()))
#             print("Total actions:", mask_t.numel())

#             logits = masked_logits(raw_logits, mask_t)

#             print("Masked logits NaN:", torch.isnan(logits).any().item())
#             print("Masked logits Inf:", torch.isinf(logits).any().item())

#             dist = Categorical(logits=logits)


#         value = float(self._critic(state_t).item())
#         if explore and self._rng.random() < self._epsilon:
#             feasible = np.flatnonzero(mask)
#             action = int(self._rng.choice(feasible))
#         elif explore:
#             action = int(dist.sample().item())
#         else:
#             action = int(torch.argmax(logits).item())
#         log_prob = float(dist.log_prob(torch.tensor(action)).item())
#         return ActionSelection(action=action, log_prob=log_prob, value=value)


#     def update(self, experiences: Sequence[Experience]) -> float:
#         if not experiences:
#             return 0.0
#         states = torch.from_numpy(np.stack([e.state for e in experiences]).astype(np.float32))
#         masks = torch.from_numpy(np.stack([e.mask for e in experiences]).astype(bool))
#         actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
#         old_log_probs = torch.tensor([float(e.log_prob) for e in experiences], dtype=torch.float32)
#         old_values = torch.tensor([float(e.value) for e in experiences], dtype=torch.float32)
#         returns = torch.tensor(
#             discounted_returns([e.reward for e in experiences], self._learning.gamma),
#             dtype=torch.float32,
#         )
#         advantages = returns - old_values

#         last_loss = 0.0
#         for _ in range(self._learning.epochs_per_update):
#             logits = masked_logits(self._actor(states), masks)
#             dist = Categorical(logits=logits)
#             new_log_probs = dist.log_prob(actions)
#             ratio = torch.exp(new_log_probs - old_log_probs)
#             surrogate_1 = ratio * advantages
#             surrogate_2 = torch.clamp(ratio, 1.0 - self._config.clip_epsilon, 1.0 + self._config.clip_epsilon) * advantages
#             policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
#             values = self._critic(states).squeeze(-1)
#             value_loss = nn.functional.mse_loss(values, returns)
#             loss = policy_loss + self._config.value_loss_coef * value_loss

#             self._optimizer.zero_grad()
#             loss.backward()
#             self._optimizer.step()
#             last_loss = float(loss.item())
#         return last_loss

#     def decay_exploration(self) -> None:
#         self._epsilon = max(self._exploration.epsilon_min, self._epsilon * self._exploration.epsilon_decay)

#     def state_dict(self):
#         return {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()}

#     def load_state_dict(self, state) -> None:
#         self._actor.load_state_dict(state["actor"])
#         self._critic.load_state_dict(state["critic"])
