import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent, discounted_returns
from training.experience import Experience


def test_discounted_returns():
    assert discounted_returns([1.0, 0.0, 2.0], gamma=0.5) == [1.0 + 0.5 * (0.0 + 0.5 * 2.0), 0.0 + 0.5 * 2.0, 2.0]


def _mask(size, feasible):
    mask = np.zeros(size, dtype=bool)
    for i in feasible:
        mask[i] = True
    return mask


def test_ppo_select_is_feasible_and_updates(base_config):
    action_size = base_config.action_space_size
    state_size = 12
    agent = PPOAgent(base_config.ppo, base_config.exploration, state_size, action_size, seed=1)
    mask = _mask(action_size, [0, 3])
    sel = agent.select(np.zeros(state_size, dtype=np.float32), mask, explore=True)
    assert sel.action in (0, 3)
    assert sel.log_prob is not None and sel.value is not None

    experiences = [
        Experience(np.zeros(state_size, dtype=np.float32), sel.action, 1.0, mask, sel.log_prob, sel.value)
        for _ in range(4)
    ]
    loss = agent.update(experiences)
    assert isinstance(loss, float)


def test_ppo_eval_is_deterministic_and_masked(base_config):
    action_size = base_config.action_space_size
    state_size = 12
    agent = PPOAgent(base_config.ppo, base_config.exploration, state_size, action_size, seed=2)
    mask = _mask(action_size, [1, 2])
    a = agent.select(np.ones(state_size, dtype=np.float32), mask, explore=False).action
    b = agent.select(np.ones(state_size, dtype=np.float32), mask, explore=False).action
    assert a == b and a in (1, 2)


def test_dqn_select_and_update(base_config):
    action_size = base_config.action_space_size
    state_size = 12
    agent = DQNAgent(base_config.dqn, base_config.exploration, state_size, action_size, seed=3)
    mask = _mask(action_size, [0, action_size - 1])
    sel = agent.select(np.zeros(state_size, dtype=np.float32), mask, explore=False)
    assert sel.action in (0, action_size - 1)
    experiences = [
        Experience(np.zeros(state_size, dtype=np.float32), sel.action, -1.0, mask) for _ in range(4)
    ]
    assert isinstance(agent.update(experiences), float)


def test_dqn_exploration_decay(base_config):
    agent = DQNAgent(base_config.dqn, base_config.exploration, 12, base_config.action_space_size, seed=4)
    start = agent._epsilon
    agent.decay_exploration()
    assert agent._epsilon < start


def test_ppo_decay_exploration_is_a_no_op(base_config):
    # PPO's exploration is the policy's own categorical sampling, not an epsilon-greedy
    # schedule (docs/training_diagnosis.md 4.3) -- decay_exploration() is inherited from
    # Agent's no-op default, and PPOAgent carries no _epsilon at all.
    agent = PPOAgent(base_config.ppo, base_config.exploration, 12, base_config.action_space_size, seed=4)
    assert not hasattr(agent, "_epsilon")
    agent.decay_exploration()
