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
        Experience(np.full(state_size, float(i), dtype=np.float32), sel.action, 1.0, mask) for i in range(8)
    ]
    # A blocked arrival in the middle folds into the preceding decision's reward.
    experiences.insert(3, Experience(np.ones(state_size, dtype=np.float32), None, -1.0, None, has_action=False))
    loss = agent.update(experiences)
    assert isinstance(loss, float) and loss > 0.0
    assert agent.last_update_stats["replay_size"] == 7  # 8 decisions -> 7 transitions
    assert agent.last_update_stats["blocked_frac"] == pytest.approx(1 / 9)


def test_dqn_build_transitions_folds_blocked_arrivals(base_config):
    from agents.dqn import build_transitions

    s = lambda v: np.full(3, float(v), dtype=np.float32)
    m = np.ones(2, dtype=bool)
    gamma = 0.5
    experiences = [
        Experience(s(0), 0, 1.0, m),
        Experience(s(1), None, -1.0, None, has_action=False),
        Experience(s(2), None, -1.0, None, has_action=False),
        Experience(s(3), 1, 1.0, m),
        Experience(s(4), 0, 1.0, m),
    ]
    t = build_transitions(experiences, gamma)
    assert len(t) == 2
    # decision 0 earns its own reward plus two discounted blocked penalties, then
    # bootstraps from decision 3's state with gamma^3.
    assert t[0].reward == 1.0 + gamma * -1.0 + gamma**2 * -1.0
    assert t[0].discount == gamma**3
    assert float(t[0].next_state[0]) == 3.0
    assert t[1].reward == 1.0 and t[1].discount == gamma and float(t[1].next_state[0]) == 4.0


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
