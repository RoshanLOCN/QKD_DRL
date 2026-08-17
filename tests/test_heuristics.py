import numpy as np
import pytest

from agents.heuristics import FirstFitAgent, RandomFitAgent


def test_first_fit_picks_lowest_feasible_index():
    mask = np.array([False, False, True, False, True])
    assert FirstFitAgent().select(np.zeros(1), mask, explore=False).action == 2


def test_random_fit_stays_within_feasible():
    mask = np.array([False, True, False, True, True])
    agent = RandomFitAgent(seed=7)
    feasible = set(np.flatnonzero(mask).tolist())
    for _ in range(20):
        assert agent.select(np.zeros(1), mask, explore=True).action in feasible


def test_empty_mask_raises():
    empty = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError):
        FirstFitAgent().select(np.zeros(1), empty, explore=False)
