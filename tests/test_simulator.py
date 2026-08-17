import pytest

pytest.importorskip("torch")

from training import simulator
from training.simulator import (
    FF,
    PPO,
    RF,
    build_ppo_agent,
    checkpoint_path_for,
    evaluate,
    initialize_system,
    train,
)


def test_train_then_evaluate_end_to_end(base_config):
    ctx = initialize_system(base_config)
    agent = build_ppo_agent(base_config, ctx)
    summary = train(
        base_config,
        agent,
        buffer_size=base_config.ppo.learning.buffer_size,
        checkpoint_path=checkpoint_path_for(base_config.training.checkpoint_path, PPO),
    )
    assert 0.0 <= summary["best_bp"] <= 1.0
    assert len(summary["history"]) == base_config.training.num_episodes

    results = evaluate(base_config, methods=[PPO, FF, RF], checkpoint_base=base_config.training.checkpoint_path)
    for rate in base_config.evaluation.arrival_rates:
        for method in (PPO, FF, RF):
            bp, ru = results[(method, rate)]
            assert 0.0 <= bp <= 1.0
            assert 0.0 <= ru.aggregate <= 1.0


def test_missing_training_checkpoint_raises_helpful_error(base_config):
    with pytest.raises(FileNotFoundError, match="train.*DQN|checkpoint.*DQN|No checkpoint.*DQN"):
        evaluate(base_config, methods=["DQN"], checkpoint_base=base_config.training.checkpoint_path)


def test_training_history_excel_snapshot_is_created(tmp_path, base_config):
    history = [{"episode": 0, "bp": 0.1, "G_t": 1.0}, {"episode": 1, "bp": 0.2, "G_t": 2.0}]
    out_path = tmp_path / "training_results.xlsx"

    df = simulator.write_training_history(history, "PPO", str(out_path), "2024-01-01T00:00:00+00:00")

    assert out_path.exists()
    assert len(df) == 2
    assert list(df.columns)[:3] == ["agent", "run_timestamp", "episode"]


def test_initialize_system_builds_caches(base_config):
    ctx = initialize_system(base_config)
    assert ctx.env.state_size > 0
    assert ctx.caches.candidate_routes("A", "C")
