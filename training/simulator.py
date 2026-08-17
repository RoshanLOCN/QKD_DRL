"""Simulator -- system initialization, Algorithm 2 training loop, and evaluation.

This is the top-level runnable. It builds the whole system from a :class:`SimulationConfig`
(config-only; no defaults) and a topology file, trains PPO or DQN with best-BP
checkpointing, and evaluates PPO/DQN/FF/RF over an arrival-rate sweep.
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib.util
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import pandas as pd
from datetime import datetime, timezone
from agents.base import Agent
from agents.dqn import DQNAgent
from agents.heuristics import FirstFitAgent, RandomFitAgent
from agents.ppo import PPOAgent
from configs.config import SimulationConfig
from core.resource_grid import SlotTable
from core.routing import RouteCaches, precompute_routes
from core.topology import Topology, load_topology
from env.environment import JointRRCSAEnvironment
from env.reward import compute_reward
from request.traffic import TrafficGenerator
from training.experience import Experience, ExperienceBuffer
from training.metrics import EpisodeMetrics, ResourceUtilization, resource_utilization

PPO = "PPO"
DQN = "DQN"
FF = "FF"
RF = "RF"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Simulator running on: {DEVICE}")

@dataclass(frozen=True)
class SystemContext:
    config: SimulationConfig
    topology: Topology
    caches: RouteCaches
    slot_table: SlotTable
    env: JointRRCSAEnvironment


def initialize_system(config: SimulationConfig) -> SystemContext:
    """Phase 0: load topology, allocate the slot table, precompute routes, build env."""
    topology = load_topology(config.topology_path)
    
    if config.network.cores_per_link < 3:
        raise ValueError("network.cores_per_link must be >= 3 (QC, CC, and >=1 data core)")
        
    slot_table = SlotTable(topology.num_links, config.network.cores_per_link, config.network.fsus_per_core)
    
    print(f"Precomputing routes for {topology.num_nodes} nodes...")
    caches = precompute_routes(topology, config.routing.k1, config.routing.k2, config.quantum.r_qkd_km)
    print("Done precomputing routes.")
    
    env = JointRRCSAEnvironment(config, topology, caches, slot_table)
    return SystemContext(config=config, topology=topology, caches=caches, slot_table=slot_table, env=env)


def _run_episode(
    ctx: SystemContext,
    agent: Agent,
    traffic: TrafficGenerator,
    num_requests: int,
    explore: bool,
    buffer: Optional[ExperienceBuffer],
    buffer_size: Optional[int],
) -> EpisodeMetrics:
    metrics = EpisodeMetrics(gamma=agent.gamma if agent.gamma is not None else 0.99)
    for _ in range(num_requests):
        current_time = traffic.advance_time()
        ctx.env.release_and_reassign(current_time, agent, explore)
        qlr = traffic.generate_request()
        outcome = ctx.env.provision(qlr, agent, explore)
        reward = compute_reward(outcome.result, ctx.config.reward)
        metrics.record(outcome.result.served, reward)

        if buffer is not None and agent.trainable:
            if outcome.action_taken:
                assert outcome.state is not None and outcome.selection is not None and outcome.mask is not None
                buffer.append(
                    Experience(
                        state=outcome.state,
                        action=outcome.selection.action,
                        reward=reward,
                        mask=outcome.mask,
                        log_prob=outcome.selection.log_prob,
                        value=outcome.selection.value,
                    )
                )
            elif outcome.state is not None:
                # Blocked arrival: no action to attribute a policy gradient to, but the
                # state and a critic value are still real -- buffering it (when the agent
                # supports it) is what lets the -X reward reach the learner at all. See
                # docs/training_diagnosis.md section 2.
                blocked_value = agent.estimate_value(outcome.state)
                if blocked_value is not None:
                    buffer.append(
                        Experience(
                            state=outcome.state,
                            action=None,
                            reward=reward,
                            mask=None,
                            value=blocked_value,
                            has_action=False,
                        )
                    )

            if buffer_size is not None and buffer.is_full():
                agent.update(buffer.items())
                buffer.clear()
                agent.decay_exploration()
    return metrics


def train(config: SimulationConfig, agent: Agent, buffer_size: int, checkpoint_path: str) -> Dict:
    """Algorithm 2: train ``agent`` with best-BP checkpointing. Returns run history."""
    ctx = initialize_system(config)
    traffic = TrafficGenerator(config.traffic, ctx.topology)
    buffer = ExperienceBuffer(buffer_size)
    # Checkpointing on the rolling mean of the last `checkpoint_bp_window` episodes,
    # not the raw per-episode BP: at this task's blocking rates, a single episode's BP
    # is dominated by binomial sampling noise (see docs/training_diagnosis.md section 1),
    # so checkpointing on it saves whichever episode got the luckiest arrival sequence
    # rather than the best policy.
    bp_window: deque = deque(maxlen=config.training.checkpoint_bp_window)
    best_rolling_bp = math.inf
    history: List[Dict] = []

    # Display training load
    erlang_load = config.traffic.arrival_rate * config.traffic.mean_holding_time
    print("=" * 70)
    print(f"TRAINING LOAD PARAMETERS:")
    print(f"  - Arrival Rate (lambda) : {config.traffic.arrival_rate}")
    print(f"  - Mean Holding Time (mu): {config.traffic.mean_holding_time}")
    print(f"  - Total Erlang Load     : {erlang_load:.2f} Erlangs")
    print(f"  - Requests per Episode  : {config.training.requests_per_episode}")
    print("=" * 70)
    print(f"Starting training loop across {config.training.num_episodes} episodes...")

    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_every = max(1, min(100, config.training.num_episodes // 10))

    for episode in range(config.training.num_episodes):
        ctx.env.reset()
        traffic.reset(seed=config.traffic.seed + episode)
        
        metrics = _run_episode(
            ctx, agent, traffic, config.training.requests_per_episode, True, buffer, buffer_size
        )
        bp_window.append(metrics.blocking_probability)
        rolling_bp = sum(bp_window) / len(bp_window)

        history.append(
            {
                "episode": episode,
                "bp": metrics.blocking_probability,
                "rolling_bp": rolling_bp,
                "G_t": metrics.discounted_cumulative_reward,
            }
        )

        if episode % save_every == 0 or episode == config.training.num_episodes - 1:
            write_training_history(
                history,
                agent="train",
                path=os.path.join(os.path.dirname(checkpoint_path) or ".", "training_results.xlsx"),
                run_time=run_time,
            )

        # INCLUDES G_t ONLY DURING TRAINING
        log_msg = (
            f"Episode {episode + 1:02d}/{config.training.num_episodes} | "
            f"BP: {metrics.blocking_probability:.4f} | RollingBP({len(bp_window)}): {rolling_bp:.4f} | "
            f"G_t: {metrics.discounted_cumulative_reward:.4f}"
        )

        diagnostics = getattr(agent, "last_update_stats", None)
        if diagnostics:
            log_msg += (
                f" | policy_loss={diagnostics.get('policy_loss', 0.0):.4f} "
                f"value_loss={diagnostics.get('value_loss', 0.0):.4f} "
                f"entropy={diagnostics.get('entropy', 0.0):.4f} "
                f"approx_kl={diagnostics.get('approx_kl', 0.0):.4f} "
                f"clip_frac={diagnostics.get('clip_fraction', 0.0):.3f} "
                f"action_frac={diagnostics.get('action_fraction', 0.0):.3f}"
            )

        # Only checkpoint once the window is full: a partial window is exactly the
        # single-noisy-episode problem this is meant to fix.
        if len(bp_window) == bp_window.maxlen and rolling_bp < best_rolling_bp:
            best_rolling_bp = rolling_bp
            save_checkpoint(agent, checkpoint_path)
            log_msg += f" [New Best Rolling BP! Saved -> {checkpoint_path}]"

        print(log_msg)

    print(f"Training complete. Overall Best Rolling BP: {best_rolling_bp:.4f}")
    return {"best_bp": best_rolling_bp, "history": history, "checkpoint": checkpoint_path}


def evaluate(
    config: SimulationConfig, methods: Sequence[str], checkpoint_base: str
) -> Dict[Tuple[str, float], Tuple[float, ResourceUtilization]]:
    """Evaluate each method over the arrival-rate sweep (deterministic, no exploration)."""
    ctx = initialize_system(config)
    traffic = TrafficGenerator(config.traffic, ctx.topology)
    results: Dict[Tuple[str, float], Tuple[float, ResourceUtilization]] = {}

    holding_time = config.traffic.mean_holding_time
    print("=" * 70)
    print("EVALUATION / TESTING LOADS:")
    for rate in config.evaluation.arrival_rates:
        load = rate * holding_time
        print(f"  - Arrival Rate (lambda): {rate:5.1f} | Erlang Load: {load:6.1f} Erlangs")
    print("=" * 70)

    for method in methods:
        try:
            agent = _build_eval_agent(method, config, ctx, checkpoint_base)
        except FileNotFoundError as exc:
            print(f"Skipping {method}: {exc}")
            continue

        for rate in config.evaluation.arrival_rates:
            ctx.env.reset()
            traffic.reset(seed=config.traffic.seed)
            traffic.configure_arrival_rate(rate)
            metrics = _run_episode(
                ctx, agent, traffic, config.evaluation.requests_per_run, False, None, None
            )
            ru = resource_utilization(ctx.slot_table, ctx.env.partition, ctx.topology.num_links)
            results[(method, rate)] = (metrics.blocking_probability, ru)

    return results


def _build_eval_agent(method: str, config: SimulationConfig, ctx: SystemContext, base: str) -> Agent:
    action_size = config.action_space_size
    if method == PPO:
        checkpoint = checkpoint_path_for(base, PPO)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"No checkpoint found for {PPO} at '{checkpoint}'. Train the PPO agent first with: "
                "python -m training.simulator train --config <config> --agent PPO"
            )
        agent = PPOAgent(
            config.ppo, config.exploration, ctx.env.state_size, action_size, config.traffic.seed, device=DEVICE
        )
        load_checkpoint(agent, checkpoint)
        return agent
    if method == DQN:
        checkpoint = checkpoint_path_for(base, DQN)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"No checkpoint found for {DQN} at '{checkpoint}'. Train the DQN agent first with: "
                "python -m training.simulator train --config <config> --agent DQN"
            )
        agent = DQNAgent(
            config.dqn, config.exploration, ctx.env.state_size, action_size, config.traffic.seed, device=DEVICE
        )
        load_checkpoint(agent, checkpoint)
        return agent
    if method == FF:
        return FirstFitAgent()
    if method == RF:
        return RandomFitAgent(config.traffic.seed)
    raise ValueError(f"unknown evaluation method {method!r}")


def build_ppo_agent(config: SimulationConfig, ctx: SystemContext) -> PPOAgent:
    return PPOAgent(
        config.ppo, config.exploration, ctx.env.state_size, config.action_space_size, config.traffic.seed,
        device=DEVICE,
    )


def build_dqn_agent(config: SimulationConfig, ctx: SystemContext) -> DQNAgent:
    return DQNAgent(
        config.dqn, config.exploration, ctx.env.state_size, config.action_space_size, config.traffic.seed,
        device=DEVICE,
    )


def checkpoint_path_for(base: str, method: str) -> str:
    return f"{base}.{method.lower()}.pt"


def append_to_excel(new_rows: pd.DataFrame, path: str) -> pd.DataFrame:
    """Append ``new_rows`` to the Excel file at ``path``, creating it if it doesn't exist.

    Excel has no native incremental-append (unlike a CSV you could open in 'a' mode), so
    this reads whatever is already there, concatenates the new rows underneath, and
    rewrites the whole file. Used so that running train/evaluate more than once (e.g.
    once per agent, or once per experiment) keeps accumulating into the SAME two files
    rather than each run overwriting the last one's results.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        if os.path.exists(path):
            existing = pd.read_excel(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_excel(path, index=False)
        return combined
    except PermissionError as exc:
        raise PermissionError(
            f"Could not write Excel results to '{path}'. Close any open workbook for this file and try again."
        ) from exc


def write_training_history(history: List[Dict], agent: str, path: str, run_time: Optional[str] = None) -> pd.DataFrame:
    """Write a snapshot of the current training history to Excel.

    This is used during long training loops so progress is preserved even if the process is
    interrupted before the full run completes. Unlike evaluation exports, training is treated
    as a fresh run snapshot rather than an append to avoid duplicate rows from repeated writes.
    """
    if not history:
        return pd.DataFrame()

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.insert(0, "agent", agent)
    history_df.insert(1, "run_timestamp", run_time or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        history_df.to_excel(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not write training history to '{path}'. Close any open Excel workbook for this file and try again."
        ) from exc
    return history_df


def save_checkpoint(agent: Agent, path: str) -> None:
    if not hasattr(agent, "state_dict"):
        raise TypeError("agent does not support checkpointing")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(agent.state_dict(), path)


def load_checkpoint(agent: Agent, path: str) -> None:
    if not hasattr(agent, "load_state_dict"):
        raise TypeError("agent does not support checkpointing")
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    agent.load_state_dict(state_dict)  


def load_config_from_file(path: str) -> SimulationConfig:
    spec = importlib.util.spec_from_file_location("_user_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import config from {path!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "build_config"):
        raise ValueError(f"config file {path!r} must define build_config()")

    return module.build_config()


def _main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SS-EON QKD DRL simulator")
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--config", required=True, help="Python file exposing build_config()")
    parser.add_argument("--agent", choices=[PPO, DQN], default=PPO, help="agent to train")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[PPO, FF, RF],
        help="methods to evaluate; RL baselines are only included when their checkpoint exists",
    )
    args = parser.parse_args(argv)

    config = load_config_from_file(args.config)

    if args.command == "train":
        ctx = initialize_system(config)
        if args.agent == PPO:
            agent: Agent = build_ppo_agent(config, ctx)
            buffer_size = config.ppo.learning.buffer_size
        else:
            agent = build_dqn_agent(config, ctx)
            buffer_size = config.dqn.learning.buffer_size
            
        summary = train(config, agent, buffer_size, checkpoint_path_for(config.training.checkpoint_path, args.agent))
        print(f"best BP={summary['best_bp']:.4f} checkpoint={summary['checkpoint']}")
        
        # Training summary retains G_t log
        for entry in summary["history"]:
            print(f"  episode {entry['episode']:2d}: BP={entry['bp']:.4f} G_t={entry['G_t']:.4f}")

        # --- EXPORT LOGIC ---
        # One fixed file for ALL training data, across every agent and every run --
        # not one file per agent. Each run's rows are tagged with the agent and a
        # timestamp so separate runs/agents stay distinguishable inside the same file
        # instead of overwriting each other.
        run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_dir = os.path.dirname(config.training.checkpoint_path) or "."
        training_path = os.path.join(out_dir, "training_results.xlsx")
        history_df = write_training_history(summary["history"], args.agent, training_path, run_time)
        print(f"Training results saved to {training_path} ({len(history_df)} rows)")

    else:
        results = evaluate(config, args.methods, config.training.checkpoint_path)
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (NO CUMULATIVE REWARD DISPLAYED):")
        print("=" * 70)
        
        # Prepare a list to collect data for pandas
        export_data = []

        for (method, rate), (bp, ru) in sorted(results.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            erlang = rate * config.traffic.mean_holding_time
            print(
                f"Method: {method:4s} | Rate (lambda): {rate:4.1f} | Load: {erlang:5.1f} Erlangs | "
                f"BP: {bp:.4f} | RU(agg={ru.aggregate:.4f} qc={ru.qc:.4f} cc={ru.cc:.4f} dc={ru.dc:.4f})"
            )
            
            # Append row to export list
            export_data.append({
                "Method": method,
                "Arrival Rate": rate,
                "Erlang Load": erlang,
                "Blocking Probability": bp,
                "Aggregate RU": ru.aggregate,
                "QC RU": ru.qc,
                "CC RU": ru.cc,
                "DC RU": ru.dc
            })

        # --- EXPORT LOGIC ---
        # One fixed file for ALL testing/evaluation data, accumulated across every
        # evaluate() run (not overwritten each time) -- same pattern as training above.
        run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        df = pd.DataFrame(export_data)
        df.insert(0, "run_timestamp", run_time)

        out_dir = os.path.dirname(config.training.checkpoint_path) or "."
        evaluation_path = os.path.join(out_dir, "evaluation_results.xlsx")
        append_to_excel(df, evaluation_path)
        print(f"Evaluation results appended to {evaluation_path}")


if __name__ == "__main__":
    _main()