"""Placement-behaviour statistics for any method (greedy, no exploration).

Runs ``requests`` arrivals at one arrival rate on a fresh network with the evaluation
seed and reports, per method: BP, and the mean of each decision-quality term the
extended reward grades (route ratio H_min/H, fit F/L, XT footprint, compactness), plus
how often the DC channel lands on each data core (hub-core usage is the tell-tale for
XT-awareness). Run from the repo root, e.g.::

    python -m tools.placement_stats --config examples/example_config.py --rate 30 \
        --methods FF RF PPO --ppo-checkpoint examples/checkpoints/best_model.ppo.pt
"""

from __future__ import annotations

import argparse
import collections

from agents.heuristics import FirstFitAgent, RandomFitAgent
from request.traffic import TrafficGenerator
from training.simulator import (
    build_dqn_agent,
    build_ppo_agent,
    initialize_system,
    load_checkpoint,
    load_config_from_file,
)


def run(ctx, agent, config, rate: float, requests: int):
    ctx.env.reset()
    traffic = TrafficGenerator(config.traffic, ctx.topology)
    traffic.reset(seed=config.traffic.seed)
    traffic.configure_arrival_rate(rate)
    served = 0
    sums = collections.Counter()
    dc_cores = collections.Counter()
    for _ in range(requests):
        t = traffic.advance_time()
        ctx.env.release_and_reassign(t, agent, False)
        res = ctx.env.provision(traffic.generate_request(), agent, False).result
        if not res.served:
            continue
        served += 1
        sums["route_k1"] += res.hops_min_qc_cc / res.hops_qc_cc
        sums["route_k2"] += res.hops_min_dc / res.hops_dc
        sums["fit"] += res.fit
        sums["xt"] += res.xt
        sums["compact"] += res.compactness
        dc_cores[res.block_dc.core] += 1
    bp = 1.0 - served / requests
    means = {k: v / served for k, v in sums.items()} if served else {}
    return bp, means, dc_cores, served


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", default=["FF", "RF"])
    parser.add_argument("--ppo-checkpoint")
    parser.add_argument("--dqn-checkpoint")
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    ctx = initialize_system(config)
    print(f"rate {args.rate} | {args.requests} requests | seed {config.traffic.seed}")
    print(f"{'method':<8}{'BP':>8}{'route_k1':>10}{'route_k2':>10}{'fit':>8}{'xt':>8}{'compact':>9}  DC-core usage")
    for method in args.methods:
        if method == "FF":
            agent = FirstFitAgent()
        elif method == "RF":
            agent = RandomFitAgent(config.traffic.seed)
        elif method == "PPO":
            agent = build_ppo_agent(config, ctx)
            load_checkpoint(agent, args.ppo_checkpoint)
        elif method == "DQN":
            agent = build_dqn_agent(config, ctx)
            load_checkpoint(agent, args.dqn_checkpoint)
        else:
            raise SystemExit(f"unknown method {method}")
        bp, m, cores, served = run(ctx, agent, config, args.rate, args.requests)
        usage = " ".join(f"c{c}:{n / served:.0%}" for c, n in sorted(cores.items())) if served else "-"
        print(
            f"{method:<8}{bp:8.4f}{m.get('route_k1', 0):10.3f}{m.get('route_k2', 0):10.3f}"
            f"{m.get('fit', 0):8.3f}{m.get('xt', 0):8.3f}{m.get('compact', 0):9.3f}  {usage}"
        )


if __name__ == "__main__":
    main()
