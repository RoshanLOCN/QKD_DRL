"""Print per-link spectrum occupancy for FF/RF allocations to verify continuity.

For each served request it lists, per channel (QC, CC, DC), the route, the core, the
allocated block, and then -- for EVERY link of that route separately -- which FSUs of
that block are occupied on that link. Continuity (and core continuity) holds when the
occupied range is identical on every link. Run from the repo root::

    python -m tools.verify_continuity --config examples/example_config.py --agent FF
    python -m tools.verify_continuity --config examples/example_config.py --agent RF
"""

from __future__ import annotations

import argparse

import numpy as np

from agents.heuristics import FirstFitAgent, RandomFitAgent
from training.simulator import initialize_system, load_config_from_file
from request.traffic import TrafficGenerator


def _describe(ctx, record, slot):
    topo = ctx.topology
    part = ctx.env.partition
    k1_links = topo.route_link_indices(record.k1_route)
    k2_links = topo.route_link_indices(record.k2_route)
    channels = (
        ("QC", record.k1_route, k1_links, part.core_qc, record.block_qc),
        ("CC", record.k1_route, k1_links, part.core_cc, record.block_cc),
        ("DC", record.k2_route, k2_links, record.block_dc.core, record.block_dc),
    )
    ok = True
    for name, route, links, core, block in channels:
        start, end = block.start, block.start + block.size
        print(f"  {name}: route {'-'.join(map(str, route))} | core {core} | FSUs [{start}, {end})")
        for link in links:
            free = slot.free_mask([link], core)
            occupied = np.flatnonzero(~free[start:end]) + start
            same = len(occupied) == block.size
            ok &= same
            u, v = route[list(links).index(link)], route[list(links).index(link) + 1]
            rng = f"[{occupied[0]}, {occupied[-1] + 1})" if len(occupied) else "NONE"
            print(f"      link {u}-{v} (idx {link}): occupied {rng} {'OK' if same else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--agent", choices=["FF", "RF"], default="FF")
    parser.add_argument("--requests", type=int, default=6)
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    ctx = initialize_system(config)
    agent = FirstFitAgent() if args.agent == "FF" else RandomFitAgent(config.traffic.seed)
    traffic = TrafficGenerator(config.traffic, ctx.topology)
    traffic.reset(seed=config.traffic.seed)

    all_ok = True
    shown = 0
    while shown < args.requests:
        traffic.advance_time()
        qlr = traffic.generate_request()
        outcome = ctx.env.provision(qlr, agent, explore=False)
        if not outcome.result.served:
            continue
        record = next(r for r in ctx.env.registry.due_for_release(float("inf")) if r.qlr.request_id == qlr.request_id)
        print(f"Request {qlr.request_id}: {qlr.source} -> {qlr.destination}  [{args.agent}]")
        all_ok &= _describe(ctx, record, ctx.slot_table)
        shown += 1

    print("=" * 70)
    print("RESULT:", "every channel occupies the SAME FSU range on the SAME core on EVERY link"
          if all_ok else "CONTINUITY VIOLATION FOUND")


if __name__ == "__main__":
    main()
