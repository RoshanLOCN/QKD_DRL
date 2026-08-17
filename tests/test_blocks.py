from core.blocks import (
    feasible_blocks_multi_core,
    feasible_blocks_single_core,
)
from core.resource_grid import SlotTable


def _table(num_links=1, cores=3, fsus=20):
    return SlotTable(num_links, cores, fsus)


def test_single_core_closest_fit_ranking():
    table = _table()
    # Carve free runs of length 2 (0-1), 5 (4-8), 3 (12-14) on core 0.
    table.occupy([0], 0, 2, 2)      # occupy 2-3 -> run 0-1 (len2), then free from 4
    table.occupy([0], 0, 9, 3)      # occupy 9-11 -> run 4-8 (len5)
    table.occupy([0], 0, 15, 5)     # occupy 15-19 -> run 12-14 (len3)
    blocks = feasible_blocks_single_core([0], 0, 3, table, cap=5)
    # runs >= 3: len3 (start12) and len5 (start4); closest to 3 is len3 first.
    assert [(b.start, b.size) for b in blocks] == [(12, 3), (4, 3)]


def test_single_core_no_run_large_enough():
    table = _table(fsus=4)
    table.occupy([0], 0, 2, 1)  # runs: len2 (0-1) and len1 (3)
    assert feasible_blocks_single_core([0], 0, 3, table, cap=5) == ()


def test_single_core_cap_applied():
    table = _table(fsus=20)
    blocks = feasible_blocks_single_core([0], 0, 2, table, cap=1)
    assert len(blocks) == 1


def test_continuity_intersection_across_links():
    table = _table(num_links=2)
    # Free intersection must hold on BOTH links; block on link 1 shrinks the common run.
    table.occupy([1], 0, 5, 20 - 5)  # link1 core0: free only 0-4
    blocks = feasible_blocks_single_core([0, 1], 0, 4, table, cap=5)
    assert all(b.start + b.size <= 5 for b in blocks)


def test_multi_core_spans_data_pool():
    table = _table(cores=4, fsus=10)
    # Fill core 2 entirely; DC must fall back to core 3.
    table.occupy([0], 2, 0, 10)
    blocks = feasible_blocks_multi_core([0], [2, 3], 4, table, cap=5)
    assert blocks and all(b.core == 3 for b in blocks)


def test_xt_avoided_excludes_slots_occupied_on_adjacent_core():
    # Mirrors the paper's Fig. 2 demonstration: core 1 occupies slot 0-2, so an
    # XT-avoided search on adjacent core 0 must not offer that slot range at all,
    # even though core 0 itself is entirely free.
    table = _table(cores=3, fsus=10)
    table.occupy([0], 1, 0, 3)
    adjacency = {0: (1,), 1: (0, 2), 2: (1,)}

    without_xt = feasible_blocks_single_core([0], 0, 3, table, cap=5)
    assert (0, 3) in [(b.start, b.size) for b in without_xt]

    with_xt = feasible_blocks_single_core([0], 0, 3, table, cap=5, adjacency=adjacency)
    assert (0, 3) not in [(b.start, b.size) for b in with_xt]
    # The remaining free run (3-9) on core 0 is unaffected -- core 2 (not adjacent
    # to core 0) stays irrelevant, and core 1 is free there too.
    assert (3, 3) in [(b.start, b.size) for b in with_xt]


def test_xt_avoided_no_effect_without_adjacency():
    table = _table(cores=3, fsus=10)
    table.occupy([0], 1, 0, 3)
    blocks = feasible_blocks_single_core([0], 0, 3, table, cap=5, adjacency=None)
    assert (0, 3) in [(b.start, b.size) for b in blocks]
