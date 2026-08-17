import pytest

from core.resource_grid import CorePartition, SlotTable
from configs.config import NetworkConfig


def test_partition_from_config():
    net = NetworkConfig(
        cores_per_link=4, fsus_per_core=8, fsu_width_ghz=12.5, core_qc_index=0, core_cc_index=1,
        xt_avoided=False,
    )
    part = CorePartition.from_config(net)
    assert part.core_qc == 0 and part.core_cc == 1 and part.data_cores == (2, 3)
    assert part.adjacency == {}


def test_partition_adjacency_populated_when_xt_avoided():
    net = NetworkConfig(
        cores_per_link=4, fsus_per_core=8, fsu_width_ghz=12.5, core_qc_index=0, core_cc_index=1,
        xt_avoided=True,
    )
    part = CorePartition.from_config(net)
    # 4-core fallback layout is a linear chain: 0-1-2-3.
    assert part.adjacency[0] == (1,)
    assert part.adjacency[1] == (0, 2)
    assert part.adjacency[2] == (1, 3)
    assert part.adjacency[3] == (2,)


def test_occupy_release_roundtrip():
    table = SlotTable(2, 3, 10)
    assert table.is_block_free([0, 1], 2, 3, 4)
    table.occupy([0, 1], 2, 3, 4)
    assert not table.is_block_free([0, 1], 2, 3, 4)
    assert table.occupied_fsus_on_core(2) == 2 * 4
    table.release([0, 1], 2, 3, 4)
    assert table.is_block_free([0, 1], 2, 3, 4)
    assert table.occupied_fsus_on_core(2) == 0


def test_occupy_rejects_partially_used_block():
    table = SlotTable(1, 2, 10)
    table.occupy([0], 0, 5, 2)
    with pytest.raises(ValueError):
        table.occupy([0], 0, 4, 3)  # overlaps 5


def test_free_mask_intersection():
    table = SlotTable(2, 1, 6)
    table.occupy([0], 0, 0, 2)
    table.occupy([1], 0, 4, 2)
    mask = table.free_mask([0, 1], 0)
    assert list(mask) == [False, False, True, True, False, False]


def test_out_of_range_block_is_not_free():
    table = SlotTable(1, 1, 5)
    assert not table.is_block_free([0], 0, 3, 5)
