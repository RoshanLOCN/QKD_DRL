from core.resource_grid import CorePartition, SlotTable
from training.metrics import EpisodeMetrics, resource_utilization


def test_episode_metrics_bp_and_ar():
    m = EpisodeMetrics()
    m.record(served=True, reward=2.0)
    m.record(served=False, reward=-1.0)
    m.record(served=True, reward=2.0)
    assert m.total == 3 and m.served == 2
    assert abs(m.blocking_probability - (1 - 2 / 3)) < 1e-9
    assert abs(m.average_reward - (3.0 / 3)) < 1e-9


def test_resource_utilization_per_role():
    partition = CorePartition(core_qc=0, core_cc=1, data_cores=(2, 3))
    table = SlotTable(num_links=2, cores_per_link=4, fsus_per_core=10)
    table.occupy([0, 1], 0, 0, 5)   # QC: 2 links * 5 = 10 of 2*10 = 20 -> 0.5
    table.occupy([0], 2, 0, 4)      # DC: 4 of 2*2*10 = 40 -> 0.1
    ru = resource_utilization(table, partition, num_links=2)
    assert abs(ru.qc - 0.5) < 1e-9
    assert ru.cc == 0.0
    assert abs(ru.dc - 0.1) < 1e-9
    assert abs(ru.aggregate - (10 + 0 + 4) / (20 + 20 + 40)) < 1e-9
