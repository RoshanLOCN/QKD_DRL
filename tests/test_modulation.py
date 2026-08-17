from configs.config import ModulationConfig, ModulationFormat
from core.modulation import ModulationTable, required_fs, select_modulation


def _config():
    return ModulationConfig(
        formats=(
            ModulationFormat("BPSK", 1.0, 4000.0),
            ModulationFormat("QPSK", 2.0, 2000.0),
            ModulationFormat("8QAM", 3.0, 1000.0),
            ModulationFormat("16QAM", 4.0, 500.0),
        ),
        guard_band_fsus=1,
    )


def test_selects_highest_efficiency_that_reaches():
    cfg = _config()
    assert select_modulation(400.0, cfg).name == "16QAM"   # all reach; pick highest eta
    assert select_modulation(1500.0, cfg).name == "QPSK"   # 8QAM/16QAM too short
    assert select_modulation(3000.0, cfg).name == "BPSK"


def test_unreachable_returns_none():
    cfg = _config()
    assert select_modulation(5000.0, cfg) is None
    assert required_fs(5000.0, 10.0, cfg) is None


def test_required_fs_formula():
    cfg = _config()
    # 400 km -> 16QAM (eta=4); 10 / (12.5*4) = 0.2 -> ceil = 1; + guard 1 = 2
    assert ModulationTable(cfg, 12.5).required_fs(400.0, 10.0) == 2
    # 3000 km -> BPSK (eta=1); 100 / (12.5*1) = 8 -> ceil 8; + guard 1 = 9
    assert ModulationTable(cfg, 12.5).required_fs(3000.0, 100.0) == 9


def test_efficiency_lookup():
    table = ModulationTable(_config(), 12.5)
    assert table.efficiency(400.0) == 4.0
    assert table.efficiency(5000.0) is None
