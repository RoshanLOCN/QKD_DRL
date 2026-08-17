import dataclasses

import pytest

from configs.config import NetworkConfig, RoutingConfig


def test_network_requires_at_least_three_cores():
    with pytest.raises(ValueError):
        NetworkConfig(cores_per_link=2, fsus_per_core=8, fsu_width_ghz=12.5, core_qc_index=0, core_cc_index=1)


def test_network_reserved_cores_must_be_distinct():
    with pytest.raises(ValueError):
        NetworkConfig(cores_per_link=4, fsus_per_core=8, fsu_width_ghz=12.5, core_qc_index=1, core_cc_index=1)


def test_data_core_pool_excludes_reserved_cores():
    net = NetworkConfig(cores_per_link=5, fsus_per_core=8, fsu_width_ghz=12.5, core_qc_index=0, core_cc_index=2)
    assert net.data_core_indices == (1, 3, 4)


def test_action_space_size(base_config):
    expected = (
        base_config.routing.k1
        * base_config.candidates.i_qc
        * base_config.candidates.i_cc
        * base_config.routing.k2
        * base_config.candidates.i_dc
    )
    assert base_config.action_space_size == expected


def test_routing_rejects_non_positive():
    with pytest.raises(ValueError):
        RoutingConfig(k1=0, k2=1)
