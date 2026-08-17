"""Modulation module -- distance-adaptive spectrum sizing for CC and DC only.

QC keeps a fixed, protocol-determined footprint (``F^{QC}_min``) and is handled in the
environment; this module implements the classical-channel sizing of Eqs. (5)-(6):

    F = ceil( bandwidth / (Delta f * eta(m*)) ) + G

where ``m*`` is the highest-spectral-efficiency modulation format whose reach covers
the path length. If no format's reach covers the path, the channel cannot be sized on
that route and the request is treated as infeasible on it.
"""

from __future__ import annotations

import math
from typing import Optional

from configs.config import ModulationConfig, ModulationFormat


def select_modulation(path_length_km: float, config: ModulationConfig) -> Optional[ModulationFormat]:
    """Return the highest-efficiency format whose reach covers ``path_length_km``.

    Distance-adaptive modulation [4]: among all formats that can physically reach the
    path length, pick the most spectrally efficient one. Returns ``None`` if no format
    reaches that far.
    """
    reachable = [f for f in config.formats if f.max_reach_km >= path_length_km]
    if not reachable:
        return None
    return max(reachable, key=lambda f: f.spectral_efficiency)


def required_fs(path_length_km: float, bandwidth: float, config: ModulationConfig) -> Optional[int]:
    """Compute the FSU requirement for a classical channel (CC or DC), Eqs. (5)-(6).

    Returns ``None`` when the path is unreachable by any modulation format.
    """
    fmt = select_modulation(path_length_km, config)
    if fmt is None:
        return None
    slots = math.ceil(bandwidth / (config.fsu_width_ghz * fmt.spectral_efficiency))
    return slots + config.guard_band_fsus


# ``required_fs`` needs the FSU width, which lives in NetworkConfig, not ModulationConfig.
# To keep the modulation table self-contained we thread the width in via a small wrapper
# so callers pass a single object. The width is injected by the environment at build time.
class ModulationTable:
    """Bound modulation table: table + FSU width, ready to size a classical channel."""

    def __init__(self, config: ModulationConfig, fsu_width_ghz: float) -> None:
        if fsu_width_ghz <= 0:
            raise ValueError("fsu_width_ghz must be > 0")
        self._config = config
        self._fsu_width_ghz = fsu_width_ghz

    def select(self, path_length_km: float) -> Optional[ModulationFormat]:
        return select_modulation(path_length_km, self._config)

    def required_fs(self, path_length_km: float, bandwidth: float) -> Optional[int]:
        fmt = self.select(path_length_km)
        if fmt is None:
            return None
        slots = math.ceil(bandwidth / (self._fsu_width_ghz * fmt.spectral_efficiency))
        return slots + self._config.guard_band_fsus

    def efficiency(self, path_length_km: float) -> Optional[float]:
        fmt = self.select(path_length_km)
        return None if fmt is None else fmt.spectral_efficiency
