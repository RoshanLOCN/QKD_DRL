"""DRL-based joint QC/CC/DC provisioning for QKD-secured SS-EON (Tokyo12).

This package is a faithful implementation of the three specification documents:
the IEEE paper, the QKD-ON -> SS-EON migration note, and the algorithmic-flow guide.

Every Quantum Lightpath Request (QLR) is provisioned as three coordinated channels
-- Quantum (QC), Classical Control (CC), Data (DC) -- resolved jointly, in one
five-part action ``(k1, i_QC, i_CC, k2, i_DC)``, all-or-nothing.

No numeric parameter is embedded in the code: everything is supplied through the
configuration objects in :mod:`sseon_qkd_drl.configs.config`, and the network graph
is loaded from an external file (:func:`sseon_qkd_drl.core.topology.load_topology`).
"""
