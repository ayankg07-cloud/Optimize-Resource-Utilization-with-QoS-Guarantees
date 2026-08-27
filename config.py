"""
Global constants and hyperparameters for the QoS-MWU simulation.

Reference: "Learning to Optimize Resource Utilization with QoS Guarantees"
           (IEEE INFOCOM 2025)

Notation:
    N       -- number of queues (users)
    T       -- simulation time horizon
    RHO     -- fixed QoS resource shares promised to each queue
    SEED    -- random seed for reproducibility
"""

import numpy as np

N = 3                                  # number of queues, indexed 0..N-1
T = 10000                              # total simulation horizon

# rho(n): fixed QoS resource share promised to queue n.
RHO = np.array([0.3, 0.3, 0.4])

SEED = 42

