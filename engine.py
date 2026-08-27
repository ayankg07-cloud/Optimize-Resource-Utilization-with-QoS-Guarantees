"""
Discrete-time simulation engine for the QoS resource-allocation problem.

    Q_{t+1}(n) = max{ L_t(n) + Q_t(n) - H_t(n),  0 }

and the completed-workload definition:

    w_t(n) = min{ H_t(n),  Q_t(n) + L_t(n) }
"""

import numpy as np
from config import N, T, RHO


def simulate(
    policy_fn,
    L: np.ndarray,
    T: int = T,
    N: int = N,
    rho: np.ndarray = RHO,
    policy_state: dict | None = None,
) -> dict:
    """

    Parameters:
    policy_fn : callable
        Allocation policy with signature:
            policy_fn(Q, L_t, t, rho, state) -> H_t
        where
            Q   : np.ndarray (N,)  -- current queue lengths
            L_t : np.ndarray (N,)  -- workload arriving this step
            t   : int              -- current timestep
            rho : np.ndarray (N,)  -- QoS shares
            state : dict or None   -- mutable policy state (for Greedy)
        Must return H_t : np.ndarray (N,) with sum(H_t) <= 1.
    L : np.ndarray, shape (T, N)
        Pre-generated workload arrivals.
    T : int
        Number of timesteps to simulate.
    N : int
        Number of queues.
    rho : np.ndarray, shape (N,)
        QoS resource shares.
    policy_state : dict or None
        Initial mutable state for stateful policies (e.g. Greedy categories).
        Passed by reference to policy_fn at each step.

    Returns:
    results : dict with keys
        'Q' : np.ndarray (T, N) -- queue lengths at end of each step
        'w' : np.ndarray (T, N) -- completed workload at each step
        'H' : np.ndarray (T, N) -- allocation decisions at each step
    """
    EPS = 1e-9 

    # History arrays
    Q_history = np.zeros((T, N))
    w_history = np.zeros((T, N))
    H_history = np.zeros((T, N))

    # Initial queue lengths 
    Q = np.zeros(N)

    for t in range(T):
        L_t = L[t]
        H_t = policy_fn(Q, L_t, t, rho, policy_state)
        #sanity checks
        assert H_t.shape == (N,), f"H_t shape mismatch at t={t}: {H_t.shape}"
        assert np.all(H_t >= -EPS), f"Negative allocation at t={t}: {H_t}"
        assert np.sum(H_t) <= 1.0 + EPS, (
            f"Capacity violated at t={t}: sum(H)={np.sum(H_t):.6f}"
        )

        #w_t(n) = min{ H_t(n), Q_t(n) + L_t(n) }
        available = Q + L_t
        w_t = np.minimum(H_t, available)

        #Q_{t+1}(n) = max{ L_t(n) + Q_t(n) - H_t(n), 0 }
        Q_new = np.maximum(available - H_t, 0.0)

        H_history[t] = H_t
        w_history[t] = w_t
        Q_history[t] = Q_new
        Q = Q_new

    return {
        'Q': Q_history,
        'w': w_history,
        'H': H_history,
    }
