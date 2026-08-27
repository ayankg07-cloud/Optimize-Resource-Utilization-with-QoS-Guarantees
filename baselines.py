"""
Baseline allocation policies for the QoS resource-allocation simulation.

Three baselines:

    1. Static QoS       -- allocate rho(n) every step, no adaptation.
    2. Online Proportional -- redistribute capacity among active queues
                            proportionally to their QoS shares.
    3. Greedy (3-category state machine) -- C1/C2/C3 categories with
                            uniform allocation to C1 queues.

All policy functions share the same signature:

    policy_fn(Q, L_t, t, rho, state) -> H_t

"""

import numpy as np


def static_qos_policy(
    Q: np.ndarray,
    L_t: np.ndarray,
    t: int,
    rho: np.ndarray,
    state: dict | None = None,
) -> np.ndarray:
    """
    Always allocate the fixed QoS share rho(n) regardless of queue state.

        H_t(n) = rho(n)   for all t, all n

    """
    return rho.copy()

def online_proportional_policy(
    Q: np.ndarray,
    L_t: np.ndarray,
    t: int,
    rho: np.ndarray,
    state: dict | None = None,
) -> np.ndarray:
    """
    Redistribute capacity among active queues proportional to their QoS shares.

    A queue is "active" if it has pending work: Q_t(n) > 0 or L_t(n) > 0.

        if A_t non-empty:
            H_t(n) = rho(n) / Σ_{j ∈ A_t} rho(j)    for n ∈ A_t
            H_t(n) = 0                              for n ∉ A_t
        else:
            H_t(n) = rho(n)                           for all n
    """
    N = len(Q)
    H = np.zeros(N)

    active = (Q > 0) | (L_t > 0)

    if np.any(active):
        rho_active_sum = rho[active].sum()
        H[active] = rho[active] / rho_active_sum
    else:
        H[:] = rho

    return H

def make_greedy_state(N: int) -> dict:
    """
    Create the initial mutable state for the greedy policy.

    All queues start in category C1 (active, receiving allocation).

    Categories:
    C1 : active queues currently receiving non-zero allocation.
    C2 : active queues currently receiving zero allocation.
    C3 : empty (inactive) queues with zero allocation.
    """
    return {
        'C1': set(range(N)),
        'C2': set(),
        'C3': set(),
    }


def greedy_policy(
    Q: np.ndarray,
    L_t: np.ndarray,
    t: int,
    rho: np.ndarray,
    state: dict | None = None,
) -> np.ndarray:
    """
    Greedy policy with 3-category state machine.

    Transition rules:
        1. C1 queue stays in C1 if non-empty, else moves to C3.
        2. C3 queue moves to C2 if it becomes non-empty, else stays in C3.
        3. If ALL C1 queues moved to C3 → all C2 queues move to C1.

    Allocation: resource split uniformly among C1 queues.
        H_t(n) = 1 / |C1|    for n ∈ C1
        H_t(n) = 0            otherwise
    """
    if state is None:
        raise ValueError("Greedy policy requires a state dict. Use make_greedy_state().")

    N = len(Q)
    C1, C2, C3 = state['C1'], state['C2'], state['C3']

    non_empty = set(n for n in range(N) if (Q[n] + L_t[n]) > 0)
    new_C1 = set()
    new_C2 = set(C2)
    new_C3 = set()

    moved_to_C3_from_C1 = set()
    for n in C1:
        if n in non_empty:
            new_C1.add(n)
        else:
            new_C3.add(n)
            moved_to_C3_from_C1.add(n)

    for n in C3:
        if n in non_empty:
            new_C2.add(n)
        else:
            new_C3.add(n)
    all_c1_emptied = (len(new_C1) == 0) and (len(C1) > 0)
    if all_c1_emptied:
        new_C1 = new_C2
        new_C2 = set()

    # if C1 is still empty (no C2 queues existed either),
    if len(new_C1) == 0 and len(non_empty) > 0:
        new_C1 = non_empty.copy()
        new_C2 -= non_empty
        new_C3 -= non_empty

    state['C1'] = new_C1
    state['C2'] = new_C2
    state['C3'] = new_C3

    H = np.zeros(N)
    if len(new_C1) > 0:
        alloc = 1.0 / len(new_C1)
        for n in new_C1:
            H[n] = alloc

    return H
