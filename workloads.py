"""
Synthetic workload generator for the QoS resource-allocation simulation.

Produces bursty, adversarial-style arrival patterns using Bernoulli and Uniform
random variables, with different burstiness profiles per queue to stress-test
the baseline allocation policies.

"""

import numpy as np
from config import N, T, SEED


def generate_workloads(
    T: int = T,
    N: int = N,
    seed: int = SEED,
) -> np.ndarray:
    """
    Generate a synthetic workload matrix L[t, n].

    Each queue has an independent Bernoulli arrival indicator multiplied by a
    Uniform magnitude, creating asymmetric bursty profiles:

        Queue 0: moderate freq, moderate size  -- Bernoulli(0.5) x U(0.10, 0.30)
        Queue 1: low freq, high size           -- Bernoulli(0.2) x U(0.30, 0.60)
        Queue 2: high freq, low size           -- Bernoulli(0.8) x U(0.05, 0.15)

    L : np.ndarray, shape (T, N)
        L[t, n] is the workload arriving at queue n at time t.
    """
    rng = np.random.default_rng(seed)

    # Per-queue arrival parameters:
    profiles = [
        (0.5, 0.10, 0.30),   # Queue 0: moderate frequency, moderate size
        (0.2, 0.30, 0.60),   # Queue 1: low frequency, high size
        (0.8, 0.05, 0.15),   # Queue 2: high frequency, low size
    ]

    if N != len(profiles):
        raise ValueError(
            f"Default profiles are defined for {len(profiles)} queues, got N={N}. "
            "Extend the profiles list or pass a custom generator."
        )

    L = np.zeros((T, N))
    for n, (p, lo, hi) in enumerate(profiles):
        arrivals = rng.binomial(1, p, size=T)          # 0 or 1
        magnitudes = rng.uniform(lo, hi, size=T)       # burst size
        L[:, n] = arrivals * magnitudes

    return L
