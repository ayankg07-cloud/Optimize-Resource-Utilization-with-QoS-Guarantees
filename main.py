"""
Runs Static QoS, Online Proportional, and Greedy policies on the same
workload sequence, then generates:

    1. Queue-length plot:       Q_t(n) vs t   (one subplot per queue)
    2. Cumulative-workload plot: Sum w vs t      (one curve per baseline)
"""

import numpy as np
import matplotlib.pyplot as plt

from config import N, T, RHO, SEED
from workloads import generate_workloads
from engine import simulate
from baselines import (
    static_qos_policy,
    online_proportional_policy,
    greedy_policy,
    make_greedy_state,
)


def run_all_baselines():
    """Run all three baselines on the same workload and return results."""
    print(f"Generating workloads: T={T}, N={N}, seed={SEED}")
    L = generate_workloads(T, N, SEED)
    print(f"  Workload stats -- mean per queue: {L.mean(axis=0)}")
    print(f"  Workload stats -- max  per queue: {L.max(axis=0)}")
    print()

    results = {}

    #static QOS
    print("Running Baseline 1--Static QoS")
    results['Static QoS'] = simulate(static_qos_policy, L, T, N, RHO)
    print("  [PASS] Done")

    #Online Proportional
    print("Running Baseline 2--Online Proportional")
    results['Online Proportional'] = simulate(online_proportional_policy, L, T, N, RHO)
    print("  [PASS] Done")

    #Greedy
    print("Running Baseline 3--Greedy")
    greedy_state = make_greedy_state(N)
    results['Greedy'] = simulate(greedy_policy, L, T, N, RHO, greedy_state)
    print("  [PASS] Done")
    print()

    return L, results


def verify_results(results: dict):
    """Run automated sanity checks on all baseline results."""
    all_passed = True
    for name, res in results.items():
        Q, w, H = res['Q'], res['w'], res['H']
        checks = []

        nan_ok = not (np.any(np.isnan(Q)) or np.any(np.isnan(w)) or np.any(np.isnan(H)))
        checks.append(("No NaN values", nan_ok))

        q_ok = np.all(Q >= -1e-9)
        checks.append(("Q >= 0", q_ok))

        # sum(H) <= 1 + eps
        cap_ok = np.all(H.sum(axis=1) <= 1.0 + 1e-6)
        checks.append(("Capacity <= 1", cap_ok))

        # Bounded queue growth(for stable system check)
        max_q = Q.max()
        bounded = max_q < 1000
        checks.append((f"Bounded queues (max={max_q:.1f})", bounded))

        print(f"\n  [{name}]")
        for label, ok in checks:
            status = "[PASS]" if ok else "[FAIL]"
            print(f"    {status}  {label}")
            if not ok:
                all_passed = False

        # Summary stats
        total_work = w.sum()
        print(f"    Total completed workload: {total_work:.1f}")
        print(f"    Final queue lengths:      {Q[-1]}")

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print()
    return all_passed


def plot_queue_lengths(results: dict):
    """
    Plot Q_t(n) vs t -- one subplot per queue, overlaying all baselines.
    """
    fig, axes = plt.subplots(N, 1, figsize=(14, 3.5 * N), sharex=True)
    if N == 1:
        axes = [axes]

    colors = {'Static QoS': '#2196F3', 'Online Proportional': '#FF9800', 'Greedy': '#4CAF50'}
    timesteps = np.arange(T)

    for n in range(N):
        ax = axes[n]
        for name, res in results.items():
            ax.plot(
                timesteps, res['Q'][:, n],
                label=name, color=colors[name], alpha=0.8, linewidth=0.7,
            )
        ax.set_ylabel(f'Queue {n+1}\nQ_t({n+1})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Queue Lengths Over Time (Day 1 Sanity Check)', fontsize=13)
    axes[-1].set_xlabel('Time step t')
    plt.tight_layout()
    plt.savefig('plot_queue_lengths.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cumulative_workload(results: dict):
    """
    Plot cumulative completed workload Sum_n Sum_{t'=1}^{t} w_{t'}(n) vs t.
    One curve per baseline.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Static QoS': '#2196F3', 'Online Proportional': '#FF9800', 'Greedy': '#4CAF50'}
    timesteps = np.arange(T)

    for name, res in results.items():
        cum_work = res['w'].sum(axis=1).cumsum()
        ax.plot(
            timesteps, cum_work,
            label=name, color=colors[name], linewidth=1.5,
        )

    ax.set_xlabel('Time step t')
    ax.set_ylabel('Cumulative Completed Workload')
    ax.set_title('Resource Utilization , Cumulative Workload (Day 1 Sanity Check)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_cumulative_workload.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():

    L, results = run_all_baselines()

    verify_results(results)
    plot_queue_lengths(results)
    plot_cumulative_workload(results)
    print("Done!")


if __name__ == '__main__':
    main()
