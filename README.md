# Optimize-Resource-Utilization-with-QoS-Guarantees

This project implements the resource allocation optimization simulation described in the paper *"Learning to Optimize Resource Utilization with QoS Guarantees"* (IEEE INFOCOM 2025).

## Current Status
Currently completing **Day 1** of a 5-day implementation sprint. The focus of Day 1 was to build the simulation sandbox and three baseline allocation policies.

### Completed Work (Day 1)
- **Simulation Sandbox:** Implemented a discrete-time simulation engine (`engine.py`) that models queue dynamics and completed workload tracking based on the paper's model.
- **Workload Generator:** Created a bursty, adversarial-style synthetic workload generator (`workloads.py`) using Bernoulli and Uniform distributions to stress-test the allocation policies.
- **Baselines (`baselines.py`):**
  - **Static QoS:** Allocates a fixed portion of resources strictly according to the QoS requirements.
  - **Online Proportional:** Dynamically redistributes capacity among active queues proportional to their QoS shares.
  - **Greedy:** A 3-category state machine that uniformly splits resources among actively backlogged queues.
- **Sanity Checks & Visualizations (`main.py`):** Runs all policies on the same workload sequence to verify stability and capacity constraints, and plots the queue lengths and cumulative resource utilization.

## Usage
To run the simulation and generate baseline comparison plots:
```bash
python main.py
```
