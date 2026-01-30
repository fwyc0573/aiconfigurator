# Comparative Analysis: aiconfigurator vs. Frontier

## Modification History

| Date       | Summary of Changes |
|------------|-------------------|
| 2026-01-09 | Initial creation of comparative analysis document |

## 1. Executive Summary

This document presents a comparative analysis between **aiconfigurator** (the baseline configuration tool) and **Frontier** (the target discrete-event simulation architecture).

**Key Finding**: `aiconfigurator` employs a **steady-state analytical modeling** approach based on hardware profiling and rate matching. In contrast, Frontier utilizes **discrete-event simulation (DES)** to model dynamic system behavior, request scheduling, and detailed queuing effects.

## 2. Simulation Mechanism Comparison

| Feature | aiconfigurator (Baseline) | Frontier (Target) |
|---------|---------------------------|-------------------|
| **Core Paradigm** | **Analytical / Rate-Based** | **Discrete-Event Simulation (DES)** |
| **Request Generation** | Static assumptions (Batch Size, ISL/OSL). No arrival process. | Dynamic arrival processes (Poisson, Trace-driven). |
| **Scheduling** | Implicit "perfect" scheduling. Models capacity, not decisions. | Explicit scheduling logic (FCFS, Priority, Preemption). |
| **Latency Prediction** | Interpolation of profiled kernel data + Linear scaling. | Simulation of execution timeline + Contention modeling. |
| **Disaggregation** | **Rate Matching**: Balances prefill/decode throughput analytically. | **Flow Control**: Simulates token transfer and KV-cache transfer delays. |
| **Metrics** | Throughput (Tokens/s), Latency (TTFT/TPOT) at saturation. | Tail Latency (P99), Queue Length, Jitter, transient behavior. |

### 2.1 aiconfigurator Deep Dive

`aiconfigurator`'s logic resides primarily in `src/aiconfigurator/sdk/inference_session.py`.

*   **Operation-Level Profiling**: Break down LLM inference into atomic operations (GEMM, Attention, NCCL).
*   **Database Query**: Retrieve latency from `PerfDatabase` (Silicon data) using interpolation.
*   **Throughput Estimation**:
    *   $$ \text{Throughput} = \frac{\text{Batch Size}}{\text{Latency}} $$
    *   Applies "Degradation Factors" (e.g., `_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9`) to account for real-world inefficiencies.
*   **Optimization**: Exhaustively searches the configuration space (TP/PP/DP sizes) to maximize Tokens/s/GPU under SLA constraints.

## 3. Research Perspective: Baseline Comparison Framework

To use `aiconfigurator` as a valid baseline for Frontier research, comparisons must bridge the gap between "Capacity Modeling" and "Dynamic Simulation".

### 3.1 Recommended Comparison Dimensions

1.  **Capacity Accuracy (Saturation Point)**
    *   **Question**: Does Frontier's max throughput match aiconfigurator's predicted capacity?
    *   **Method**: Run Frontier with infinite load. Compare realized Throughput vs aiconfigurator's theoretical max.

2.  **Configuration Search Efficiency**
    *   **Question**: Can Frontier find better configurations by modeling transient effects?
    *   **Method**: Take the "Best Config" from aiconfigurator. Run it in Frontier. Check if P99 latency meets SLA under realistic traffic (Poisson).

3.  **Disaggregation Overheads**
    *   **Gap**: aiconfigurator assumes a fixed "correction scale" for disaggregation overhead.
    *   **Frontier Advantage**: Can explicitly model network contention and KV-cache transmission delays.

### 3.2 Proposed Experiments

1.  **The "Static" Validation**:
    *   Configure Frontier with constant arrival rate = aiconfigurator's predicted throughput.
    *   Measure if latency stays stable. If queues explode, aiconfigurator is overly optimistic.

2.  **The "Burst" Stress Test**:
    *   aiconfigurator cannot model bursts.
    *   Demonstrate Frontier's value by showing how "Optimal" static configurations fail under bursty traffic.

## 4. Modeling Granularity

*   **aiconfigurator**: High fidelity at the *kernel* level (uses real GPU profiling data). Low fidelity at the *system* level (ignores queuing).
*   **Frontier**: Variable fidelity. Can use aiconfigurator's database for kernel duration, but adds high fidelity *system* dynamics.

## 5. Conclusion

`aiconfigurator` provides a fast, optimistic upper-bound on performance. It is an excellent tool for **Space Pruning** (narrowing down TP/PP candidates). Frontier should be positioned as the **Verification & Refinement** layer that captures the complex dynamics of production traffic.
