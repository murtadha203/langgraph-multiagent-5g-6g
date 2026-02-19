
# Key Performance Results

This document summarizes the key performance achievements of the Polymorphic IBN system, as detailed in **Chapter 5** of the Thesis.

## 1. The "Cell Edge" Breakthrough
The most significant result is the system's ability to handle the "Dead Zone" at the cell edge.

*   **Metric:** Task Verification Success Rate
*   **Scenario:** High-speed user (120 km/h) at the cell boundary.
*   **Result (Polymorphic AI):** **94% Success Rate**.
*   **Result (Baseline 5G):** ~60% Success Rate.
*   **Why?** The "Strategic Retreat" logic switches to local processing *before* the connection fails, while standard baselines keep trying (and failing) to offload.

## 2. Energy Efficiency
By minimizing failed transmissions and optimizing for the "Green Mode" intent:

*   **Metric:** Energy Consumption (Joules per Task).
*   **Result:** **Order-of-Magnitude Reduction (~40x)** in energy waste during critical failure scenarios (0.1J vs 4.0J).
*   **Mechanism:** The "Throughput Guardrail" prevents the agent from wasting power on transmissions that are mathematically guaranteed to fail.

## 3. Stability (Zero Ping-Pong)
The "TeleLLM" orchestrator successfully eliminates unnecessary handovers.

*   **Metric:** Handover Count per Minute.
*   **Result:** **Zero Ping-Pong Handovers** in stable conditions.
*   **Mechanism:** The "Strategic Shield" enforces a dwell time and hysteresis margin, preventing the rapid switching seen in raw RL agents.

## 4. Polymorphic Adaptation
The system successfully morphs its behavior based on the `INTENT` vector:

| Intent | Latency | Energy | Behavior Observed |
| :--- | :--- | :--- | :--- |
| **URLLC** | **Low** | High | Aggressive offloading; Connects to closest strong tower. |
| **Green IoT** | High | **Low** | Passive; Processes locally; Avoids handovers. |

*Confirmed by the Radar Plot in the Thesis (Figure 5.4).*
