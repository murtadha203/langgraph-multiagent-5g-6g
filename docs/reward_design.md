The reward function is the foundational component that shapes the agent's behavior. The project utilizes a multi-objective reward structure to balance Quality of Service (QoS), energy efficiency, and network stability.

---

## 1. Multi-Objective Reward Equation

$$ R_{total} = w_{lat} \cdot R_{latency} + w_{eng} \cdot R_{energy} + w_{stab} \cdot R_{stability} + w_{rel} \cdot R_{reliability} $$

Where the weights ($w_{k}$) are dynamically adjusted based on the system intent (Intent-Based Networking) or fixed during specific training phases.

### A. Latency Reward ($R_{latency}$)
Encourages low-latency connections (High Throughput / Low Delay).
$$ R_{latency} = \frac{Throughput_{current}}{Throughput_{max}} $$

### B. Energy Reward ($R_{energy}$)
Penalizes high power consumption.
$$ R_{energy} = 1.0 - \frac{Power_{current}}{Power_{max}} $$

### C. Stability Reward ($R_{stability}$)
Penalizes "Ping-Pong" handovers (rapid switching back and forth).
$$ R_{stability} = -1.0 \times \text{Count(Recent Handovers within 2s)} $$
*Note: A single necessary handover is NOT penalized. Only rapid oscillations incur costs.*

### D. Coverage / Outage Penalty ($R_{coverage}$)
Critical penalty to prevent agents from getting stuck in dead zones (RSRP < -95 dBm).
$$ R_{coverage} = \begin{cases} -2.0 & \text{if RSRP } < -95 \text{ dBm (The Floor is Lava)} \\ 0.0 & \text{otherwise} \end{cases} $$
*Note: This value (-2.0) is scaled by a weight and factor of 10, resulting in a -20.0 effective penalty, which is clipped to remain gradient-friendly.*

### E. Normalization
All raw rewards are clipped to the range **[-20.0, +20.0]** to ensure training stability and prevent gradient explosions.

---

## 3. Project Curriculum

To ensure robust policy convergence, a multi-phase curriculum learning strategy is employed:

| Phase | Designation | Primary Focus | Objective |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Exploration | State-space discovery | Map environmental dynamics and action outcomes. |
| **Phase 2** | Optimization | Performance maximization | Maximize $R_{latency}$ and $R_{energy}$ efficiency. |
| **Phase 3** | Stabilization | Volatility suppression | Enforce $R_{stability}$ constraints and noise robustness. |
| **Phase 4** | **Polymorphic** | **Intent Adaptation** | Generalize policy to handle dynamic $\alpha, \beta, \gamma$ weights. |

---

## 4. Intent-Based Weighting

The agents are optimized using the **Proximal Policy Optimization (PPO)** objective, which improves training stability by clipping policy updates:

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ] $$

Where:
*   $r_t(\theta)$: Probability ratio between new and old policies.
*   $\hat{A}_t$: Advantage estimate computed via **Generalized Advantage Estimation (GAE)**.
*   $\epsilon$: Clipping coefficient (set to 0.2).

---

## 4. Intent-Based Weighting
During system operation, the high-level intent vector dynamically modulates the reward components:
*   **Latency-Critical**: $w_{lat} \uparrow$ (Focus on throughput).
*   **Energy-Saving**: $w_{eng} \uparrow$ (Focus on battery conservation).
*   **High-Reliability**: $w_{rel} \uparrow$ (Minimize failure probability).
