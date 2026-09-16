# Task 3: Physics Consistency Guide

## Overview

**Task 3** evaluates the **physical realism** of your traffic reconstructions. Pure data-driven machine learning models frequently output predictions that appear statistically reasonable according to RMSE, but catastrophically violate the laws of physics — for example, vehicles randomly appearing or disappearing between sensor detectors.

> **CRITICAL RULE**:
> **There is NO separate submission file for Task 3.**
> Task 3 is scored automatically by the leaderboard on the **Task 1 state rows** (`speed_kmh` and `flow_vph`).
>
> Together, **Task 1 (35%) + Task 3 (15%) = 50% of your total leaderboard score!**

---

## 1. Evaluation Metric & Components

The physical consistency score $S_{\text{physics}}$ combines two physics principles:

$$S_{\text{physics}} = \frac{1}{3} S_{\text{FD}} + \frac{2}{3} S_{\text{LWR}}$$

### Component 1: Fundamental Diagram Compliance ($S_{\text{FD}}$, 33.3% weight)
Evaluates whether each $(v, q)$ pair lies on the calibrated triangular Fundamental Diagram:
- **Free-flow branch** ($k \le k_{\text{crit}}$): $q = v_f \cdot k$, where speed remains near free-flow speed $v_f$.
- **Congested branch** ($k > k_{\text{crit}}$): $q = w(k_{\text{jam}} - k)$, where $w = \frac{q_{\text{cap}}}{k_{\text{jam}} - k_{\text{crit}}}$ is the backward wave speed.

**Penalties**:
- Speeds exceeding physical road capability ($v > v_f$).
- Flows exceeding maximum road capacity ($q > q_{\text{cap}}$).
- Zero flow reported on active corridors (`EMPTY_FLOOR_VPH = 50.0`).

---

### Component 2: LWR Vehicle Conservation ($S_{\text{LWR}}$, 66.7% weight)
Evaluates the discrete **Lighthill-Whitham-Richards (LWR)** conservation law on each link $l$ over each 5-minute interval $\Delta t = \frac{5}{60}\text{ hours}$:

$$\Delta N_l(t) = N_l(t + \Delta t) - N_l(t)$$

$$\text{Flux}_l(t) = \Delta t \cdot \left[ q_{\text{inflow}}(t) + q_{\text{on-ramp}}(t) - q_{\text{outflow}}(t) - q_{\text{off-ramp}}(t) \right]$$

$$\text{Conservation Residual } r = \Delta N_l(t) - \text{Flux}_l(t)$$

Where:
- $N_l(t) = k_l(t) \times L_l$ is the total vehicle accumulation on the link of length $L_l$ (km).
- An ideal, conservative traffic state achieves $r \approx 0$.

---

## 2. Why Task 3 Cannot Be Accurately Scored Locally

As documented in the official benchmark repository:
- True network boundary flows (inflows entering the corridor and outflows exiting) are organizer-held secrets.
- While `trafficflowbench-public/src/task3/score_task3.py` can be executed locally, it falls back to estimating boundary fluxes purely from participant-submitted flows. This introduces approximation errors larger than the subtle conservation signal itself.
- **Action**: Use the local script for debugging sanity checks, and rely on the Kaggle public leaderboard for your true $S_{\text{physics}}$ score.

---

## 3. How to Win Task 3 (Physics-Informed Strategies)

To maximize your score on both Task 1 and Task 3:

### Strategy 1: Physical Bound Clipping
Never allow unconstrained regression models to output non-physical values. Clip predictions per link using `fd_parameters.csv`:
```python
pred_speed = np.clip(pred_speed, 0.0, 1.05 * free_speed_kmh)
pred_flow = np.clip(pred_flow, 50.0, 1.05 * capacity_vph)
```

### Strategy 2: Minimum-Norm Conservation Projection
Apply symmetric boundary flow corrections to balance adjacent links:
$$q_{\text{in}} \leftarrow q_{\text{in}} + \frac{r}{2 \Delta t}, \quad q_{\text{out}} \leftarrow q_{\text{out}} - \frac{r}{2 \Delta t}$$

### Strategy 3: Physics-Informed Neural Networks (PINNs)
When training deep neural networks (e.g. ST-GCN or Transformer) for Task 1, include physics penalty terms directly in the loss function:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{FD}} + \lambda_2 \mathcal{L}_{\text{conservation}}$$

---

## 4. Submission Instructions

- When submitting to Kaggle, ensure your Task 1 predictions are physics-refined.
- Your Task 1 rows in `sample_submission.csv` will automatically generate your Task 3 score on the leaderboard!
