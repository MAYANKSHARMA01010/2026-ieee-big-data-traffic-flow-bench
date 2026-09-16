# Task 2: Online Queue-Propagation Forecasting Guide

## Overview

**Task 2** focuses on **short-term queue and bottleneck forecasting** across freeway corridors. Freeway congestion spreads rapidly upstream as backward-propagating shockwaves. Highway management centers need advance warning to activate ramp metering, adjust variable speed limits, and deploy traveler advisories before queues fully form.

- **Weight in Overall Score**: **30%** (second highest weight in the competition).
- **History Window**: **60 minutes** (12 five-minute time steps, $[T-60, T)$).
- **Forecast Horizon**: **30 minutes** (6 five-minute time steps, $[T, T+30)$).

---

## 1. What You Predict

For each scored window and target cell in the 30-minute horizon:
- `queue_pred`: A binary classification value (`1` = queue present, `0` = free-flow / no queue).

**Columns in the Task 2 CSV export**:
```
window_id,timestamp,link_id,queue_pred
```

> **IMPORTANT**: Predictions must strictly be binary `0` or `1`. Continuous probabilities will fail validation.

---

## 2. Queue Definition & Critical Speed Thresholds

A highway link is considered **queued** when its measured speed drops below its link-specific critical cutoff:
$$\text{speed\_kmh} \le v_{\text{cut}}$$

Where $v_{\text{cut}}$ is obtained from:
1. `fd_parameters.csv` (`v_cut` column, derived from Fundamental Diagram calibration).
2. Fallback: $0.60 \times v_{\text{free}}$ (60% of free-flow speed from `links.csv`).

Only cells meeting the data quality threshold (`is_score_eligible == True`, i.e., `pct_observed >= 75%`) are score-eligible.

---

## 3. Window Conditions & Dataset Layout

Each corridor panel contains multiple evaluation windows under two distinct operational conditions:

1. **`queue_onset`**:
   - The visible 60-minute history shows **no established queue**, but a queue forms downstream during the 30-minute forecast horizon.
   - Tests the model's ability to detect impending breakdown before it happens.
2. **`queue_ongoing`**:
   - The history already shows an **established queue** ($\ge 2$ queued observations in history).
   - Tests the model's ability to predict queue dissipation or upstream shockwave propagation.

### Corridors Scored on Task 2

Task 2 is scored on **8 of the 10 panels**:
- `D7_I10_E` / `D7_I10_W`
- `D7_I210_E` / `D7_I210_W`
- `D7_I405_N` / `D7_I405_S`
- `D12_I5_N` / `D12_I5_S`

*(Note: `D12_I405_N` and `D12_I405_S` have no Task 2 windows).*

---

## 4. Evaluation Metric: Space-Time IoU

Each window is scored by comparing binary predictions $\hat{Y}$ against organizer ground truth $Y$ over all score-eligible link $\times$ time cells in the 30-minute horizon:

$$\text{IoU} = \frac{|\hat{Y} \cap Y|}{|\hat{Y} \cup Y|}$$

- If both $\hat{Y}$ and $Y$ are empty (no queue occurred and none was predicted), $\text{IoU} = 1.0$.
- Window scores are averaged across conditions (`queue_onset` and `queue_ongoing`), then across panels.

---

## 5. Modeling Strategies & Improvement Opportunities

### Baseline: Queue Persistence
The baseline takes the active queue status at $T - 5\text{ min}$ (the final observed step in history) and holds it constant across all 6 future time steps.

### Advanced Approaches to Boost IoU:
1. **Kinematic Shockwave Speed Propagation**:
   - Compute the upstream propagation velocity $w = \frac{\Delta q}{\Delta k}$.
   - Progressively shift queue boundaries to upstream links for $t = T+5, \dots, T+30$.
2. **LightGBM / XGBoost Binary Classifier**:
   - **Features**:
     - Current speed, flow, and density at $T-5, T-10, T-15$.
     - Downstream and upstream speeds and spatial gradients $\frac{\partial v}{\partial x}$.
     - Temporal rate of speed decline $\frac{\partial v}{\partial t}$.
     - On-ramp and off-ramp inflow/outflow near the link.
   - **Probability Calibration & Threshold Search**: Tune classification threshold specifically to maximize IoU rather than standard cross-entropy.
3. **Spatio-Temporal Graph Neural Networks (ST-GNNs)**:
   - Use the corridor topology graph (`lwr_mainline_topology.csv`) with DCRNN or Spatial-Temporal Graph Convolutional Networks (STGCN) to predict queue probability maps.

---

## 6. Export & Submission Pipeline

1. Run [`notebooks/task2_queue_forecasting.ipynb`](../notebooks/task2_queue_forecasting.ipynb).
2. The notebook generates [`datasets/task2_queue_submission.csv`](../datasets/task2_queue_submission.csv) (exactly 87,000 rows for `validation`).
3. Merge with Task 1 and Task 4 via the official merger:
   ```bash
   python trafficflowbench-public/src/merge_submissions.py \
     --state datasets/task1_state_submission.csv \
     --queue datasets/task2_queue_submission.csv \
     --odme  task4_submission.csv \
     --key   kaggle_public/submission_key.csv \
     --output submission.csv
   ```
