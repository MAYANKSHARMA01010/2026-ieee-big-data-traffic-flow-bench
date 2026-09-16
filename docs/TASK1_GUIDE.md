# Task 1: Offline Traffic-State Reconstruction Guide

## Overview

**Task 1** focuses on **traffic state estimation (imputation)** across 10 directional freeway corridors. Freeway loop detectors frequently drop out or experience transmission outages. When sensors fail, models must reconstruct the missing traffic state from remaining observations and historical patterns.

- **Weight in Overall Score**: **35%** (highest weight of all tasks).
- **Physical Link**: Your Task 1 predictions are also directly evaluated on **Task 3 (Physics Consistency, 15%)**, making high-quality, physics-compliant reconstruction worth **50% of the entire competition**.

---

## 1. What You Predict

For every masked target cell identified in the split template:
- `speed_kmh`: Mean vehicle speed in kilometers per hour.
- `flow_vph`: Total vehicle flow (volume) in vehicles per hour.

**Columns in the Task 1 CSV export**:
```
panel,timestamp,station_id,link_id,mask_regime,speed_kmh,flow_vph
```

---

## 2. Masking Regimes & Evaluation Metric

The dataset features three sensor outage regimes, where each calendar day belongs to exactly one regime:
- **R1**: 20% sensor outage rate
- **R2**: 30% sensor outage rate
- **R3**: 50% sensor outage rate

### Evaluation Metric: Normalized RMSE ($S_{\text{state}}$)

$$\text{NRMSE}_{\text{speed}} = \frac{\text{RMSE}(\text{speed})}{25.0}$$

$$\text{NRMSE}_{\text{flow}} = \frac{\text{RMSE}(\text{flow} / \text{lanes})}{600.0}$$

$$\text{Regime Error} = 0.54 \times \text{NRMSE}_{\text{speed}} + 0.46 \times \text{NRMSE}_{\text{flow}}$$

$$S_{\text{regime}} = \exp(-\text{Regime Error})$$

The final panel score $S_{\text{state}}$ is the unweighted macro-mean across all three regimes:
$$S_{\text{state}} = \frac{1}{3} (S_{R1} + S_{R2} + S_{R3})$$

> **Note on Lanes**: Flow is evaluated **per lane** (divided by station lane count from `fd_parameters.csv`). A missing prediction receives 0 speed and flow, penalizing the score.

---

## 3. Recommended Improvement Ideas Beyond the Baseline

The baseline uses a link-specific **Weekday $\times$ 5-minute Time-of-Day slot historical mean** profile ($7 \times 288 = 2,016$ slots/week).

To achieve top leaderboard performance:

1. **Spatial Neighbor / Kriging Features**:
   - Traffic is spatially correlated along corridors. Incorporate real-time readings from adjacent upstream and downstream sensors observed during the same time step.
2. **Gradient Boosted Decision Trees (LightGBM / CatBoost)**:
   - Features: Time of day, day of week, upstream sensor speed/flow, downstream sensor speed/flow, rolling 15/30-minute moving averages, link road geometry (lanes, speed limit).
3. **Spatio-Temporal Graph Neural Networks (ST-GNNs)**:
   - Use the corridor link adjacency (`lwr_mainline_topology.csv`) with DCRNN, STGCN, or Spatio-Temporal Graph WaveNet.
4. **Physics Constraints (Safeguarding Task 3)**:
   - Clip predicted speeds and flows to follow Fundamental Diagram capacities ($v \le v_{\text{free}}$, $q \le q_{\text{cap}}$).
   - Ensure mass conservation across adjacent links.

---

## 4. Submission Pipeline

1. Run the Jupyter notebook [`notebooks/task1_traffic_state_estimation.ipynb`](../notebooks/task1_traffic_state_estimation.ipynb).
2. The notebook exports predictions to [`datasets/task1_state_submission.csv`](../datasets/task1_state_submission.csv).
3. Merge Task 1 with Task 2 and Task 4 outputs using the official merger:
   ```bash
   python trafficflowbench-public/src/merge_submissions.py \
     --state datasets/task1_state_submission.csv \
     --queue queue_submission.csv \
     --odme  task4_submission.csv \
     --key   kaggle_public/submission_key.csv \
     --output submission.csv
   ```
