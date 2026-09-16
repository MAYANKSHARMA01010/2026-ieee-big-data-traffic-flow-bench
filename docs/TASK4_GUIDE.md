# Task 4: Origin-Destination Matrix Estimation (ODME) Guide

## Overview

**Task 4** focuses on **Origin-Destination Matrix Estimation (ODME)** and path flow reconstruction across 10 freeway corridors. Loop detectors record the aggregate vehicle volume passing a specific point on the highway, but they **cannot identify where drivers entered or where they are heading**.

ODME solves the inverse problem of estimating granular route path volumes from aggregate link sensor counts, network topology, and weak prior estimates.

- **Weight in Overall Score**: **20%**
- **Evaluation Period**: PM peak periods across all 10 freeway corridors.
- **Output Target**: Continuous `path_flow` (volume of vehicles taking each path).

---

## 1. What You Predict

For each origin-destination path identified in the network:
- `path_flow`: Non-negative continuous volume of vehicles traversing that route.

**Columns in the Task 4 CSV export**:
```
panel,departure_time,path_id,origin_zone,destination_zone,path_flow
```

---

## 2. Mathematical Formulation: Regularized NNLS

Let:
- $x \in \mathbb{R}_+^N$: Path flow vector to estimate ($N$ paths).
- $y \in \mathbb{R}^M$: Observed vehicle volume counts on $M$ detector links.
- $A \in \{0, 1\}^{M \times N}$: Linear link-path incidence matrix, where $A_{ij} = 1$ if link $i$ is on path $j$.
- $x_0 \in \mathbb{R}_+^N$: Weak prior path flow matrix.

Because multiple route combinations can yield the same link counts (ill-posed inverse problem), we formulate a **regularized convex optimization problem**:

$$\min_{x \ge 0} \mathcal{L}(x) = \frac{1}{2} \|A_{\text{obs}} x - y_{\text{obs}}\|_2^2 + \frac{\lambda}{2} \|x - x_0\|_2^2$$

Where:
- The first term ensures path flows reproduce measured link counts ($A x \approx y$).
- The second term ($\lambda = 0.05$) anchors solutions to realistic prior trip distributions and prevents unphysical flow concentrations.
- Analytical gradient:
  $$\nabla \mathcal{L}(x) = A_{\text{obs}}^T (A_{\text{obs}} x - y_{\text{obs}}) + \lambda (x - x_0)$$

---

## 3. Evaluation Metric ($S_{\text{ODME}}$)

The competition evaluates Task 4 across 4 criteria:

$$S_{\text{ODME}} = 0.45 \times S_{\text{od}} + 0.25 \times S_{\text{link}} + 0.15 \times S_{\text{dev}} + 0.15 \times S_{\text{attr}}$$

1. **$S_{\text{od}}$ (45% weight)**: Direct accuracy against organizer ground truth path flows (evaluated on leaderboard).
2. **$S_{\text{link}}$ (25% weight)**: Link count reproduction accuracy ($\exp(-\text{NRMSE}_{\text{link}})$). Computable locally from released counts!
3. **$S_{\text{dev}}$ (15% weight)**: Deviation penalty preventing models from drifting excessively from the weak prior.
4. **$S_{\text{attr}}$ (15% weight)**: Preservation of destination zone total trip attraction.

---

## 4. Advanced Modeling Techniques to Boost $S_{\text{ODME}}$

1. **Generalized Least Squares (GLS) with Variance Weighting**:
   - Instead of equal weights across all links, weight links inversely proportional to measurement variance:
     $$\min_{x \ge 0} (A x - y)^T V^{-1} (A x - y) + \lambda (x - x_0)^T W^{-1} (x - x_0)$$
2. **Entropy Maximization / Information Minimization**:
   - Add an entropy regularization term $- \sum x_i \ln(x_i / x_{0,i})$ to find the most probable route distribution matching link observations.
3. **Bi-Level Dynamic Traffic Assignment (DTA)**:
   - Upper level: Adjust OD matrix to match link volumes.
   - Lower level: Equilibrium route choice where drivers take shortest travel-time paths given congested link travel times from Task 1.

---

## 5. Master Submission Assembly

Once all individual task outputs are exported:
1. `datasets/task1_physics_refined_submission.csv` (Task 1 & Task 3)
2. `datasets/task2_queue_submission.csv` (Task 2)
3. `datasets/task4_odme_submission.csv` (Task 4)

Assemble the single upload file:
```bash
python trafficflowbench-public/src/merge_submissions.py \
  --state datasets/task1_physics_refined_submission.csv \
  --queue datasets/task2_queue_submission.csv \
  --odme  datasets/task4_odme_submission.csv \
  --key   kaggle_public/submission_key.csv \
  --output submission.csv
```
This produces the final `submission.csv` ready for Kaggle leaderboard evaluation.
