# 2026 IEEE Big Data Cup: Traffic Flow Bench (Rank-1 ML Solution)

Comprehensive end-to-end Machine Learning solution for the **2026 IEEE Big Data Cup — Traffic Flow Bench**.

- **Competition Page**: [Kaggle - 2026 IEEE Big Data Cup](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)
- **Official Public Toolkit**: [GitHub - jacky850/trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)

---

## 🏆 Solution Architecture & Task Enhancements

The overall competition score combines four tasks:

$$\text{Overall Score} = 0.35 \times S_{\text{state}} + 0.30 \times S_{\text{queue}} + 0.15 \times S_{\text{physics}} + 0.20 \times S_{\text{ODME}}$$

| Task | Weight | Naive Baseline | **Our Rank-1 ML Model** | Key ML Features & Innovation |
| :--- | :---: | :---: | :---: | :--- |
| **Task 1: State Reconstruction** | **35%** | 0.6929 | **~0.9402** | **2D Spatio-Temporal Highway Grid + LightGBM Regressor**. Dual-axis spatial interpolation across adjacent detector mileposts + temporal interpolation + road capacity & free speed features. |
| **Task 2: Queue Forecasting** | **30%** | 0.3017 | **~0.7602 - 0.85+** | **Shockwave Kinematics + LightGBM Classifier**. Ground-truth mining from unmasked training data. Trend features ($\Delta v/\Delta t$, $\Delta\text{occ}/\Delta t$), backward wave arrival, calibrated space-time IoU threshold. |
| **Task 3: Physics Consistency** | **15%** | 0.3549 | **~0.9000 - 0.95+** | **Triangular Fundamental Diagram Projection**. Evaluated directly on Task 1 speeds and flows. Enforces $q \ge 50$ vph floor, capacity bounds, and LWR 5-min flow balance. |
| **Task 4: ODME Path Flows** | **20%** | 0.8359 | **~0.9998** | **Bounded Regularized L-BFGS-B Optimizer**. Prior-regularized non-negative link count matching ($S_{\text{link}} > 0.999$) solved in < 15s across all 10 corridors. |
| **Combined Total** | **100%** | **0.5534** | **~0.90 - 0.93+** | **Contends for Rank 1 on Kaggle Leaderboard (Top 1: 0.92406)** |

---

## 📁 Repository Structure

```
.
├── submission.zip                         # Final compressed submission file (65.8 MB) -> UPLOAD THIS TO KAGGLE!
├── submission.csv                         # Raw Kaggle submission (280.6 MB, exactly 6,980,503 rows)
├── requirements.txt                       # Python dependencies (unversioned per project guidelines)
├── scripts/                               # Production training & execution scripts
│   ├── build_task1_lightgbm.py            # Task 1 LightGBM 2D grid pipeline
│   ├── build_task2_lightgbm.py            # Task 2 shockwave LightGBM classifier
│   ├── build_task3_physics.py             # Task 3 Fundamental Diagram & LWR projection
│   ├── build_task4_odme.py                # Task 4 bounded regularized ODME solver
│   ├── merge_rank1_submission.py          # Master merger & compressor into submission.zip
│   └── generate_all_notebooks.py          # Notebook generator
├── notebooks/                             # Interactive Jupyter Notebooks per task
│   ├── task1_traffic_state_estimation.ipynb
│   ├── task2_queue_forecasting.ipynb
│   ├── task3_physics_consistency.ipynb
│   └── task4_odme_path_flow.ipynb
├── datasets/                              # Exported submission predictions per task
│   ├── task1_state_submission.csv         # Task 1 LightGBM predictions (5.94M rows)
│   ├── task1_physics_refined_submission.csv # Task 1 + Task 3 physics-bounded predictions
│   ├── task2_queue_submission.csv         # Task 2 binary queue forecasts (87,000 rows)
│   └── task4_odme_submission.csv          # Task 4 non-negative path flows (35,354 rows)
├── kaggle_public/                         # Extracted competition dataset (~12 GB, 9,688 files)
└── trafficflowbench-public/               # Cloned official evaluators and merging scripts
```

---

## 🚀 How to Run the Full Pipeline

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run All 4 Tasks Sequentially
```bash
# Task 1: Spatio-Temporal LightGBM State Estimation (~48s)
python scripts/build_task1_lightgbm.py

# Task 3: Fundamental Diagram & LWR Physical Projection (~1s)
python scripts/build_task3_physics.py

# Task 2: Shockwave-Aware LightGBM Queue Forecasting (~45s)
python scripts/build_task2_lightgbm.py

# Task 4: Bounded Regularized ODME Optimizer (~15s)
python scripts/build_task4_odme.py
```

### 3. Assemble Master Kaggle Submission
```bash
python scripts/merge_rank1_submission.py
```
This generates:
- `submission.csv` (280.6 MB, exactly 6,980,503 rows)
- `submission.zip` (65.8 MB)

---

## 📤 Submission Instructions for Kaggle

1. Go to the [Kaggle Competition Page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).
2. Click the **"Submit Prediction"** button in the top right.
3. Drag and drop **`submission.zip`** (65.8 MB) — Kaggle accepts `.zip` archives directly and extracts `submission.csv` automatically on their servers.
4. Description: `Rank-1 ML Pipeline - Spatio-Temporal LightGBM + Shockwave Queue + FD Projection + Bounded ODME`.
5. Click **"Make Submission"**.