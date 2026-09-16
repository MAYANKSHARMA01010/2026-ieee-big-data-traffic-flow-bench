# 2026 IEEE Big Data Cup: Traffic Flow Bench

Official repository and complete benchmark solution for the **2026 IEEE Big Data Cup — Traffic Flow Bench**.

- **Competition Page**: [Kaggle - 2026 IEEE Big Data Cup](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)
- **Dataset Download**: [Kaggle Dataset](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
- **Official Public Toolkit**: [GitHub - jacky850/trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)

---

## 🏆 Project Overview & Tasks Suite

The challenge evaluates highway traffic modeling across **10 freeway corridors** under a single unified submission:

$$\text{Overall Score} = 0.35 \times S_{\text{state}} + 0.30 \times S_{\text{queue}} + 0.15 \times S_{\text{physics}} + 0.20 \times S_{\text{ODME}}$$

| Task | Weight | Target | Notebook | Documentation | Exported CSV |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Task 1: State Estimation** | **35%** | `speed_kmh`, `flow_vph` | [`notebooks/task1_traffic_state_estimation.ipynb`](notebooks/task1_traffic_state_estimation.ipynb) | [`docs/TASK1_GUIDE.md`](docs/TASK1_GUIDE.md) | `datasets/task1_state_submission.csv` |
| **Task 2: Queue Forecasting** | **30%** | `queue_pred` (0 or 1) | [`notebooks/task2_queue_forecasting.ipynb`](notebooks/task2_queue_forecasting.ipynb) | [`docs/TASK2_GUIDE.md`](docs/TASK2_GUIDE.md) | `datasets/task2_queue_submission.csv` |
| **Task 3: Physics Consistency** | **15%** | *(Scored on Task 1)* | [`notebooks/task3_physics_consistency.ipynb`](notebooks/task3_physics_consistency.ipynb) | [`docs/TASK3_GUIDE.md`](docs/TASK3_GUIDE.md) | `datasets/task1_physics_refined_submission.csv` |
| **Task 4: ODME Path Flows** | **20%** | `path_flow` | [`notebooks/task4_odme_path_flow.ipynb`](notebooks/task4_odme_path_flow.ipynb) | [`docs/TASK4_GUIDE.md`](docs/TASK4_GUIDE.md) | `datasets/task4_odme_submission.csv` |

---

## 📁 Repository Structure

```
.
├── README.md                              # Main documentation and workflow guide
├── requirements.txt                       # Python dependencies (unversioned)
├── submission.csv                         # Final assembled Kaggle submission (6,980,503 rows)
├── notebooks/                             # Interactive Jupyter Notebooks per task
│   ├── task1_traffic_state_estimation.ipynb
│   ├── task2_queue_forecasting.ipynb
│   ├── task3_physics_consistency.ipynb
│   └── task4_odme_path_flow.ipynb
├── docs/                                  # Deep-dive guides and math specifications
│   ├── TASK1_GUIDE.md
│   ├── TASK2_GUIDE.md
│   ├── TASK3_GUIDE.md
│   └── TASK4_GUIDE.md
├── datasets/                              # Exported submission predictions per task
│   ├── task1_state_submission.csv         # Task 1 baseline predictions (~54 MB)
│   ├── task1_physics_refined_submission.csv# Task 1 + Task 3 physics-bounded predictions (~54 MB)
│   ├── task2_queue_submission.csv         # Task 2 binary queue forecasts (~5.4 MB)
│   └── task4_odme_submission.csv          # Task 4 non-negative path flows (~2.1 MB)
├── kaggle_public/                         # Extracted competition dataset (~12 GB, 9,688 files)
│   ├── sample_submission.csv              # Official Kaggle template
│   ├── submission_key.csv                 # Master submission index
│   ├── config/                            # Contracts, calendar, corridor definitions
│   ├── corridors/<PANEL>/                 # Network topologies and train/val/private observations
│   ├── task1/<PANEL>/<split>/             # Task 1 templates
│   ├── task2/<PANEL>/<split>/             # Task 2 parquets and window indices
│   └── task4/<PANEL>/<split>/             # Task 4 link counts and weak priors
└── trafficflowbench-public/               # Cloned official evaluators and merging scripts
```

---

## 🚀 Environment Setup

Initialize the virtual environment and install all packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Task Execution

### 1. Task 1: Traffic State Estimation
- **Problem**: Impute missing `speed_kmh` and `flow_vph` from detector outages (20%, 30%, 50% dropouts).
- **Run Notebook**: [`notebooks/task1_traffic_state_estimation.ipynb`](notebooks/task1_traffic_state_estimation.ipynb)
- **Output**: `datasets/task1_state_submission.csv`

### 2. Task 2: Online Queue Forecasting
- **Problem**: Given 60 minutes of history, predict binary queue formation (`queue_pred`) across the next 30-minute horizon.
- **Run Notebook**: [`notebooks/task2_queue_forecasting.ipynb`](notebooks/task2_queue_forecasting.ipynb)
- **Output**: `datasets/task2_queue_submission.csv` (87,000 binary predictions)

### 3. Task 3: Physics Consistency
- **Problem**: Ensure speeds and flows obey the triangular Fundamental Diagram and LWR vehicle conservation laws.
- **Run Notebook**: [`notebooks/task3_physics_consistency.ipynb`](notebooks/task3_physics_consistency.ipynb)
- **Output**: `datasets/task1_physics_refined_submission.csv` (applies capacity & speed bounds to Task 1)

### 4. Task 4: Origin-Destination Matrix Estimation (ODME)
- **Problem**: Estimate continuous route flows (`path_flow`) from link sensor counts and weak priors using regularized NNLS.
- **Run Notebook**: [`notebooks/task4_odme_path_flow.ipynb`](notebooks/task4_odme_path_flow.ipynb)
- **Output**: `datasets/task4_odme_submission.csv` (35,354 path flows)

---

## 📦 Assembling the Master Submission (`submission.csv`)

To merge all individual task outputs into the final upload file keyed by `submission_key.csv`:

```bash
python trafficflowbench-public/src/merge_submissions.py \
  --state datasets/task1_physics_refined_submission.csv \
  --queue datasets/task2_queue_submission.csv \
  --odme  datasets/task4_odme_submission.csv \
  --key   kaggle_public/submission_key.csv \
  --output submission.csv
```

### Verification of `submission.csv`:
- **Total Rows**: Exactly **6,980,503 rows** (matching `sample_submission.csv`).
- **Columns**: `submission_id`, `task`, `speed_kmh`, `flow_vph`, `queue_pred`, `path_flow`.
- **Integrity**: 0 NaNs / nulls, valid non-negative values.
- **Kaggle Ready**: Ready to be uploaded directly to the competition leaderboard!