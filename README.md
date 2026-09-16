# 2026 IEEE Big Data Cup: Traffic Flow Bench

Official repository for competing in the **2026 IEEE Big Data Cup — Traffic Flow Bench**.

- **Competition**: [Kaggle - 2026 IEEE Big Data Cup](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)
- **Dataset**: [Kaggle Dataset Download](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
- **Official Public Baselines & Code**: [GitHub - jacky850/trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)

---

## What You Have To Do in This Hackathon (In Short)

The challenge is to reconstruct, forecast, and estimate highway traffic across **10 freeway corridors** (5 freeways, both directions) using sensor data, network physics, and origin-destination routes.

The competition evaluates a **single combined submission file** scored across **4 complementary tasks**:

$$\text{Overall Score} = 0.35 \times S_{\text{state}} + 0.30 \times S_{\text{queue}} + 0.15 \times S_{\text{physics}} + 0.20 \times S_{\text{ODME}}$$

### The 4 Scored Tasks

| Task | Weight | What to Predict | What it does |
| :--- | :---: | :--- | :--- |
| **Task 1: Traffic State Estimation** | **35%** | `speed_kmh`, `flow_vph` | Impute missing traffic readings where detectors went offline (under 20%, 30%, and 50% sensor outage regimes). |
| **Task 2: Queue / Bottleneck Forecasting** | **30%** | `queue_pred` (0 or 1) | Given 60 minutes of history, predict binary queue formation across links for the next 30-minute horizon (evaluated by space-time IoU). |
| **Task 3: Physics Consistency** | **15%** | *(Scored on Task 1)* | **No separate file needed.** Evaluates whether your Task 1 speeds and flows obey vehicle conservation (LWR model) and fundamental diagrams. |
| **Task 4: Origin-Destination Matrix Estimation** | **20%** | `path_flow` | Estimate path-level traffic volume matching observed link sensor counts and weak prior matrices. |

---

## How Submissions Work: `submission_key.csv` & Merging

You upload a single CSV file: [`sample_submission.csv`](file:///Users/mayanksharma/Downloads/New_Projects/Kaggle-Hackathons/2026-ieee-big-data-traffic-flow-bench/kaggle_public/sample_submission.csv) (~6.98 million rows).

### The Role of [`submission_key.csv`](file:///Users/mayanksharma/Downloads/New_Projects/Kaggle-Hackathons/2026-ieee-big-data-traffic-flow-bench/kaggle_public/submission_key.csv)
`submission_key.csv` is the master reference that maps every `submission_id` to its exact context:
- `task`: `state`, `queue`, or `odme`
- `split`: `validation` (public leaderboard) or `private` (private leaderboard)
- `panel`: e.g., `D7_I10_E`, `D12_I5_N`
- `timestamp`, `station_id`, `link_id`, `mask_regime`, `window_id`, `path_id`, `origin_zone`, `destination_zone`

### Submission Columns
For each row in the submission:
- If `task == "state"`: fill `speed_kmh` and `flow_vph`, set other columns to `0`.
- If `task == "queue"`: fill `queue_pred` (binary `0` or `1`), set other columns to `0`.
- If `task == "odme"`: fill `path_flow`, set other columns to `0`.
- **Never leave any cell empty** (nulls cause immediate rejection).

---

## Repository & Data Layout

```
.
├── README.md
├── trafficflowbench-public/        # Cloned official toolkit & baselines
│   ├── docs/RUN_LOCAL.md          # Step-by-step local guide
│   ├── requirements.txt           # Python dependencies
│   └── src/
│       ├── merge_submissions.py   # Merges task predictions into final submission.csv
│       ├── score_overall.py       # Local overall scorer
│       ├── task1/                 # Task 1 baseline builder & scorer
│       ├── task2/                 # Task 2 baseline builder & scorer
│       ├── task3/                 # Task 3 physics evaluator
│       └── task4/                 # Task 4 ODME builder & scorer
└── kaggle_public/                 # Extracted competition dataset (~12 GB, 9688 files)
    ├── README.md
    ├── sample_submission.csv      # Upload template
    ├── submission_key.csv         # Submission lookup index
    ├── config/                    # Contracts, calendar, and corridor configs
    ├── corridors/<PANEL>/         # Network topology and train/val/private observations
    ├── task1/<PANEL>/<split>/     # Per-corridor Task 1 templates
    ├── task2/<PANEL>/<split>/     # Per-corridor Task 2 history parquets & windows
    └── task4/<PANEL>/<split>/     # Per-corridor Task 4 link counts & weak priors
```

---

## Quickstart: Running Locally

### 1. Install Dependencies
```bash
pip install -r trafficflowbench-public/requirements.txt
```

### 2. Set Environment Variable
```bash
export REL="$(pwd)/kaggle_public"
```

### 3. Build Baseline Predictions

```bash
# Task 1: Traffic State (validation split)
python trafficflowbench-public/src/task1/build_task1_baseline_submission.py \
  --release-root "$REL" --split validation --output state_submission.csv

# Task 2: Queue Persistence (validation split)
python trafficflowbench-public/src/task2/build_task2_persistence_submission.py \
  --release-root "$REL" --split validation --output queue_submission.csv

# Task 4: ODME (validation split)
python trafficflowbench-public/src/task4/build_task4_odme_artifacts.py \
  --release-root "$REL" --split validation --output-root task4_odme
```

### 4. Merge into Final Submission via `submission_key.csv`
Use the official merger to combine all task outputs into a submission-ready CSV:
```bash
python trafficflowbench-public/src/merge_submissions.py \
  --state state_submission.csv \
  --queue queue_submission.csv \
  --odme  task4_odme/baseline_submission.csv \
  --key   "$REL/submission_key.csv" \
  --output submission.csv
```

---

## Splits & Validation Strategy

- **`train` (June 2030 – Feb 2031, 273 days)**: Contains both masked inputs and unmasked true observations. Use this split to train your models and evaluate `score_task1.py` locally.
- **`validation` (March 2031, 31 days)**: Evaluates the **Public Leaderboard**.
- **`private` (April 2031, 30 days)**: Evaluates the **Private Leaderboard** (decides the final winners).