# 2026 IEEE Big Data Cup: Traffic Flow Bench

Official benchmark repository for the **2026 IEEE Big Data Cup - Traffic Flow Bench**.

- **Competition Page**: [Kaggle - 2026 IEEE Big Data Cup](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)
- **Data Page**: [Kaggle Data Download](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
- **Official Public Baselines & Evaluators**: [GitHub - jacky850/trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)

---

## Dataset Setup

The dataset package has been extracted to [`kaggle_public/`](./kaggle_public/) (which is ignored by git to keep the repository lightweight).

- **Total Extracted Size**: ~12 GB (9,688 files)
- **Folder Location**: `kaggle_public/`

### Directory Layout

```
kaggle_public/
├── README.md
├── sample_submission.csv          # File to fill and upload for leaderboard evaluation
├── submission_key.csv             # Maps submission_id to panel, timestamp, station, link, etc.
├── config/
│   ├── synthetic_release_v1.json  # Calendar & release contract
│   ├── corridors.json             # 10 directional freeway panels
│   └── competition_contract.json  # Task definitions & scoring weights
├── corridors/<PANEL>/
│   ├── network/                   # Topology, links, capacity, FD parameters, OD prior
│   ├── train/                     # Observations (mainline_states, mainline_states_masked, ramp_states)
│   ├── validation/                # Public leaderboard split (mainline_states_masked, ramp_states)
│   └── private/                   # Private leaderboard split (mainline_states_masked, ramp_states)
├── task1/<PANEL>/<split>/         # Sample submission templates for Task 1 (State Estimation)
├── task2/<PANEL>/<split>/         # Window index, history parquet, queue templates for Task 2
└── task4/<PANEL>/<split>/         # Synthetic link counts, weak prior, path flow templates for Task 4
```

---

## Corridors

Ten directional panels across five major freeway corridors:
- `D7_I10_E` / `D7_I10_W`
- `D7_I210_E` / `D7_I210_W`
- `D7_I405_N` / `D7_I405_S`
- `D12_I5_N` / `D12_I5_S`
- `D12_I405_N` / `D12_I405_S` *(No Task 2 windows; scored on tasks 1, 3, 4)*

---

## Tasks Overview

1. **Task 1 (Traffic State Estimation)**: Impute missing `speed_kmh` and `flow_vph` for masked detector cells across R1 (20%), R2 (30%), and R3 (50%) outage regimes.
2. **Task 2 (Queue Formation & Bottleneck Prediction)**: Predict binary queue formation (`queue_pred`) for 30-minute horizons.
3. **Task 3 (Physics Consistency)**: Scored directly on your Task 1 predictions for consistency with traffic flow conservation and fundamental diagrams.
4. **Task 4 (Origin-Destination Matrix Estimation - ODME)**: Estimate path flows (`path_flow`) matching observed link volumes and prior OD matrices.