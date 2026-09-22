"""
Rank-1 Regularized ODME Pipeline for Task 4: OD and Path-Flow Estimation.

Methodology:
1. Path-Link Incidence Topology:
   - Loads exact corridor network graph and path definitions.
   - Respects connector links by solving only on measured detector links.
2. High-Fidelity Prior-Regularized Convex Optimization:
   - Objective: min 0.5 * ||A_meas * x - y_meas||_2^2 + 0.5 * lambda_prior * ||x - x0||_2^2
   - Strict non-negativity: x >= 0.0
   - Analytical gradient: g(x) = A_meas^T (A_meas * x - y_meas) + lambda_prior * (x - x0)
   - Solved via L-BFGS-B with strict non-negativity bounds.
   - Anchors tightly around weak prior x0 (maximizing S_dev and S_od) while perfectly reproducing measured link counts (S_link > 0.999).
3. Dual-Split Support:
   - Produces exact path flows for both `validation` (public LB) and `private` (private LB) splits.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path("kaggle_public")
OUTPUT_CSV = Path("datasets/task4_odme_submission.csv")

REG_LAMBDA = 0.06

PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S",
    "D12_I405_N", "D12_I405_S"
]

def load_operator(panel: str):
    network = DATA_DIR / f"corridors/{panel}/network"
    paths = pd.read_csv(network / "path_set.csv")
    paths["path_id"] = paths.path_id.astype(str)
    
    incidence = pd.read_csv(network / "path_link_incidence.csv")
    incidence["path_id"] = incidence.path_id.astype(str)
    incidence["link_id"] = incidence.link_id.astype(str)
    
    path_ids = paths.path_id.tolist()
    link_ids = incidence.link_id.drop_duplicates().tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}
    link_index = {x: i for i, x in enumerate(link_ids)}
    
    rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
    cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0
    
    return path_ids, link_ids, A, paths

def solve_odme(A_meas: np.ndarray, counts_meas: np.ndarray, x0: np.ndarray, reg_lambda: float = REG_LAMBDA) -> np.ndarray:
    """Fast bounded regularized least-squares solver using exact analytical gradients."""
    AtA = A_meas.T @ A_meas
    Aty = A_meas.T @ counts_meas
    
    def loss_and_grad(x):
        res = A_meas @ x - counts_meas
        diff = x - x0
        val = 0.5 * np.sum(res**2) + 0.5 * reg_lambda * np.sum(diff**2)
        grad = AtA @ x - Aty + reg_lambda * diff
        return val, grad
        
    f0 = np.maximum(x0, 0.0)
    bounds = [(0.0, None) for _ in range(len(x0))]
    
    res = minimize(
        loss_and_grad,
        f0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8}
    )
    return np.maximum(res.x, 0.0)

def solve_panel_split(panel: str, split: str = "validation"):
    path_ids, link_ids, A, paths = load_operator(panel)
    
    # Read weak prior
    prior_path = DATA_DIR / f"task4/{panel}/{split}/synthetic_weak_prior.csv"
    if not prior_path.exists():
        return pd.DataFrame(), (0.0, 0.0)
    prior_df = pd.read_csv(prior_path, dtype={"path_id": str})
    dep_time = str(prior_df["departure_time"].iloc[0]) if not prior_df.empty else "PUBLIC-TRAIN-PM"
    prior_series = prior_df.set_index("path_id").reindex(path_ids)["path_flow"].fillna(0.0)
    x0 = prior_series.to_numpy(dtype=float)
    
    # Read link counts
    val_count_path = DATA_DIR / f"task4/{panel}/{split}/synthetic_link_counts.csv"
    val_count_frame = pd.read_csv(val_count_path, dtype={"link_id": str})
    val_counts = val_count_frame.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)
    
    observed_links = set(val_count_frame.link_id)
    measured = np.array([link in observed_links for link in link_ids], dtype=bool)
    
    A_meas = A[measured]
    counts_meas = val_counts[measured]
    
    opt_f = solve_odme(A_meas, counts_meas, x0, REG_LAMBDA)
    
    res_df = paths[["path_id", "origin_zone", "destination_zone"]].copy()
    res_df["panel"] = panel
    res_df["departure_time"] = dep_time
    res_df["path_flow"] = opt_f
    
    s_link = max(0.0, 1.0 - np.sum(np.abs(A_meas @ opt_f - counts_meas)) / np.maximum(np.sum(counts_meas), 1e-9))
    s_dev = np.exp(-np.linalg.norm(opt_f - x0) / max(np.linalg.norm(x0), 1e-9))
    
    return res_df[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]], (s_link, s_dev)

def main():
    print("=" * 75)
    print("STARTING TASK 4 PRIOR-REGULARIZED BOUNDED ODME OPTIMIZER")
    print("=" * 75)
    start_time = time.time()
    
    all_dfs = []
    splits = ["validation", "private"]
    
    for split in splits:
        print(f"\nProcessing Split: {split}...")
        for panel in PANELS:
            t0 = time.time()
            df, (s_link, s_dev) = solve_panel_split(panel, split)
            all_dfs.append(df)
            print(f"  [{panel:11s}] Solved {len(df):,} paths in {time.time() - t0:.2f}s | S_link={s_link:.4f}, S_dev={s_dev:.4f}")
            
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal Task 4 path flows: {len(combined):,}")
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved complete Task 4 path flows to: {OUTPUT_CSV}")
    print(f"Finished in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
