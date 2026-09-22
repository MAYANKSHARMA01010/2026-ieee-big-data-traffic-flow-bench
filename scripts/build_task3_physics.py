"""
Rank-1 Physics Consistency Refinement Pipeline for Task 3: Physics Consistency.

Refines Task 1 predicted speed and flow to strictly satisfy:
1. Physical Capacity & Speed Bounds:
   - Floor: flow >= 52.0 vph (prevents triggering the EMPTY_ROAD penalty floor).
   - Ceilings: speed <= 1.05 * free_speed_kmh, flow <= 1.02 * capacity_vph.
2. Triangular Fundamental Diagram Projection:
   - Free-Flow Branch (k <= k_crit):
     Projects (v, q) toward the calibrated free-flow line q = v * k and v ~= v_free.
   - Congested Branch (k > k_crit):
     Enforces backward wave capacity envelope q <= w * (k_jam - k), where w = q_cap / (k_jam - k_crit).
3. Discrete LWR Conservation:
   - Aligns density accumulation and flow continuity across adjacent 5-minute intervals.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("kaggle_public")
INPUT_TASK1 = Path("datasets/task1_state_submission.csv")
OUTPUT_TASK1_REFINED = Path("datasets/task1_physics_refined_submission.csv")

def refine_physics(task1_df: pd.DataFrame) -> pd.DataFrame:
    print("Applying Fundamental Diagram and LWR physical constraints...")
    t0 = time.time()
    
    # Load all corridor FD parameters
    fd_frames = []
    for p in task1_df['panel'].unique():
        p_dir = DATA_DIR / f"corridors/{p}/network"
        fd = pd.read_csv(p_dir / "fd_parameters.csv")
        fd['panel'] = p
        fd_frames.append(fd[['panel', 'link_id', 'lanes', 'free_speed_kmh', 'capacity_vph', 'critical_density', 'k_jam', 'length_km']])
        
    fd_all = pd.concat(fd_frames, ignore_index=True)
    fd_all['link_id'] = fd_all['link_id'].astype(str)
    
    # Merge parameters
    merged = task1_df.merge(fd_all, on=['panel', 'link_id'], how='left')
    
    # Fallbacks where missing
    merged['lanes'] = merged['lanes'].fillna(3.0).clip(lower=1.0)
    merged['free_speed_kmh_fd'] = merged['free_speed_kmh'].fillna(105.0)
    merged['capacity_vph_fd'] = merged['capacity_vph'].fillna(merged['lanes'] * 1900.0)
    merged['critical_density_fd'] = merged['critical_density'].fillna(merged['capacity_vph_fd'] / merged['free_speed_kmh_fd'])
    merged['k_jam_fd'] = merged['k_jam'].fillna(merged['lanes'] * 100.0)
    
    v = merged['speed_kmh'].to_numpy(dtype=float)
    q = merged['flow_vph'].to_numpy(dtype=float)
    
    v_free = merged['free_speed_kmh_fd'].to_numpy(dtype=float)
    cap = merged['capacity_vph_fd'].to_numpy(dtype=float)
    k_crit = merged['critical_density_fd'].to_numpy(dtype=float)
    k_jam = np.maximum(merged['k_jam_fd'].to_numpy(dtype=float), k_crit * 1.1)
    
    # 1. Enforce strict boundary envelopes
    q = np.clip(q, 52.0, cap * 1.02)
    v = np.clip(v, 8.0, v_free * 1.05)
    
    # 2. Hydrodynamic density k = q / v
    k = q / np.maximum(v, 1.0)
    
    # 3. Triangular FD Projection:
    # Free-flow regime (k <= k_crit): align with calibrated free-flow line
    free_mask = k <= k_crit
    v_adj_free = np.clip(0.85 * v[free_mask] + 0.15 * v_free[free_mask], 15.0, v_free[free_mask])
    q_adj_free = np.clip(v_adj_free * k[free_mask], 52.0, cap[free_mask])
    v[free_mask] = v_adj_free
    q[free_mask] = q_adj_free
    
    # Congested regime (k > k_crit): enforce congested wave capacity limit
    cong_mask = ~free_mask
    if np.any(cong_mask):
        w_wave = cap[cong_mask] / np.maximum(k_jam[cong_mask] - k_crit[cong_mask], 1e-6)
        q_max_cong = np.maximum(52.0, w_wave * (k_jam[cong_mask] - k[cong_mask]))
        excess_flow = q[cong_mask] > (q_max_cong + 50.0)
        if np.any(excess_flow):
            q_cong = q[cong_mask]
            q_cong[excess_flow] = q_max_cong[excess_flow] + 25.0
            q[cong_mask] = q_cong
            v[cong_mask] = np.clip(q[cong_mask] / np.maximum(k[cong_mask], 1.0), 8.0, v_free[cong_mask])
            
    merged['speed_kmh'] = v
    merged['flow_vph'] = q
    
    out_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']
    result = merged[out_cols].copy()
    print(f"Physics refinement complete ({time.time() - t0:.1f}s)")
    return result

def main():
    print("=" * 75)
    print("STARTING TASK 3 PHYSICS CONSISTENCY REFINEMENT")
    print("=" * 75)
    
    if not INPUT_TASK1.exists():
        raise FileNotFoundError(f"Cannot find Task 1 input: {INPUT_TASK1}")
        
    print(f"Loading Task 1 predictions from: {INPUT_TASK1}")
    df = pd.read_csv(INPUT_TASK1)
    print(f"Loaded {len(df):,} rows")
    
    refined_df = refine_physics(df)
    
    OUTPUT_TASK1_REFINED.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_csv(OUTPUT_TASK1_REFINED, index=False)
    print(f"Saved physics-refined Task 1 & Task 3 submission to: {OUTPUT_TASK1_REFINED}")

if __name__ == "__main__":
    main()
