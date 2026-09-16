"""
Assemble the Master Unified Submission file for Kaggle.

Format:
  submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
Total rows: exactly 6,980,503 rows matching sample_submission.csv.
"""
import time
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

KEY_PATH = Path("kaggle_public/submission_key.csv")
STATE_PATH = Path("datasets/task1_physics_refined_submission.csv")
QUEUE_PATH = Path("datasets/task2_queue_submission.csv")
ODME_PATH = Path("datasets/task4_odme_submission.csv")

OUTPUT_CSV = Path("submission.csv")
OUTPUT_ZIP = Path("submission.zip")

CHUNK_SIZE = 1_000_000

def main():
    print("=" * 70)
    print("ASSEMBLING MASTER RANK-1 SUBMISSION FOR KAGGLE")
    print("=" * 70)
    start_time = time.time()
    
    # 1. Load Task 1 (State & Physics)
    print("\nLoading Task 1 (Physics-Refined State)...")
    t0 = time.time()
    state_df = pd.read_csv(STATE_PATH)
    state_df['timestamp'] = pd.to_datetime(state_df['timestamp'], utc=True)
    state_keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    state_df = state_df.drop_duplicates(state_keys).set_index(state_keys)
    print(f"  Loaded {len(state_df):,} state rows in {time.time() - t0:.1f}s")
    
    # 2. Load Task 2 (Queue)
    print("Loading Task 2 (LightGBM Queue Predictions)...")
    t0 = time.time()
    queue_df = pd.read_csv(QUEUE_PATH)
    queue_df['timestamp'] = pd.to_datetime(queue_df['timestamp'], utc=True)
    queue_keys = ["window_id", "timestamp", "link_id"]
    queue_df = queue_df.drop_duplicates(queue_keys).set_index(queue_keys)
    print(f"  Loaded {len(queue_df):,} queue rows in {time.time() - t0:.1f}s")
    
    # 3. Load Task 4 (ODME Path Flows)
    print("Loading Task 4 (Bounded ODME Path Flows)...")
    t0 = time.time()
    odme_df = pd.read_csv(ODME_PATH)
    odme_keys = ["panel", "departure_time", "path_id"]
    odme_df = odme_df.drop_duplicates(odme_keys).set_index(odme_keys)
    print(f"  Loaded {len(odme_df):,} ODME rows in {time.time() - t0:.1f}s")
    
    # 4. Stream submission_key.csv and populate matching rows
    print("\nStreaming submission_key.csv and mapping predictions...")
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        
    written = 0
    first = True
    gaps = {"state": 0, "queue": 0, "odme": 0}
    
    for chunk in pd.read_csv(KEY_PATH, chunksize=CHUNK_SIZE, dtype=str):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        out = pd.DataFrame({
            "submission_id": chunk["submission_id"].astype("int64"),
            "task": chunk["task"].astype(str),
            "speed_kmh": 0.0,
            "flow_vph": 0.0,
            "queue_pred": 0.0,
            "path_flow": 0.0
        })
        
        # Match State
        is_state = out["task"] == "state"
        if is_state.any():
            state_idx = pd.MultiIndex.from_frame(chunk.loc[is_state, state_keys])
            matched_s = state_df.reindex(state_idx)
            s_vals = matched_s["speed_kmh"].to_numpy()
            q_vals = matched_s["flow_vph"].to_numpy()
            gaps["state"] += int(pd.isna(s_vals).sum())
            out.loc[is_state, "speed_kmh"] = pd.Series(s_vals).fillna(0.0).to_numpy()
            out.loc[is_state, "flow_vph"] = pd.Series(q_vals).fillna(0.0).to_numpy()
            
        # Match Queue
        is_queue = out["task"] == "queue"
        if is_queue.any():
            queue_idx = pd.MultiIndex.from_frame(chunk.loc[is_queue, queue_keys])
            matched_q = queue_df.reindex(queue_idx)
            q_pred_vals = matched_q["queue_pred"].to_numpy()
            gaps["queue"] += int(pd.isna(q_pred_vals).sum())
            out.loc[is_queue, "queue_pred"] = pd.Series(q_pred_vals).fillna(0.0).to_numpy()
            
        # Match ODME
        is_odme = out["task"] == "odme"
        if is_odme.any():
            odme_idx = pd.MultiIndex.from_frame(chunk.loc[is_odme, odme_keys])
            matched_od = odme_df.reindex(odme_idx)
            path_vals = matched_od["path_flow"].to_numpy()
            gaps["odme"] += int(pd.isna(path_vals).sum())
            out.loc[is_odme, "path_flow"] = pd.Series(path_vals).fillna(0.0).to_numpy()
            
        out_cols = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
        out[out_cols].to_csv(OUTPUT_CSV, mode="a", header=first, index=False)
        first = False
        written += len(out)
        print(f"  Processed {written:,} rows...", end="\r", flush=True)
        
    print(f"\nWrote exactly {written:,} rows to {OUTPUT_CSV}")
    print("Coverage report across tasks:")
    for t_name, n_gap in gaps.items():
        print(f"  {t_name:6s}: unmatched gaps = {n_gap:,}")
        
    # 5. Compress to submission.zip
    print("\nCompressing to submission.zip for fast Kaggle upload...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(OUTPUT_CSV, arcname="submission.csv")
        
    csv_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    zip_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"submission.csv size: {csv_mb:.1f} MB")
    print(f"submission.zip size: {zip_mb:.1f} MB")
    print(f"\nCompleted all steps in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
