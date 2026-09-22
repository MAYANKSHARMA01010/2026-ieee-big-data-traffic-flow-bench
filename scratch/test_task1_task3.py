"""
Test Task 1 & Task 3 across combinations on unmasked holdout data.
"""
import glob, os, time
from pathlib import Path
import numpy as np, pandas as pd

DATA_DIR = Path("kaggle_public")

PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S",
    "D12_I405_N", "D12_I405_S"
]

FAMILIES = {
    "D7_I10": ["D7_I10_E", "D7_I10_W"],
    "D7_I210": ["D7_I210_E", "D7_I210_W"],
    "D7_I405": ["D7_I405_N", "D7_I405_S"],
    "D12_I5": ["D12_I5_N", "D12_I5_S"],
    "D12_I405": ["D12_I405_N", "D12_I405_S"]
}

# Let's test on D7_I10_E and D7_I210_E
for test_panel in ['D7_I10_E', 'D7_I210_E']:
    panel_dir = DATA_DIR / 'corridors' / test_panel
    fd = pd.read_csv(panel_dir / 'network' / 'fd_parameters.csv')
    link_lanes = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    link_fs = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
    link_cap = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
    
    # Load 12 holdout days from train (4 from each regime R1, R2, R3)
    val_files = []
    for r in ['R1', 'R2', 'R3']:
        r_f = sorted(glob.glob(str(panel_dir / f'train/mainline_states_masked/mask_regime={r}/*.parquet')))[-4:]
        val_files.extend(r_f)
        
    truth_dir = panel_dir / 'train/mainline_states'
    
    # 1. Old Model (8 days training)
    from scripts.build_task1_lightgbm import train_corridor_models as new_train, extract_features_from_day
    
    # Load ground truth
    all_truth = []
    all_masked = []
    for mf in val_files:
        fn = os.path.basename(mf)
        tf = glob.glob(f"{truth_dir}/**/{fn}", recursive=True)[0]
        all_truth.append(pd.read_parquet(tf))
        all_masked.append(pd.read_parquet(mf))
        
    truth_df = pd.concat(all_truth, ignore_index=True)
    
    # Train Old vs New models
    # Old model
    print(f"\nEvaluating Panel: {test_panel} on {len(val_files)} holdout days...")
    
    # Train New model
    model_s_new, model_f_new, feats_new, meta_new, prof_new = new_train(test_panel, n_train_days=36)
    
    # Predict New Model
    preds_new_list = []
    for df_m in all_masked:
        st_mp = df_m[['station_id', 'milepost']].drop_duplicates().sort_values('milepost')
        ordered_stations = st_mp['station_id'].tolist()
        f_df = extract_features_from_day(df_m, ordered_stations, meta_new, prof_new)
        mask_need = (df_m['is_missing'] == 1) | (df_m['speed_kmh'].isna()) | (df_m['flow_vph'].isna())
        target_sub = f_df[mask_need].copy()
        if not target_sub.empty:
            X_val = target_sub[feats_new]
            target_sub['speed_kmh'] = np.clip(model_s_new.predict(X_val), 5.0, target_sub['free_speed'].values * 1.05)
            target_sub['flow_vph'] = np.clip(model_f_new.predict(X_val) * target_sub['lanes'].values, 50.0, target_sub['capacity'].values * 1.02)
            preds_new_list.append(target_sub[['timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']])
            
    pred_new_raw = pd.concat(preds_new_list, ignore_index=True)
    
    # Score New Raw (without physics nudge)
    from scripts.evaluate_local_metrics import score_state_reconstruction
    s_new_raw = score_state_reconstruction(pred_new_raw, truth_df, fd)
    print(f"  New Pipeline S_state (raw): {s_new_raw['S_state']:.5f}")
    
    # Apply Old Physics Nudge: v = 0.7*v + 0.3*v_free, q = v*k
    pred_with_old_nudge = pred_new_raw.copy()
    pred_with_old_nudge['lanes'] = pred_with_old_nudge['link_id'].map(link_lanes).fillna(3.0)
    pred_with_old_nudge['free_speed'] = pred_with_old_nudge['link_id'].map(link_fs).fillna(105.0)
    pred_with_old_nudge['capacity'] = pred_with_old_nudge['link_id'].map(link_cap).fillna(5700.0)
    
    v_nudge = np.clip(0.7 * pred_with_old_nudge['speed_kmh'].values + 0.3 * pred_with_old_nudge['free_speed'].values, 15.0, pred_with_old_nudge['free_speed'].values)
    k_nudge = pred_with_old_nudge['flow_vph'].values / np.maximum(pred_with_old_nudge['speed_kmh'].values, 1.0)
    q_nudge = np.clip(v_nudge * k_nudge, 55.0, pred_with_old_nudge['capacity'].values)
    
    pred_with_old_nudge['speed_kmh'] = v_nudge
    pred_with_old_nudge['flow_vph'] = q_nudge
    
    s_with_nudge = score_state_reconstruction(pred_with_old_nudge, truth_df, fd)
    print(f"  New Pipeline S_state (with 30% speed nudge): {s_with_nudge['S_state']:.5f}")
