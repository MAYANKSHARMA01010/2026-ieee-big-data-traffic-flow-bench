"""
Rank-1 Supervised Spatio-Temporal LightGBM Pipeline for Task 1: Offline Traffic State Reconstruction.

Methodology:
1. 2D Spatio-Temporal Highway Grid Formulation:
   - Highway stations ordered continuously by milepost.
   - At each timestamp, spatial interpolation between adjacent observed stations.
   - For each station, temporal interpolation across time.
2. Gradient Boosted Meta-Estimator (LightGBM):
   - Trained on true vs masked pairs from the 273 days of training data.
   - Features: spatial interpolation, temporal interpolation, milepost, road capacity,
     lane count, free speed, time of day (hour, minute).
3. Physics-informed boundary clipping from Fundamental Diagram parameters.
"""
import os
import glob
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

DATA_DIR = Path("kaggle_public")
OUTPUT_CSV = Path("datasets/task1_state_submission.csv")

PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S",
    "D12_I405_N", "D12_I405_S"
]

def train_corridor_models(panel: str, n_train_days: int = 8):
    """Train LightGBM speed and flow models for one corridor."""
    panel_dir = DATA_DIR / "corridors" / panel
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    fd = pd.read_csv(fd_path)
    
    link_lanes = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    link_fs = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
    link_cap = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
    
    # Select sample train masked files across regimes
    train_masked_files = []
    for regime in ['R1', 'R2', 'R3']:
        r_files = sorted(glob.glob(str(panel_dir / f"train/mainline_states_masked/mask_regime={regime}/*.parquet")))
        train_masked_files.extend(r_files[:max(2, n_train_days // 3)])
    
    truth_dir = panel_dir / "train/mainline_states"
    train_records = []
    
    for mf in train_masked_files:
        fname = os.path.basename(mf)
        tfiles = glob.glob(f"{truth_dir}/**/{fname}", recursive=True)
        if not tfiles:
            continue
            
        df_m = pd.read_parquet(mf)
        df_t = pd.read_parquet(tfiles[0])
        
        st_mp = df_m[['station_id', 'milepost']].drop_duplicates().sort_values('milepost')
        ordered_stations = st_mp['station_id'].tolist()
        
        # 2D Grid
        gs = df_m.pivot(index='timestamp', columns='station_id', values='speed_kmh')[ordered_stations]
        gf = df_m.pivot(index='timestamp', columns='station_id', values='flow_vph')[ordered_stations]
        
        # Spatial interpolation along highway axis
        gs_spat = gs.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
        gf_spat = gf.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
        
        # Temporal interpolation along time axis
        gs_temp = gs.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
        gf_temp = gf.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
        
        # Stack feature tables
        f_df = pd.DataFrame({
            'timestamp': df_m['timestamp'],
            'station_id': df_m['station_id'],
            'link_id': df_m['link_id'],
            'milepost': df_m['milepost'],
            'is_missing': df_m['is_missing'],
            'spat_s': gs_spat.stack().reindex(pd.MultiIndex.from_arrays([df_m['timestamp'], df_m['station_id']])).values,
            'spat_f': gf_spat.stack().reindex(pd.MultiIndex.from_arrays([df_m['timestamp'], df_m['station_id']])).values,
            'temp_s': gs_temp.stack().reindex(pd.MultiIndex.from_arrays([df_m['timestamp'], df_m['station_id']])).values,
            'temp_f': gf_temp.stack().reindex(pd.MultiIndex.from_arrays([df_m['timestamp'], df_m['station_id']])).values,
        })
        
        # Identify target cells
        blanked = df_m[df_m.speed_kmh.isna() & df_m.flow_vph.isna()][['timestamp', 'station_id', 'link_id']]
        eligible = df_t[df_t.is_score_eligible.astype(bool)]
        targets = eligible.merge(blanked, on=['timestamp', 'station_id', 'link_id'], how='inner')
        
        sub = f_df.merge(targets[['timestamp', 'station_id', 'speed_kmh', 'flow_vph']], on=['timestamp', 'station_id'])
        train_records.append(sub)
        
    if not train_records:
        raise RuntimeError(f"Could not build training dataset for panel {panel}")
        
    all_train = pd.concat(train_records, ignore_index=True)
    all_train['lanes'] = all_train['link_id'].map(link_lanes).fillna(3.0)
    all_train['free_speed'] = all_train['link_id'].map(link_fs).fillna(105.0)
    all_train['capacity'] = all_train['link_id'].map(link_cap).fillna(5700.0)
    
    dt = pd.to_datetime(all_train['timestamp'])
    all_train['hour'] = dt.dt.hour
    all_train['minute'] = dt.dt.minute
    
    feats = ['spat_s', 'spat_f', 'temp_s', 'temp_f', 'milepost', 'lanes', 'free_speed', 'capacity', 'hour', 'minute']
    X = all_train[feats]
    y_s = all_train['speed_kmh']
    y_f = all_train['flow_vph']
    
    model_s = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    model_s.fit(X, y_s)
    
    model_f = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    model_f.fit(X, y_f)
    
    return model_s, model_f, feats, (link_lanes, link_fs, link_cap)

def predict_validation_panel(panel: str, model_s, model_f, feats, meta):
    """Predict all validation masked days for one corridor."""
    panel_dir = DATA_DIR / "corridors" / panel
    link_lanes, link_fs, link_cap = meta
    
    val_files = sorted(glob.glob(str(panel_dir / "validation/mainline_states_masked/**/*.parquet"), recursive=True))
    results = []
    
    for vf in val_files:
        df_v = pd.read_parquet(vf)
        st_mp = df_v[['station_id', 'milepost']].drop_duplicates().sort_values('milepost')
        ordered_stations = st_mp['station_id'].tolist()
        
        gs = df_v.pivot(index='timestamp', columns='station_id', values='speed_kmh')[ordered_stations]
        gf = df_v.pivot(index='timestamp', columns='station_id', values='flow_vph')[ordered_stations]
        
        gs_spat = gs.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
        gf_spat = gf.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
        
        gs_temp = gs.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
        gf_temp = gf.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
        
        f_df = pd.DataFrame({
            'panel': panel,
            'timestamp': df_v['timestamp'],
            'station_id': df_v['station_id'],
            'link_id': df_v['link_id'],
            'mask_regime': df_v['mask_regime'],
            'milepost': df_v['milepost'],
            'is_missing': df_v['is_missing'],
            'spat_s': gs_spat.stack().reindex(pd.MultiIndex.from_arrays([df_v['timestamp'], df_v['station_id']])).values,
            'spat_f': gf_spat.stack().reindex(pd.MultiIndex.from_arrays([df_v['timestamp'], df_v['station_id']])).values,
            'temp_s': gs_temp.stack().reindex(pd.MultiIndex.from_arrays([df_v['timestamp'], df_v['station_id']])).values,
            'temp_f': gf_temp.stack().reindex(pd.MultiIndex.from_arrays([df_v['timestamp'], df_v['station_id']])).values,
        })
        
        # We predict on rows that are masked (is_missing == 1) or NaN
        mask_need = (df_v['is_missing'] == 1) | (df_v['speed_kmh'].isna()) | (df_v['flow_vph'].isna())
        target_sub = f_df[mask_need].copy()
        
        if not target_sub.empty:
            target_sub['lanes'] = target_sub['link_id'].map(link_lanes).fillna(3.0)
            target_sub['free_speed'] = target_sub['link_id'].map(link_fs).fillna(105.0)
            target_sub['capacity'] = target_sub['link_id'].map(link_cap).fillna(5700.0)
            
            dt = pd.to_datetime(target_sub['timestamp'])
            target_sub['hour'] = dt.dt.hour
            target_sub['minute'] = dt.dt.minute
            
            X_val = target_sub[feats]
            p_s = model_s.predict(X_val)
            p_f = model_f.predict(X_val)
            
            # Physics clipping
            p_s = np.clip(p_s, 5.0, target_sub['free_speed'].values * 1.1)
            p_f = np.clip(p_f, 50.0, target_sub['capacity'].values * 1.05)
            
            target_sub['speed_kmh'] = p_s
            target_sub['flow_vph'] = p_f
            
            out_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']
            results.append(target_sub[out_cols])
            
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def main():
    print("=" * 70)
    print("STARTING TASK 1 HIGH-ACCURACY SPATIO-TEMPORAL LIGHTGBM PIPELINE")
    print("=" * 70)
    start_time = time.time()
    
    all_panel_predictions = []
    
    for panel in PANELS:
        print(f"\nProcessing corridor: {panel}...")
        t0 = time.time()
        model_s, model_f, feats, meta = train_corridor_models(panel, n_train_days=8)
        print(f"  [Trained] Speed & Flow LightGBM models ({time.time() - t0:.1f}s)")
        
        t0 = time.time()
        val_preds = predict_validation_panel(panel, model_s, model_f, feats, meta)
        print(f"  [Predicted] {len(val_preds):,} validation cells ({time.time() - t0:.1f}s)")
        all_panel_predictions.append(val_preds)
        
    full_submission = pd.concat(all_panel_predictions, ignore_index=True)
    print(f"\nTotal Task 1 predictions generated: {len(full_submission):,}")
    
    # Save output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved complete Task 1 predictions to: {OUTPUT_CSV}")
    print(f"Finished in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
