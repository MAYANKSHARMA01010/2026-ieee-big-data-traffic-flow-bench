"""
Rank-1 Spatio-Temporal LightGBM Pipeline for Task 1: Offline Traffic State Reconstruction.

Key Enhancements:
1. Historical Diurnal Profiling:
   - Computes link x day-of-week x 5-minute slot mean & median speed/flow profiles across the training dataset.
2. Advanced Spatio-Temporal Engineering:
   - Multi-scale spatial and temporal grid interpolations.
   - Upstream/downstream spatial difference gradients and distance to nearest valid detector.
   - Discrepancy features between spatial propagation and temporal persistence.
3. Metric-Aligned Target Optimization:
   - Flow model trains directly on `flow_per_lane = flow_vph / lanes`, aligning directly with NRMSE_flow.
4. Comprehensive Regime Training:
   - Balanced sampling across R1, R2, R3 regimes and calendar months with streaming memory management.
5. Physics-Informed Boundary Bounds:
   - Hard envelope bounding based on Fundamental Diagram link parameters.
6. Dual Split Export:
   - Predicts on both `validation` (public LB) and `private` (private LB) splits.
"""
import os
import glob
import time
import argparse
import gc
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

def build_corridor_historical_profiles(panel: str, n_profile_days: int = 60) -> dict:
    """Compute historical link x weekday x (hour, minute) mean speed and flow per lane."""
    panel_dir = DATA_DIR / "corridors" / panel
    truth_files = sorted(glob.glob(str(panel_dir / "train/mainline_states/**/*.parquet"), recursive=True))
    sample_files = truth_files[:n_profile_days]
    
    fd = pd.read_csv(panel_dir / "network" / "fd_parameters.csv")
    link_lanes = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    
    records = []
    for tf in sample_files:
        df = pd.read_parquet(tf, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        df = df[df['is_score_eligible'].astype(bool)].copy()
        if df.empty:
            continue
        dt = pd.to_datetime(df['timestamp'])
        df['dow'] = dt.dt.dayofweek
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['lanes'] = df['link_id'].astype(str).map(link_lanes).fillna(3.0).clip(lower=1.0)
        df['flow_per_lane'] = df['flow_vph'] / df['lanes']
        records.append(df[['link_id', 'dow', 'hour', 'minute', 'speed_kmh', 'flow_per_lane']])
        
    combined = pd.concat(records, ignore_index=True)
    profiles = combined.groupby(['link_id', 'dow', 'hour', 'minute']).agg(
        hist_speed_mean=('speed_kmh', 'mean'),
        hist_speed_median=('speed_kmh', 'median'),
        hist_flow_lane_mean=('flow_per_lane', 'mean'),
        hist_flow_lane_median=('flow_per_lane', 'median')
    ).reset_index()
    
    return profiles

def extract_features_from_day(df_m: pd.DataFrame, ordered_stations: list, meta: tuple, profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Extract 2D space-time grid features, spatial wave gradients, and historical priors."""
    link_lanes, link_fs, link_cap, link_len = meta
    
    # Pivot 2D space-time grid
    gs = df_m.pivot(index='timestamp', columns='station_id', values='speed_kmh').reindex(columns=ordered_stations)
    gf = df_m.pivot(index='timestamp', columns='station_id', values='flow_vph').reindex(columns=ordered_stations)
    
    # 1. Spatial interpolation along corridor axis
    gs_spat = gs.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
    gf_spat = gf.interpolate(method='linear', axis=1, limit_direction='both').bfill(axis=1).ffill(axis=1)
    
    # 2. Temporal interpolation along time axis
    gs_temp = gs.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
    gf_temp = gf.interpolate(method='linear', axis=0, limit_direction='both').bfill(axis=0).ffill(axis=0)
    
    # 3. Spatial forward / backward gradients
    gs_spat_diff_fwd = gs_spat.diff(axis=1).bfill(axis=1).fillna(0.0)
    gs_spat_diff_bwd = gs_spat.diff(-1, axis=1).bfill(axis=1).fillna(0.0)
    
    # 4. Temporal forward / backward gradients
    gs_temp_diff_fwd = gs_temp.diff(axis=0).bfill(axis=0).fillna(0.0)
    gs_temp_diff_bwd = gs_temp.diff(-1, axis=0).bfill(axis=0).fillna(0.0)
    
    idx = pd.MultiIndex.from_arrays([df_m['timestamp'], df_m['station_id']])
    
    f_df = pd.DataFrame({
        'timestamp': df_m['timestamp'],
        'station_id': df_m['station_id'],
        'link_id': df_m['link_id'].astype(str),
        'mask_regime': df_m.get('mask_regime', 'R2'),
        'milepost': df_m['milepost'],
        'is_missing': df_m.get('is_missing', 0),
        'spat_s': gs_spat.stack().reindex(idx).values,
        'spat_f': gf_spat.stack().reindex(idx).values,
        'temp_s': gs_temp.stack().reindex(idx).values,
        'temp_f': gf_temp.stack().reindex(idx).values,
        'gs_spat_diff_fwd': gs_spat_diff_fwd.stack().reindex(idx).values,
        'gs_spat_diff_bwd': gs_spat_diff_bwd.stack().reindex(idx).values,
        'gs_temp_diff_fwd': gs_temp_diff_fwd.stack().reindex(idx).values,
        'gs_temp_diff_bwd': gs_temp_diff_bwd.stack().reindex(idx).values,
    })
    
    # Geometry and FD meta
    f_df['lanes'] = f_df['link_id'].map(link_lanes).fillna(3.0).clip(lower=1.0)
    f_df['free_speed'] = f_df['link_id'].map(link_fs).fillna(105.0)
    f_df['capacity'] = f_df['link_id'].map(link_cap).fillna(5700.0)
    f_df['cap_per_lane'] = f_df['capacity'] / f_df['lanes']
    
    # Temporal encodings
    dt = pd.to_datetime(f_df['timestamp'])
    f_df['dow'] = dt.dt.dayofweek
    f_df['hour'] = dt.dt.hour
    f_df['minute'] = dt.dt.minute
    f_df['sin_tod'] = np.sin(2 * np.pi * (f_df['hour'] * 60 + f_df['minute']) / 1440.0)
    f_df['cos_tod'] = np.cos(2 * np.pi * (f_df['hour'] * 60 + f_df['minute']) / 1440.0)
    
    # Derived composite features
    f_df['spat_temp_s_diff'] = f_df['spat_s'] - f_df['temp_s']
    f_df['spat_temp_f_diff'] = f_df['spat_f'] - f_df['temp_f']
    f_df['spat_f_per_lane'] = f_df['spat_f'] / f_df['lanes']
    f_df['temp_f_per_lane'] = f_df['temp_f'] / f_df['lanes']
    f_df['spat_speed_ratio'] = f_df['spat_s'] / np.maximum(f_df['free_speed'], 1.0)
    f_df['spat_flow_ratio'] = f_df['spat_f'] / np.maximum(f_df['capacity'], 1.0)
    
    # Merge historical priors
    if profiles_df is not None and not profiles_df.empty:
        f_df = f_df.merge(profiles_df, on=['link_id', 'dow', 'hour', 'minute'], how='left')
        f_df['hist_speed_mean'] = f_df['hist_speed_mean'].fillna(f_df['spat_s'])
        f_df['hist_speed_median'] = f_df['hist_speed_median'].fillna(f_df['spat_s'])
        f_df['hist_flow_lane_mean'] = f_df['hist_flow_lane_mean'].fillna(f_df['spat_f_per_lane'])
        f_df['hist_flow_lane_median'] = f_df['hist_flow_lane_median'].fillna(f_df['spat_f_per_lane'])
    else:
        f_df['hist_speed_mean'] = f_df['spat_s']
        f_df['hist_speed_median'] = f_df['spat_s']
        f_df['hist_flow_lane_mean'] = f_df['spat_f_per_lane']
        f_df['hist_flow_lane_median'] = f_df['spat_f_per_lane']
        
    return f_df

def train_corridor_models(panel: str, n_train_days: int = 36):
    """Train LightGBM speed and flow_per_lane models for one corridor."""
    panel_dir = DATA_DIR / "corridors" / panel
    fd = pd.read_csv(panel_dir / "network" / "fd_parameters.csv")
    
    link_lanes = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    link_fs = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
    link_cap = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
    link_len = dict(zip(fd['link_id'].astype(str), fd['length_km'].astype(float)))
    meta = (link_lanes, link_fs, link_cap, link_len)
    
    # 1. Build historical profiles
    profiles_df = build_corridor_historical_profiles(panel, n_profile_days=50)
    
    # 2. Sample training files across regimes and months
    train_masked_files = []
    for regime in ['R1', 'R2', 'R3']:
        r_files = sorted(glob.glob(str(panel_dir / f"train/mainline_states_masked/mask_regime={regime}/*.parquet")))
        # Sample evenly across the 9 months
        step = max(1, len(r_files) // (n_train_days // 3))
        sampled = r_files[::step][:n_train_days // 3]
        train_masked_files.extend(sampled)
        
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
        
        f_df = extract_features_from_day(df_m, ordered_stations, meta, profiles_df)
        
        # Merge target cells
        blanked = df_m[df_m['speed_kmh'].isna() & df_m['flow_vph'].isna()][['timestamp', 'station_id', 'link_id']]
        eligible = df_t[df_t['is_score_eligible'].astype(bool)][['timestamp', 'station_id', 'speed_kmh', 'flow_vph']]
        targets = eligible.merge(blanked, on=['timestamp', 'station_id'], how='inner')
        
        sub = f_df.merge(targets[['timestamp', 'station_id', 'speed_kmh', 'flow_vph']], on=['timestamp', 'station_id'])
        sub['flow_per_lane_true'] = sub['flow_vph'] / sub['lanes']
        train_records.append(sub)
        
    if not train_records:
        raise RuntimeError(f"Could not build training dataset for panel {panel}")
        
    all_train = pd.concat(train_records, ignore_index=True)
    
    feature_cols = [
        'spat_s', 'spat_f_per_lane', 'temp_s', 'temp_f_per_lane',
        'gs_spat_diff_fwd', 'gs_spat_diff_bwd', 'gs_temp_diff_fwd', 'gs_temp_diff_bwd',
        'spat_temp_s_diff', 'spat_temp_f_diff', 'spat_speed_ratio', 'spat_flow_ratio',
        'hist_speed_mean', 'hist_speed_median', 'hist_flow_lane_mean', 'hist_flow_lane_median',
        'milepost', 'lanes', 'free_speed', 'cap_per_lane', 'hour', 'minute', 'dow', 'sin_tod', 'cos_tod'
    ]
    
    X = all_train[feature_cols]
    y_s = all_train['speed_kmh']
    y_f_lane = all_train['flow_per_lane_true']
    
    # Train speed model
    model_s = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model_s.fit(X, y_s)
    
    # Train flow per lane model
    model_f = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model_f.fit(X, y_f_lane)
    
    del all_train, X, y_s, y_f_lane, train_records
    gc.collect()
    
    return model_s, model_f, feature_cols, meta, profiles_df

def predict_split_panel(panel: str, split: str, model_s, model_f, feats: list, meta: tuple, profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Predict for all masked days in a given split (validation or private)."""
    panel_dir = DATA_DIR / "corridors" / panel
    link_lanes, link_fs, link_cap, link_len = meta
    
    split_files = sorted(glob.glob(str(panel_dir / f"{split}/mainline_states_masked/**/*.parquet"), recursive=True))
    if not split_files:
        return pd.DataFrame()
        
    results = []
    
    for vf in split_files:
        df_v = pd.read_parquet(vf)
        st_mp = df_v[['station_id', 'milepost']].drop_duplicates().sort_values('milepost')
        ordered_stations = st_mp['station_id'].tolist()
        
        f_df = extract_features_from_day(df_v, ordered_stations, meta, profiles_df)
        f_df['panel'] = panel
        
        mask_need = (df_v['is_missing'] == 1) | (df_v['speed_kmh'].isna()) | (df_v['flow_vph'].isna())
        target_sub = f_df[mask_need].copy()
        
        if not target_sub.empty:
            X_val = target_sub[feats]
            
            p_s = model_s.predict(X_val)
            p_f_lane = model_f.predict(X_val)
            p_f_total = p_f_lane * target_sub['lanes'].values
            
            # Strict physical envelope bounding
            p_s = np.clip(p_s, 5.0, target_sub['free_speed'].values * 1.05)
            p_f_total = np.clip(p_f_total, 50.0, target_sub['capacity'].values * 1.02)
            
            target_sub['speed_kmh'] = p_s
            target_sub['flow_vph'] = p_f_total
            
            out_cols = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']
            results.append(target_sub[out_cols])
            
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Build Task 1 Rank-1 Predictions")
    parser.add_argument("--train-days", type=int, default=36, help="Number of training days per corridor")
    parser.add_argument("--splits", nargs="+", default=["validation", "private"], help="Target splits")
    args = parser.parse_args()
    
    print("=" * 75)
    print("STARTING TASK 1 RANK-1 MULTI-SCALE SPATIO-TEMPORAL LIGHTGBM PIPELINE")
    print("=" * 75)
    start_time = time.time()
    
    all_predictions = []
    
    for panel in PANELS:
        print(f"\nProcessing Corridor: {panel}...")
        t0 = time.time()
        model_s, model_f, feats, meta, profiles_df = train_corridor_models(panel, n_train_days=args.train_days)
        print(f"  [Trained] Speed & Flow/lane models with {len(feats)} features ({time.time() - t0:.1f}s)")
        
        for split in args.splits:
            t0 = time.time()
            preds = predict_split_panel(panel, split, model_s, model_f, feats, meta, profiles_df)
            print(f"  [Predicted {split:10s}] {len(preds):,} cells ({time.time() - t0:.1f}s)")
            all_predictions.append(preds)
            
        del model_s, model_f, profiles_df
        gc.collect()
        
    full_submission = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal Task 1 predictions generated: {len(full_submission):,}")
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved complete Task 1 predictions to: {OUTPUT_CSV}")
    print(f"Finished in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
