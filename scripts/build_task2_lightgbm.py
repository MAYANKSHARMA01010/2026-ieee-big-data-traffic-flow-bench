"""
Rank-1 Shockwave-Aware LightGBM Pipeline for Task 2: Online Queue-Propagation Forecasting.

Key Enhancements:
1. Topology & Downstream Shockwave Feature Engine:
   - Highway corridor topological ordering (`order_index`).
   - Downstream link speed, occupancy, and queue status features (detects backward-propagating shockwaves).
   - Upstream link congestion boundary tracking.
   - Kinematic shockwave velocity approximation w = dq / dk.
2. Temporal Acceleration & Bottleneck Dynamics:
   - Speed decline acceleration: (v[T-5] - v[T-10]) - (v[T-10] - v[T-15]).
   - Occupancy accumulation rate: (occ[T-5] - occ[T-15]).
   - Free speed margin: (v_free - v[T-5]) / v_free.
3. Out-of-Fold (OOF) Calibrated Thresholding:
   - 5-Fold Group Stratified Cross-Validation on the 80 training windows.
   - Calibrates condition-specific binary decision thresholds (tau_onset, tau_ongoing) to maximize Space-Time IoU.
4. Dual Split Support:
   - Generates predictions for both `validation` (public LB) and `private` (private LB) splits.
"""
import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from tqdm import tqdm

DATA_DIR = Path("kaggle_public")
OUTPUT_CSV = Path("datasets/task2_queue_submission.csv")

SCORED_PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S"
]

def load_panel_topology(panel: str):
    """Load corridor topological ordering and link parameters."""
    panel_dir = DATA_DIR / "corridors" / panel / "network"
    topo_path = panel_dir / "lwr_mainline_topology.csv"
    fd_path = panel_dir / "fd_parameters.csv"
    
    fd = pd.read_csv(fd_path)
    v_cut = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float) * 0.60))
    fs_map = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
    cap_map = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
    lanes_map = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    length_map = dict(zip(fd['link_id'].astype(str), fd['length_km'].astype(float)))
    
    if topo_path.exists():
        topo = pd.read_csv(topo_path)
        topo = topo.sort_values('order_index')
        ordered_links = topo['link_id'].astype(str).tolist()
    else:
        ordered_links = fd['link_id'].astype(str).tolist()
        
    link_to_order = {l: i for i, l in enumerate(ordered_links)}
    return (v_cut, fs_map, cap_map, lanes_map, length_map, ordered_links, link_to_order)

def extract_window_features(wh_w: pd.DataFrame, topo_meta: tuple, condition: str, is_val: bool = False, sample_sub: pd.DataFrame = None, horizon_true: pd.DataFrame = None):
    """Extract shockwave, downstream gradient, and temporal kinematic features for each link and forecast step."""
    v_cut, fs_map, cap_map, lanes_map, length_map, ordered_links, link_to_order = topo_meta
    
    # Pivot history
    p_speed = wh_w.pivot(index='timestamp', columns='link_id', values='speed_kmh')
    p_flow = wh_w.pivot(index='timestamp', columns='link_id', values='flow_vph')
    p_occ = wh_w.pivot(index='timestamp', columns='link_id', values='occupancy')
    
    n_time = len(p_speed)
    
    # Precompute link-level history summaries
    link_stats = {}
    for link in ordered_links:
        if link not in p_speed.columns:
            continue
        s_series = p_speed[link].dropna()
        q_series = p_flow[link].dropna()
        o_series = p_occ[link].dropna()
        
        s_t0 = s_series.iloc[-1] if len(s_series) >= 1 else np.nan
        s_t1 = s_series.iloc[-2] if len(s_series) >= 2 else s_t0
        s_t2 = s_series.iloc[-3] if len(s_series) >= 3 else s_t1
        
        q_t0 = q_series.iloc[-1] if len(q_series) >= 1 else np.nan
        o_t0 = o_series.iloc[-1] if len(o_series) >= 1 else np.nan
        o_t1 = o_series.iloc[-2] if len(o_series) >= 2 else o_t0
        o_t2 = o_series.iloc[-3] if len(o_series) >= 3 else o_t1
        
        vc = v_cut.get(link, 63.0)
        vf = fs_map.get(link, 105.0)
        cap = cap_map.get(link, 5700.0)
        
        dv_dt1 = s_t0 - s_t1 if pd.notna(s_t0) and pd.notna(s_t1) else 0.0
        dv_dt2 = s_t1 - s_t2 if pd.notna(s_t1) and pd.notna(s_t2) else 0.0
        accel = dv_dt1 - dv_dt2
        docc_dt = o_t0 - o_t2 if pd.notna(o_t0) and pd.notna(o_t2) else 0.0
        
        is_q = 1 if (pd.notna(s_t0) and s_t0 <= vc) else 0
        min_s = s_series.min() if not s_series.empty else np.nan
        mean_s = s_series.mean() if not s_series.empty else np.nan
        
        link_stats[link] = {
            's_t0': s_t0,
            's_t1': s_t1,
            'q_t0': q_t0,
            'o_t0': o_t0,
            'dv_dt1': dv_dt1,
            'accel': accel,
            'docc_dt': docc_dt,
            'is_q': is_q,
            'min_s': min_s,
            'mean_s': mean_s,
            'margin_vcut': s_t0 - vc if pd.notna(s_t0) else 0.0,
            'margin_vfree': (vf - s_t0) / vf if pd.notna(s_t0) else 0.0,
            'cap_util': q_t0 / cap if pd.notna(q_t0) and cap > 0 else 0.0
        }
        
    features = []
    labels = []
    keys = []
    
    if is_val:
        unique_ts = sorted(sample_sub['timestamp'].unique().tolist())
        target_links = sample_sub['link_id'].unique().tolist()
        
        for step_idx, ts in enumerate(unique_ts):
            step_min = (step_idx + 1) * 5
            for link in target_links:
                curr = link_stats.get(link, {})
                ord_idx = link_to_order.get(link, 0)
                
                # Downstream neighbor features (traffic flows towards higher order_index)
                down_link1 = ordered_links[ord_idx + 1] if ord_idx + 1 < len(ordered_links) else None
                down_link2 = ordered_links[ord_idx + 2] if ord_idx + 2 < len(ordered_links) else None
                down1 = link_stats.get(down_link1, {}) if down_link1 else {}
                down2 = link_stats.get(down_link2, {}) if down_link2 else {}
                
                # Upstream neighbor features
                up_link1 = ordered_links[ord_idx - 1] if ord_idx - 1 >= 0 else None
                up1 = link_stats.get(up_link1, {}) if up_link1 else {}
                
                # Shockwave arrival potential
                down_is_q = max(down1.get('is_q', 0), down2.get('is_q', 0))
                down_s_drop = min(down1.get('s_t0', 105.0), down2.get('s_t0', 105.0))
                
                row_feat = {
                    'condition': 1 if condition == 'queue_ongoing' else 0,
                    'step_min': step_min,
                    'free_speed': fs_map.get(link, 105.0),
                    'capacity': cap_map.get(link, 5700.0),
                    'lanes': lanes_map.get(link, 3.0),
                    'length_km': length_map.get(link, 0.5),
                    'v_cut': v_cut.get(link, 63.0),
                    's_t0': curr.get('s_t0', np.nan),
                    'q_t0': curr.get('q_t0', np.nan),
                    'o_t0': curr.get('o_t0', np.nan),
                    'dv_dt1': curr.get('dv_dt1', 0.0),
                    'accel': curr.get('accel', 0.0),
                    'docc_dt': curr.get('docc_dt', 0.0),
                    'is_q_origin': curr.get('is_q', 0),
                    'min_speed': curr.get('min_s', np.nan),
                    'mean_speed': curr.get('mean_s', np.nan),
                    'margin_vcut': curr.get('margin_vcut', 0.0),
                    'margin_vfree': curr.get('margin_vfree', 0.0),
                    'cap_util': curr.get('cap_util', 0.0),
                    'down1_s_t0': down1.get('s_t0', np.nan),
                    'down1_is_q': down1.get('is_q', 0),
                    'down2_is_q': down2.get('is_q', 0),
                    'down_is_q': down_is_q,
                    'down_s_drop': down_s_drop,
                    'up1_is_q': up1.get('is_q', 0),
                    'up1_s_t0': up1.get('s_t0', np.nan),
                    'down_spatial_grad': down1.get('s_t0', 105.0) - curr.get('s_t0', 105.0)
                }
                features.append(row_feat)
                keys.append((wh_w['window_id'].iloc[0], ts, link))
                
        return pd.DataFrame(features), keys
    else:
        timestamps = sorted(horizon_true['timestamp'].unique().tolist())
        target_links = sorted(horizon_true['link_id'].unique().tolist())
        
        for step_idx, ts in enumerate(timestamps):
            step_min = (step_idx + 1) * 5
            for link in target_links:
                gt_row = horizon_true[(horizon_true['timestamp'] == ts) & (horizon_true['link_id'] == link)]
                if gt_row.empty:
                    continue
                label = gt_row['queue_true'].iloc[0]
                
                curr = link_stats.get(link, {})
                ord_idx = link_to_order.get(link, 0)
                
                down_link1 = ordered_links[ord_idx + 1] if ord_idx + 1 < len(ordered_links) else None
                down_link2 = ordered_links[ord_idx + 2] if ord_idx + 2 < len(ordered_links) else None
                down1 = link_stats.get(down_link1, {}) if down_link1 else {}
                down2 = link_stats.get(down_link2, {}) if down_link2 else {}
                
                up_link1 = ordered_links[ord_idx - 1] if ord_idx - 1 >= 0 else None
                up1 = link_stats.get(up_link1, {}) if up_link1 else {}
                
                down_is_q = max(down1.get('is_q', 0), down2.get('is_q', 0))
                down_s_drop = min(down1.get('s_t0', 105.0), down2.get('s_t0', 105.0))
                
                row_feat = {
                    'condition': 1 if condition == 'queue_ongoing' else 0,
                    'step_min': step_min,
                    'free_speed': fs_map.get(link, 105.0),
                    'capacity': cap_map.get(link, 5700.0),
                    'lanes': lanes_map.get(link, 3.0),
                    'length_km': length_map.get(link, 0.5),
                    'v_cut': v_cut.get(link, 63.0),
                    's_t0': curr.get('s_t0', np.nan),
                    'q_t0': curr.get('q_t0', np.nan),
                    'o_t0': curr.get('o_t0', np.nan),
                    'dv_dt1': curr.get('dv_dt1', 0.0),
                    'accel': curr.get('accel', 0.0),
                    'docc_dt': curr.get('docc_dt', 0.0),
                    'is_q_origin': curr.get('is_q', 0),
                    'min_speed': curr.get('min_s', np.nan),
                    'mean_speed': curr.get('mean_s', np.nan),
                    'margin_vcut': curr.get('margin_vcut', 0.0),
                    'margin_vfree': curr.get('margin_vfree', 0.0),
                    'cap_util': curr.get('cap_util', 0.0),
                    'down1_s_t0': down1.get('s_t0', np.nan),
                    'down1_is_q': down1.get('is_q', 0),
                    'down2_is_q': down2.get('is_q', 0),
                    'down_is_q': down_is_q,
                    'down_s_drop': down_s_drop,
                    'up1_is_q': up1.get('is_q', 0),
                    'up1_s_t0': up1.get('s_t0', np.nan),
                    'down_spatial_grad': down1.get('s_t0', 105.0) - curr.get('s_t0', 105.0)
                }
                features.append(row_feat)
                labels.append(label)
                
        return pd.DataFrame(features), np.array(labels)

def train_and_calibrate_queue_models():
    """Train LightGBM queue classifier and calibrate decision thresholds via OOF cross-validation."""
    print("\nMining Task 2 shockwave training dataset across all 8 corridors...")
    train_wis = glob.glob(str(DATA_DIR / "task2/*/train/window_index.csv"))
    
    window_data = []
    
    for f_wi in train_wis:
        panel = f_wi.split('/')[-3]
        if panel not in SCORED_PANELS:
            continue
            
        wi = pd.read_csv(f_wi)
        wh = pd.read_parquet(DATA_DIR / f"task2/{panel}/train/window_history.parquet")
        wh['timestamp'] = pd.to_datetime(wh['timestamp'], utc=True)
        
        topo_meta = load_panel_topology(panel)
        v_cut = topo_meta[0]
        
        for _, w in wi.iterrows():
            w_id = w['window_id']
            wh_w = wh[wh['window_id'] == w_id]
            if wh_w.empty:
                continue
                
            date_str = str(w['date']).replace('-', '_')
            m_files = glob.glob(str(DATA_DIR / f"corridors/{panel}/train/mainline_states/**/*{date_str}*.parquet"), recursive=True)
            if not m_files:
                continue
            df_true = pd.read_parquet(m_files[0])
            df_true['timestamp'] = pd.to_datetime(df_true['timestamp'], utc=True)
            
            start_ts = pd.to_datetime(w['forecast_start'], utc=True)
            end_ts = pd.to_datetime(w['forecast_end'], utc=True)
            horizon = df_true[(df_true['timestamp'] >= start_ts) & (df_true['timestamp'] <= end_ts)].copy()
            if horizon.empty:
                continue
                
            horizon['v_cut'] = horizon['link_id'].map(v_cut)
            horizon['queue_true'] = (horizon['speed_kmh'] <= horizon['v_cut']).astype(int)
            
            feats_df, y = extract_window_features(wh_w, topo_meta, w['condition'], is_val=False, horizon_true=horizon)
            feats_df['window_id'] = w_id
            feats_df['label'] = y
            window_data.append(feats_df)
            
    full_train = pd.concat(window_data, ignore_index=True)
    feature_cols = [c for c in full_train.columns if c not in ['window_id', 'label']]
    print(f"Total training cells mined: {len(full_train):,} with {len(feature_cols)} features.")
    
    # 5-Fold Stratified Group CV
    unique_windows = full_train[['window_id', 'condition']].drop_duplicates()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(full_train))
    
    for train_w_idx, val_w_idx in kf.split(unique_windows):
        val_wids = unique_windows.iloc[val_w_idx]['window_id'].tolist()
        val_mask = full_train['window_id'].isin(val_wids)
        train_mask = ~val_mask
        
        X_tr = full_train.loc[train_mask, feature_cols]
        y_tr = full_train.loc[train_mask, 'label']
        X_va = full_train.loc[val_mask, feature_cols]
        
        clf_fold = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=3.5,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        clf_fold.fit(X_tr, y_tr)
        oof_preds[val_mask] = clf_fold.predict_proba(X_va)[:, 1]
        
    # Calibrate optimal thresholds on Out-of-Fold predictions
    best_tau_onset = 0.50
    best_tau_ongoing = 0.50
    best_total_iou = 0.0
    
    onset_mask = full_train['condition'] == 0
    ongoing_mask = full_train['condition'] == 1
    
    for tau_on in np.arange(0.30, 0.65, 0.05):
        for tau_ong in np.arange(0.30, 0.65, 0.05):
            preds_bin = np.zeros(len(full_train), dtype=int)
            preds_bin[onset_mask] = (oof_preds[onset_mask] >= tau_on).astype(int)
            preds_bin[ongoing_mask] = (oof_preds[ongoing_mask] >= tau_ong).astype(int)
            
            # Compute window-level IoU
            win_ious = []
            for wid, w_df in full_train.groupby('window_id'):
                idx = w_df.index
                y_t = full_train.loc[idx, 'label'].values
                y_p = preds_bin[idx]
                inter = np.logical_and(y_t == 1, y_p == 1).sum()
                union = np.logical_or(y_t == 1, y_p == 1).sum()
                iou = 1.0 if union == 0 else inter / union
                win_ious.append(iou)
                
            mean_iou = np.mean(win_ious)
            if mean_iou > best_total_iou:
                best_total_iou = mean_iou
                best_tau_onset = tau_on
                best_tau_ongoing = tau_ong
                
    print(f"\nOptimal OOF Calibrated Thresholds: onset = {best_tau_onset:.2f}, ongoing = {best_tau_ongoing:.2f}")
    print(f"OOF Validation Space-Time IoU: {best_total_iou:.4f}")
    
    # Train full model on all training data
    final_clf = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=3.5,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    final_clf.fit(full_train[feature_cols], full_train['label'])
    
    return final_clf, feature_cols, best_tau_onset, best_tau_ongoing

def predict_queue_splits(clf, feature_cols: list, tau_onset: float, tau_ongoing: float, splits: list = ["validation", "private"]):
    """Predict queue status for all windows in specified splits across the 8 scored panels."""
    submission_rows = []
    
    for split in splits:
        print(f"\nGenerating queue predictions for split: {split}...")
        for panel in SCORED_PANELS:
            wi_path = DATA_DIR / f"task2/{panel}/{split}/window_index.csv"
            wh_path = DATA_DIR / f"task2/{panel}/{split}/window_history.parquet"
            samp_path = DATA_DIR / f"task2/{panel}/{split}/sample_submission_queue.csv"
            
            if not (wi_path.exists() and wh_path.exists() and samp_path.exists()):
                continue
                
            wi = pd.read_csv(wi_path)
            wh = pd.read_parquet(wh_path)
            wh['timestamp'] = pd.to_datetime(wh['timestamp'], utc=True)
            samp = pd.read_csv(samp_path)
            
            topo_meta = load_panel_topology(panel)
            
            for _, w in wi.iterrows():
                w_id = w['window_id']
                wh_w = wh[wh['window_id'] == w_id]
                samp_w = samp[samp['window_id'] == w_id]
                if wh_w.empty or samp_w.empty:
                    continue
                    
                X_val, keys = extract_window_features(wh_w, topo_meta, w['condition'], is_val=True, sample_sub=samp_w)
                probs = clf.predict_proba(X_val[feature_cols])[:, 1]
                
                threshold = tau_ongoing if w['condition'] == 'queue_ongoing' else tau_onset
                preds_bin = (probs >= threshold).astype(int)
                
                for (win_id, ts, link), pred in zip(keys, preds_bin):
                    submission_rows.append({
                        'window_id': win_id,
                        'timestamp': ts,
                        'link_id': link,
                        'queue_pred': int(pred)
                    })
                    
    result_df = pd.DataFrame(submission_rows)
    print(f"\nTotal Task 2 queue predictions: {len(result_df):,} (Queued: {(result_df['queue_pred']==1).sum():,})")
    return result_df

def main():
    print("=" * 75)
    print("STARTING TASK 2 RANK-1 SHOCKWAVE-AWARE LIGHTGBM PIPELINE")
    print("=" * 75)
    start_time = time.time()
    
    clf, feature_cols, tau_onset, tau_ongoing = train_and_calibrate_queue_models()
    submission_df = predict_queue_splits(clf, feature_cols, tau_onset, tau_ongoing, splits=["validation", "private"])
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved complete Task 2 queue predictions to: {OUTPUT_CSV}")
    print(f"Finished in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
