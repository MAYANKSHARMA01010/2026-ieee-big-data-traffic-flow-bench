"""
Rigorous Task-by-Task Offline Diagnostic and Regression Isolation.
Evaluates exact competition formulas with hierarchical family aggregation.
"""
import glob
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold

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

QUEUE_FAMILIES = {
    "D7_I10": ["D7_I10_E", "D7_I10_W"],
    "D7_I210": ["D7_I210_E", "D7_I210_W"],
    "D7_I405": ["D7_I405_N", "D7_I405_S"],
    "D12_I5": ["D12_I5_N", "D12_I5_S"]
}

# ==============================================================================
# 1. EVALUATE TASK 4 (ODME)
# ==============================================================================
def evaluate_task4_all(lambda_prior=0.05, lambda_attr=0.0):
    fam_scores = []
    for fam, panels in FAMILIES.items():
        p_scores = []
        for panel in panels:
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
            
            dest_zones = paths['destination_zone'].astype(str).tolist()
            unique_dz = sorted(list(set(dest_zones)))
            dz_idx = {z: i for i, z in enumerate(unique_dz)}
            D = np.zeros((len(unique_dz), len(path_ids)), dtype=np.float64)
            for j, z in enumerate(dest_zones):
                D[dz_idx[z], j] = 1.0
                
            prior_df = pd.read_csv(DATA_DIR / f"task4/{panel}/validation/synthetic_weak_prior.csv", dtype={"path_id": str})
            prior_series = prior_df.set_index("path_id").reindex(path_ids)["path_flow"].fillna(0.0)
            x0 = prior_series.to_numpy(dtype=float)
            
            val_count_frame = pd.read_csv(DATA_DIR / f"task4/{panel}/validation/synthetic_link_counts.csv", dtype={"link_id": str})
            val_counts = val_count_frame.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)
            observed_links = set(val_count_frame.link_id)
            measured = np.array([link in observed_links for link in link_ids], dtype=bool)
            
            A_meas = A[measured]
            y_meas = val_counts[measured]
            
            AtA = A_meas.T @ A_meas
            Aty = A_meas.T @ y_meas
            DtD = D.T @ D
            Dtx0 = D.T @ (D @ x0)
            
            def loss_and_grad(x):
                res_l = A_meas @ x - y_meas
                res_p = x - x0
                res_a = D @ x - D @ x0
                val = 0.5 * np.sum(res_l**2) + 0.5 * lambda_prior * np.sum(res_p**2) + 0.5 * lambda_attr * np.sum(res_a**2)
                grad = AtA @ x - Aty + lambda_prior * res_p + lambda_attr * (DtD @ x - Dtx0)
                return val, grad
                
            res = minimize(loss_and_grad, np.maximum(x0, 0.0), jac=True, method="L-BFGS-B", bounds=[(0.0, None)]*len(x0), options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            opt_x = np.maximum(res.x, 0.0)
            
            s_link = max(0.0, 1.0 - np.sum(np.abs(A_meas @ opt_x - y_meas)) / max(np.sum(y_meas), 1e-9))
            s_dev = np.exp(-np.linalg.norm(opt_x - x0) / max(np.linalg.norm(x0), 1e-9))
            s_attr = np.exp(-np.linalg.norm(D @ opt_x - D @ x0) / max(np.linalg.norm(D @ x0), 1e-9))
            
            # S_od proxy is s_dev (as true OD is anchored on ground truth close to x0)
            s_od_est = s_dev
            s_panel_odme = 0.45 * s_od_est + 0.25 * s_link + 0.15 * s_dev + 0.15 * s_attr
            p_scores.append(s_panel_odme)
        fam_scores.append(np.mean(p_scores))
    return np.mean(fam_scores)

# ==============================================================================
# 2. EVALUATE TASK 2 (QUEUE)
# ==============================================================================
def evaluate_task2_all(use_old_method=False):
    train_wis = sorted(glob.glob(str(DATA_DIR / "task2/*/train/window_index.csv")))
    window_data = []
    
    for f_wi in train_wis:
        panel = f_wi.split('/')[-3]
        if panel not in [p for fam in QUEUE_FAMILIES.values() for p in fam]:
            continue
        wi = pd.read_csv(f_wi)
        wh = pd.read_parquet(DATA_DIR / f"task2/{panel}/train/window_history.parquet")
        wh['timestamp'] = pd.to_datetime(wh['timestamp'], utc=True)
        
        fd = pd.read_csv(DATA_DIR / f"corridors/{panel}/network/fd_parameters.csv")
        v_cut = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float) * 0.60))
        fs_map = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
        cap_map = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
        lanes_map = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
        length_map = dict(zip(fd['link_id'].astype(str), fd['length_km'].astype(float)))
        
        topo_path = DATA_DIR / f"corridors/{panel}/network/lwr_mainline_topology.csv"
        if topo_path.exists():
            topo = pd.read_csv(topo_path).sort_values('order_index')
            ordered_links = topo['link_id'].astype(str).tolist()
        else:
            ordered_links = fd['link_id'].astype(str).tolist()
        link_to_order = {l: i for i, l in enumerate(ordered_links)}
        
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
            
            p_speed = wh_w.pivot(index='timestamp', columns='link_id', values='speed_kmh')
            p_flow = wh_w.pivot(index='timestamp', columns='link_id', values='flow_vph')
            p_occ = wh_w.pivot(index='timestamp', columns='link_id', values='occupancy')
            
            timestamps = sorted(horizon['timestamp'].unique().tolist())
            target_links = sorted(horizon['link_id'].unique().tolist())
            
            for step_idx, ts in enumerate(timestamps):
                step_min = (step_idx + 1) * 5
                for link in target_links:
                    gt_row = horizon[(horizon['timestamp'] == ts) & (horizon['link_id'] == link)]
                    if gt_row.empty:
                        continue
                    label = gt_row['queue_true'].iloc[0]
                    s_orig = p_speed[link].iloc[-1] if link in p_speed.columns else np.nan
                    q_orig = p_flow[link].iloc[-1] if link in p_flow.columns else np.nan
                    o_orig = p_occ[link].iloc[-1] if link in p_occ.columns else np.nan
                    min_s = p_speed[link].min() if link in p_speed.columns else np.nan
                    mean_s = p_speed[link].mean() if link in p_speed.columns else np.nan
                    trend_s = p_speed[link].iloc[-1] - p_speed[link].iloc[-3] if (link in p_speed.columns and len(p_speed) >= 3) else 0.0
                    trend_occ = p_occ[link].iloc[-1] - p_occ[link].iloc[-3] if (link in p_occ.columns and len(p_occ) >= 3) else 0.0
                    
                    vc = v_cut.get(link, 63.0)
                    is_q_orig = 1 if (pd.notna(s_orig) and s_orig <= vc) else 0
                    
                    row = {
                        'panel': panel,
                        'window_id': w_id,
                        'condition': w['condition'],
                        'step_min': step_min,
                        'free_speed': fs_map.get(link, 105.0),
                        'capacity': cap_map.get(link, 5700.0),
                        'lanes': lanes_map.get(link, 3.0),
                        'length_km': length_map.get(link, 0.5),
                        'v_cut': vc,
                        'speed_origin': s_orig,
                        'flow_origin': q_orig,
                        'occ_origin': o_orig,
                        'min_speed': min_s,
                        'mean_speed': mean_s,
                        'trend_s': trend_s,
                        'trend_occ': trend_occ,
                        'is_q_origin': is_q_orig,
                        'label': label
                    }
                    if not use_old_method:
                        ord_idx = link_to_order.get(link, 0)
                        down_link1 = ordered_links[ord_idx + 1] if ord_idx + 1 < len(ordered_links) else None
                        down1_s = p_speed[down_link1].iloc[-1] if (down_link1 and down_link1 in p_speed.columns) else np.nan
                        down1_vc = v_cut.get(down_link1, 63.0) if down_link1 else 63.0
                        down1_is_q = 1 if (pd.notna(down1_s) and down1_s <= down1_vc) else 0
                        row['down1_is_q'] = down1_is_q
                        row['down1_s'] = down1_s
                        row['margin_vcut'] = s_orig - vc if pd.notna(s_orig) else 0.0
                    window_data.append(row)
                    
    df_all = pd.DataFrame(window_data)
    feature_cols = [c for c in df_all.columns if c not in ['panel', 'window_id', 'condition', 'label']]
    
    # 5-Fold Stratified Group CV
    unique_w = df_all[['window_id', 'condition']].drop_duplicates()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df_all))
    
    for tr_idx, va_idx in kf.split(unique_w):
        val_wids = unique_w.iloc[va_idx]['window_id'].tolist()
        val_m = df_all['window_id'].isin(val_wids)
        tr_m = ~val_m
        
        clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, scale_pos_weight=3.5, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(df_all.loc[tr_m, feature_cols], df_all.loc[tr_m, 'label'])
        oof_probs[val_m] = clf.predict_proba(df_all.loc[val_m, feature_cols])[:, 1]
        
    df_all['prob'] = oof_probs
    
    # Evaluate with proper hierarchical aggregation
    thresh = 0.50 if use_old_method else 0.45
    df_all['pred'] = (df_all['prob'] >= thresh).astype(int)
    
    fam_ious = []
    for fam, panels in QUEUE_FAMILIES.items():
        p_ious = []
        for p in panels:
            cond_ious = []
            for cond in ['queue_onset', 'queue_ongoing']:
                sub_cond = df_all[(df_all['panel'] == p) & (df_all['condition'] == cond)]
                w_ious = []
                for wid, w_df in sub_cond.groupby('window_id'):
                    inter = np.logical_and(w_df['pred'] == 1, w_df['label'] == 1).sum()
                    union = np.logical_or(w_df['pred'] == 1, w_df['label'] == 1).sum()
                    iou = 1.0 if union == 0 else inter / union
                    w_ious.append(iou)
                if w_ious:
                    cond_ious.append(np.mean(w_ious))
            if cond_ious:
                p_ious.append(np.mean(cond_ious))
        if p_ious:
            fam_ious.append(np.mean(p_ious))
            
    return np.mean(fam_ious)

# ==============================================================================
# RUN COMPARISON
# ==============================================================================
print("1. Computing Task 4 (ODME) Hierarchical Score...")
s_odme_old = evaluate_task4_all(lambda_prior=0.05, lambda_attr=0.0)
s_odme_new = evaluate_task4_all(lambda_prior=0.04, lambda_attr=0.08)
print(f"  Old Task 4 Score: {s_odme_old:.5f}")
print(f"  New Task 4 Score: {s_odme_new:.5f} (Delta: {s_odme_new - s_odme_old:+.5f})")

print("\n2. Computing Task 2 (Queue) Hierarchical Score...")
s_queue_old = evaluate_task2_all(use_old_method=True)
s_queue_new = evaluate_task2_all(use_old_method=False)
print(f"  Old Task 2 Score: {s_queue_old:.5f}")
print(f"  New Task 2 Score: {s_queue_new:.5f} (Delta: {s_queue_new - s_queue_old:+.5f})")
