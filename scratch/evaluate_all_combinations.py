"""
Comprehensive Offline Benchmark & Root-Cause Isolation across All 4 Tasks.
Measures exact official competition metrics across all 10 corridors.
"""
import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd

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

def evaluate_task1_panel(panel: str, val_days: list, use_old: bool = False, use_old_physics: bool = False):
    panel_dir = DATA_DIR / "corridors" / panel
    fd = pd.read_csv(panel_dir / "network" / "fd_parameters.csv")
    link_lanes = dict(zip(fd['link_id'].astype(str), fd['lanes'].astype(float)))
    link_fs = dict(zip(fd['link_id'].astype(str), fd['free_speed_kmh'].astype(float)))
    link_cap = dict(zip(fd['link_id'].astype(str), fd['capacity_vph'].astype(float)))
    link_len = dict(zip(fd['link_id'].astype(str), fd['length_km'].astype(float)))
    
    # Evaluate across sample holdout days
    regime_scores = {'R1': [], 'R2': [], 'R3': []}
    
    # We will score on unmasked training days
    return

print("Evaluating Root Cause...")
