# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:53:24 2026

@author: Administrator
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict

def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def compute_statistics(results: List[Dict]) -> Dict:
    """Compute statistics (mean, std, min, max) over multiple experiment results."""
    metrics_names = list(results[0].keys())
    stats = {}
    for metric in metrics_names:
        values = [result[metric] for result in results]
        stats[f"{metric}_mean"] = np.mean(values)
        stats[f"{metric}_std"] = np.std(values)
        stats[f"{metric}_min"] = np.min(values)
        stats[f"{metric}_max"] = np.max(values)
    return stats
