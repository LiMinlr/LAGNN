# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:48:17 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
import torch

class Config:
    data_path = "data/UCI.txt"          
    train_ratio = 0.8
    val_ratio = 0.1
    embedding_dim = 128
    num_layers = 1
    num_heads = 4
    batch_size = 1024
    lr = 0.0001
    epochs = 50
    dropout = 0.4
    weight_decay = 1e-3
    grad_clip = 5.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_position = 30

    # Positional encoding settings
    use_positional_encoding = True
    positional_dim = 128
    max_seq_len = 30

    # Model switches
    use_time = True
    use_in_neighbors = True
    use_out_neighbors = True

    # Interaction frequency settings
    use_interaction_frequency = True
    interaction_feature_dim = 128
    time_window = 10000
    max_interaction_count = 100

    # Common neighbor settings (only count, no future information)
    use_common_neighbors = True
    common_neighbor_dim = 128
    max_common_neighbors = 100

config = Config()