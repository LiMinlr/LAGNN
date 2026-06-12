# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:50:29 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class InteractionFrequencyEncoder(nn.Module):
    """Encode raw interaction features into an embedding."""
    def __init__(self, feature_dim, embedding_dim):
        super(InteractionFrequencyEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, features):
        return self.encoder(features)


class CommonNeighborEncoder(nn.Module):
    """Encode common neighbor count (single feature) into an embedding."""
    def __init__(self, feature_dim, embedding_dim):
        super(CommonNeighborEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim // 4),
            nn.LayerNorm(embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, features):
        return self.encoder(features)