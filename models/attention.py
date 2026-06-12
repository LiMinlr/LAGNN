# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:51:10 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding."""
    def __init__(self, d_model, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings


class PositionalAttention(nn.Module):
    """Multi-head attention with positional encoding and feature fusion."""
    def __init__(self, in_dim, num_heads, positional_dim=32, max_seq_len=20):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        if config.use_positional_encoding:
            self.positional_encoding = LearnablePositionalEncoding(in_dim, max_seq_len)

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

        # Optional projections for additional features
        self.frequency_projection = nn.Linear(config.interaction_feature_dim, in_dim) if config.use_interaction_frequency else None
        self.common_neighbor_projection = nn.Linear(config.common_neighbor_dim, in_dim) if config.use_common_neighbors else None

        self.norm = nn.LayerNorm(in_dim)

    def forward(self, node_emb, neighbor_embs, delta_t=None, position_indices=None,
                interaction_features=None, common_neighbor_features=None):
        batch_size, num_neighbors, _ = neighbor_embs.shape

        # Normalize neighbor embeddings
        neighbor_embs = self.norm(neighbor_embs)

        # Apply positional encoding
        if config.use_positional_encoding:
            neighbor_embs = self.positional_encoding(neighbor_embs)

        # Enhance node embedding with additional features
        enhanced_node_emb = node_emb
        if config.use_interaction_frequency and interaction_features is not None and self.frequency_projection is not None:
            freq_projected = self.frequency_projection(interaction_features)
            enhanced_node_emb = enhanced_node_emb + freq_projected
        if config.use_common_neighbors and common_neighbor_features is not None and self.common_neighbor_projection is not None:
            common_projected = self.common_neighbor_projection(common_neighbor_features)
            enhanced_node_emb = enhanced_node_emb + common_projected

        # Compute Q, K, V
        q = self.query(enhanced_node_emb).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(neighbor_embs).view(batch_size, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(neighbor_embs).view(batch_size, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Time decay bias (if enabled)
        if config.use_time and delta_t is not None:
            time_bias = -0.1 * delta_t.unsqueeze(1).unsqueeze(1)
            attn_logits = attn_logits + time_bias

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate
        aggregated = torch.matmul(attn_weights, v).transpose(1, 2)
        aggregated = aggregated.reshape(batch_size, aggregated.size(1), -1).squeeze(1)
        return aggregated