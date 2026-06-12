# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:51:45 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config
from models.encoders import InteractionFrequencyEncoder, CommonNeighborEncoder
from models.attention import PositionalAttention

class LAGNN(nn.Module):
    """LAGNN: Static and Dynamic Graph Neural Network for temporal link prediction.
    Decouples static identity (node embedding) and dynamic context (neighbor aggregation)."""
    def __init__(self, num_nodes):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, config.embedding_dim)
        self.embedding.weight.requires_grad = False
        self.register_buffer('node_memory', torch.zeros(num_nodes, config.embedding_dim))
        self.register_buffer('node_last_time', torch.zeros(num_nodes))

        # Attention layers for in/out neighbors
        self.attn_layers_in = nn.ModuleList()
        self.attn_layers_out = nn.ModuleList()
        attn_kwargs = {
            'in_dim': config.embedding_dim,
            'num_heads': config.num_heads,
            'positional_dim': config.positional_dim,
            'max_seq_len': config.max_seq_len
        }
        for _ in range(config.num_layers):
            if config.use_in_neighbors:
                self.attn_layers_in.append(PositionalAttention(**attn_kwargs))
            if config.use_out_neighbors:
                self.attn_layers_out.append(PositionalAttention(**attn_kwargs))

        # Feature encoders
        if config.use_interaction_frequency:
            self.interaction_encoder = InteractionFrequencyEncoder(feature_dim=4, embedding_dim=config.interaction_feature_dim)
        if config.use_common_neighbors:
            self.common_neighbor_encoder = CommonNeighborEncoder(feature_dim=1, embedding_dim=config.common_neighbor_dim)

        # Fusion layers
        self.fusion_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            fusion_input_dim = config.embedding_dim
            if config.use_in_neighbors or config.use_out_neighbors:
                fusion_input_dim = 2 * config.embedding_dim
            if config.use_interaction_frequency:
                fusion_input_dim += config.interaction_feature_dim
            if config.use_common_neighbors:
                fusion_input_dim += config.common_neighbor_dim
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Linear(fusion_input_dim, config.embedding_dim * 2),
                    nn.LayerNorm(config.embedding_dim * 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                    nn.LayerNorm(config.embedding_dim)
                )
            )

        nn.init.xavier_normal_(self.embedding.weight)

    def reset_memory(self, nodes=None):
        if nodes is None:
            self.node_memory.zero_()
        else:
            self.node_memory[nodes] = 0

    def forward(self, nodes,
                in_neighbors_list=None, out_neighbors_list=None,
                delta_t_in_list=None, delta_t_out_list=None,
                timestamps=None,
                interaction_features=None,
                common_neighbor_features=None):
        timestamps = timestamps.float()
        h = self.embedding(nodes)

        # Encode additional features if provided
        if config.use_interaction_frequency and interaction_features is not None:
            interaction_emb = self.interaction_encoder(interaction_features)
        else:
            interaction_emb = None

        if config.use_common_neighbors and common_neighbor_features is not None:
            common_neighbor_emb = self.common_neighbor_encoder(common_neighbor_features)
        else:
            common_neighbor_emb = None

        for i in range(config.num_layers):
            aggregated_list = []

            # Process in-neighbors
            if config.use_in_neighbors and in_neighbors_list is not None:
                in_neighbors = in_neighbors_list[i]
                in_neighbor_emb = self.embedding(in_neighbors)
                if config.use_time and delta_t_in_list is not None:
                    delta_t_in = delta_t_in_list[i].float().clamp(min=1e-6)
                else:
                    delta_t_in = None
                agg_in = self.attn_layers_in[i](
                    h[:, :config.embedding_dim],
                    in_neighbor_emb, delta_t_in,
                    interaction_features=interaction_emb,
                    common_neighbor_features=common_neighbor_emb
                )
                aggregated_list.append(agg_in)

            # Process out-neighbors
            if config.use_out_neighbors and out_neighbors_list is not None:
                out_neighbors = out_neighbors_list[i]
                out_neighbor_emb = self.embedding(out_neighbors)
                if config.use_time and delta_t_out_list is not None:
                    delta_t_out = delta_t_out_list[i].float().clamp(min=1e-6)
                else:
                    delta_t_out = None
                agg_out = self.attn_layers_out[i](
                    h[:, :config.embedding_dim],
                    out_neighbor_emb, delta_t_out,
                    interaction_features=interaction_emb,
                    common_neighbor_features=common_neighbor_emb
                )
                aggregated_list.append(agg_out)

            # Fusion
            if len(aggregated_list) > 0:
                if len(aggregated_list) == 1:
                    aggregated = aggregated_list[0]
                else:
                    aggregated = torch.stack(aggregated_list).mean(dim=0)

                fusion_input = torch.cat([h[:, :config.embedding_dim], aggregated], dim=1)
                extra_features = []
                if config.use_interaction_frequency and interaction_emb is not None:
                    extra_features.append(interaction_emb)
                if config.use_common_neighbors and common_neighbor_emb is not None:
                    extra_features.append(common_neighbor_emb)
                if extra_features:
                    fusion_input = torch.cat([fusion_input] + extra_features, dim=1)
                merged = self.fusion_layers[i](fusion_input) + h[:, :config.embedding_dim]
            else:
                merged = h[:, :config.embedding_dim]
                extra_features = []
                if config.use_interaction_frequency and interaction_emb is not None:
                    extra_features.append(interaction_emb)
                if config.use_common_neighbors and common_neighbor_emb is not None:
                    extra_features.append(common_neighbor_emb)
                if extra_features:
                    fusion_input = torch.cat([merged] + extra_features, dim=1)
                    merged = self.fusion_layers[i](fusion_input) + h[:, :config.embedding_dim]

            # Reconstruct h with all features
            merged_features = [merged]
            if config.use_interaction_frequency and interaction_emb is not None:
                merged_features.append(interaction_emb)
            if config.use_common_neighbors and common_neighbor_emb is not None:
                merged_features.append(common_neighbor_emb)
            h = torch.cat(merged_features, dim=1)

            if i < config.num_layers - 1:
                h = F.leaky_relu(h, 0.1)

        # Update node memory
        self.node_memory[nodes] = h[:, :config.embedding_dim].detach()
        self.node_last_time[nodes] = timestamps.detach()
        return h.clamp(-50, 50)