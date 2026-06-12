# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:49:06 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from config.config import config

class EfficientPairInteraction:
    """Efficient storage and computation of pairwise interaction features."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.interaction_count = defaultdict(int)
        self.recent_interaction = defaultdict(list)
        self.last_interaction_time = defaultdict(float)
        self.interaction_timestamps = defaultdict(list)
        self.all_interactions = []

    def add_interaction(self, u, v, timestamp):
        """Record an interaction between u and v at given timestamp."""
        for (x, y) in [(u, v), (v, u)]:
            key = (x, y)
            self.interaction_count[key] += 1
            if len(self.recent_interaction[key]) < 10:
                self.recent_interaction[key].append(timestamp)
            else:
                self.recent_interaction[key].pop(0)
                self.recent_interaction[key].append(timestamp)
            self.last_interaction_time[key] = timestamp
            self.interaction_timestamps[key].append(timestamp)
        self.all_interactions.append((timestamp, u, v))

    def get_interaction_features(self, u, v, current_timestamp):
        """Return interaction features for pair (u, v) using only history before current_timestamp."""
        key = (u, v)
        if key not in self.interaction_count:
            return np.zeros(4)

        total_count = self.interaction_count[key]
        normalized_count = min(total_count / config.max_interaction_count, 1.0)

        recent_count = 0
        for ts in self.recent_interaction.get(key, []):
            if ts < current_timestamp and current_timestamp - ts <= config.time_window:
                recent_count += 1
        normalized_recent = min(recent_count / 10, 1.0)

        last_time = self.last_interaction_time[key]
        if last_time < current_timestamp:
            time_gap = current_timestamp - last_time
        else:
            time_gap = config.time_window
        normalized_gap = min(time_gap / config.time_window, 1.0)

        timestamps = self.interaction_timestamps.get(key, [])
        historical_timestamps = [ts for ts in timestamps if ts < current_timestamp]
        if len(historical_timestamps) >= 3:
            intervals = np.diff(sorted(historical_timestamps))
            if len(intervals) > 1:
                regularity = np.std(intervals) / (np.mean(intervals) + 1e-6)
                regularity = min(regularity, 1.0)
            else:
                regularity = 0.0
        else:
            regularity = 1.0

        return np.array([normalized_count, normalized_recent, normalized_gap, regularity])

    def get_interaction_history(self, u, v, max_history=10):
        """Return the most recent timestamps of interactions between u and v."""
        key = (u, v)
        timestamps = self.interaction_timestamps.get(key, [])
        return sorted(timestamps)[-max_history:]


class TemporalCommonNeighbors:
    """Temporal common neighbor computation (only uses historical edges)."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edge_timestamps = defaultdict(list)  # node -> [(timestamp, neighbor)]
        self.common_neighbor_cache = {}

    def add_edge(self, src, dst, timestamp):
        """Add an edge to the temporal graph."""
        self.edge_timestamps[src].append((timestamp, dst))
        self.edge_timestamps[dst].append((timestamp, src))
        self.edge_timestamps[src].sort(key=lambda x: x[0])
        self.edge_timestamps[dst].sort(key=lambda x: x[0])
        self._clear_cache_involving_nodes(src, dst)

    def _clear_cache_involving_nodes(self, u, v):
        """Invalidate cache entries that involve either u or v."""
        keys_to_remove = []
        for key in self.common_neighbor_cache.keys():
            if u in key[:2] or v in key[:2]:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.common_neighbor_cache[key]

    def _get_historical_neighbors(self, node, timestamp):
        """Return neighbors of node before the given timestamp."""
        neighbors = set()
        for ts, neighbor in self.edge_timestamps.get(node, []):
            if ts < timestamp:
                neighbors.add(neighbor)
            else:
                break
        return neighbors

    def get_common_neighbors_count(self, u, v, timestamp):
        """Return number of common neighbors between u and v before timestamp."""
        cache_key = (u, v, timestamp)
        if cache_key in self.common_neighbor_cache:
            return self.common_neighbor_cache[cache_key]
        neighbors_u = self._get_historical_neighbors(u, timestamp)
        neighbors_v = self._get_historical_neighbors(v, timestamp)
        common_count = len(neighbors_u & neighbors_v)
        self.common_neighbor_cache[cache_key] = common_count
        return common_count

    def get_common_neighbor_features(self, u, v, timestamp):
        """Return normalized common neighbor count as a feature vector."""
        common_count = self.get_common_neighbors_count(u, v, timestamp)
        normalized_count = min(common_count / config.max_common_neighbors, 1.0)
        return np.array([normalized_count])


class TemporalDataset:
    """Load and preprocess temporal graph data."""
    def __init__(self, file_path):
        raw_data = np.loadtxt(file_path, dtype=np.int64)
        self._preprocess(raw_data)

    def _preprocess(self, raw_data):
        all_nodes = np.unique(np.concatenate([raw_data[:,0], raw_data[:,1]]))
        self.node2id = {n:i for i,n in enumerate(all_nodes)}
        self.num_nodes = len(all_nodes)
        self.edges = []
        self.time_dict = {}
        self.node_in_neighbors = defaultdict(list)
        self.node_out_neighbors = defaultdict(list)
        self.node_degrees = np.zeros(self.num_nodes)
        self.node_last_time = np.zeros(self.num_nodes)

        # Initialize feature calculators
        self.interaction_stats = EfficientPairInteraction(self.num_nodes)
        self.common_neighbor_stats = TemporalCommonNeighbors(self.num_nodes)

        for src, dst, ts in raw_data:
            src_id = self.node2id[src]
            dst_id = self.node2id[dst]
            self.edges.append((src_id, dst_id, ts))
            self.time_dict[(src_id, dst_id)] = ts

            self.node_out_neighbors[src_id].append((dst_id, ts))
            self.node_in_neighbors[dst_id].append((src_id, ts))

            self.node_degrees[src_id] += 1
            self.node_degrees[dst_id] += 1

            if ts > self.node_last_time[src_id]:
                self.node_last_time[src_id] = ts
            if ts > self.node_last_time[dst_id]:
                self.node_last_time[dst_id] = ts

            if config.use_interaction_frequency:
                self.interaction_stats.add_interaction(src_id, dst_id, ts)
            if config.use_common_neighbors:
                self.common_neighbor_stats.add_edge(src_id, dst_id, ts)

        self.edges.sort(key=lambda x: x[2])

        # Sort neighbor lists by timestamp
        for node in self.node_out_neighbors:
            self.node_out_neighbors[node].sort(key=lambda x: x[1])
        for node in self.node_in_neighbors:
            self.node_in_neighbors[node].sort(key=lambda x: x[1])

        print(f"Dataset statistics:")
        print(f"  Total nodes: {self.num_nodes}")
        print(f"  Total edges: {len(self.edges)}")
        if config.use_interaction_frequency and len(self.edges) > 0:
            src, dst, ts = self.edges[-1]
            features = self.interaction_stats.get_interaction_features(src, dst, ts)
            print(f"  Example interaction features: {features}")

    def split_data(self):
        """Split edges into train/val/test chronologically."""
        n = len(self.edges)
        train_end = int(n * config.train_ratio)
        val_end = train_end + int(n * config.val_ratio)
        return self.edges[:train_end], self.edges[train_end:val_end], self.edges[val_end:]

    def get_historical_in_neighbors(self, node, timestamp, max_neighbors=config.max_position):
        """Get in-neighbors of node before timestamp, sorted by recency."""
        neighbors = []
        all_neighbors = self.node_in_neighbors.get(node, [])
        neighbor_info = []
        for idx in range(len(all_neighbors)-1, -1, -1):
            n, t = all_neighbors[idx]
            if t < timestamp:
                delta_t = timestamp - t
                neighbor_info.append((n, delta_t))
                if len(neighbor_info) >= max_neighbors:
                    break
        neighbor_info.sort(key=lambda x: x[1])  # sort by time gap (smallest first)
        neighbors = [item[0] for item in neighbor_info]
        return neighbors

    def get_historical_out_neighbors(self, node, timestamp, max_neighbors=config.max_position):
        """Get out-neighbors of node before timestamp, sorted by recency."""
        neighbors = []
        all_neighbors = self.node_out_neighbors.get(node, [])
        neighbor_info = []
        for idx in range(len(all_neighbors)-1, -1, -1):
            n, t = all_neighbors[idx]
            if t < timestamp:
                delta_t = timestamp - t
                neighbor_info.append((n, delta_t))
                if len(neighbor_info) >= max_neighbors:
                    break
        neighbor_info.sort(key=lambda x: x[1])
        neighbors = [item[0] for item in neighbor_info]
        return neighbors

    def get_interaction_features(self, u, v, timestamp):
        """Get interaction frequency features for pair (u, v) at given timestamp."""
        if config.use_interaction_frequency:
            return self.interaction_stats.get_interaction_features(u, v, timestamp)
        else:
            return np.zeros(4)

    def get_common_neighbor_features(self, u, v, timestamp):
        """Get common neighbor features for pair (u, v) at given timestamp."""
        if config.use_common_neighbors:
            return self.common_neighbor_stats.get_common_neighbor_features(u, v, timestamp)
        else:
            return np.zeros(1)