import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import math
import logging
from config.config import config
from models.sdgnn import LAGNN   # 修改导入类名

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer for LAGNN model."""
    def __init__(self, dataset, train_edges, seed=42):
        self.dataset = dataset
        self.model = LAGNN(dataset.num_nodes).to(config.device)   # 修改类名
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        self.reset_interval = 1000
        self.global_step = 0
        self.best_mrr = 0
        self.seed = seed

    def _get_time_delta(self, src_nodes, neighbors_list, batch_timestamps, direction='in'):
        """Compute time differences between source nodes and their neighbors."""
        if not config.use_time:
            time_deltas = [[0.0] * len(neighbors) for neighbors in neighbors_list]
            return torch.FloatTensor(time_deltas).clamp(min=1e-6).to(config.device)
        time_deltas = []
        for src, neighbors, ts in zip(src_nodes, neighbors_list, batch_timestamps):
            if direction == 'in':
                deltas = [ts - self.dataset.time_dict.get((int(n), int(src)), 0) for n in neighbors]
            else:
                deltas = [ts - self.dataset.time_dict.get((int(src), int(n)), 0) for n in neighbors]
            time_deltas.append(deltas)
        return torch.FloatTensor(time_deltas).clamp(min=1e-6).to(config.device)

    def _prepare_batch(self, batch_edges):
        """Prepare a batch of edges for training."""
        src = torch.LongTensor([e[0] for e in batch_edges]).to(config.device)
        pos_dst = torch.LongTensor([e[1] for e in batch_edges]).to(config.device)
        timestamps = torch.LongTensor([e[2] for e in batch_edges]).to(config.device)

        in_neighbor_lists, out_neighbor_lists = [], []
        delta_t_in_lists, delta_t_out_lists = [], []
        interaction_features_list = []
        common_neighbor_features_list = []

        # Collect pair features
        for src_node, dst_node, ts in zip(src.cpu().numpy(), pos_dst.cpu().numpy(), timestamps.cpu().numpy()):
            if config.use_interaction_frequency:
                interaction_feat = self.dataset.get_interaction_features(src_node, dst_node, ts)
                interaction_features_list.append(interaction_feat)
            if config.use_common_neighbors:
                common_feat = self.dataset.get_common_neighbor_features(src_node, dst_node, ts)
                common_neighbor_features_list.append(common_feat)

        if config.use_interaction_frequency:
            interaction_features = torch.FloatTensor(interaction_features_list).to(config.device)
        else:
            interaction_features = None
        if config.use_common_neighbors:
            common_neighbor_features = torch.FloatTensor(common_neighbor_features_list).to(config.device)
        else:
            common_neighbor_features = None

        # Build neighbor lists for each layer
        for _ in range(config.num_layers):
            batch_in_neighbors, batch_out_neighbors = [], []
            for src_node, ts in zip(src.cpu().numpy(), timestamps.cpu().numpy()):
                if config.use_in_neighbors:
                    in_nbrs = self.dataset.get_historical_in_neighbors(src_node, ts)
                    if not in_nbrs:
                        in_nbrs = [src_node]
                    batch_in_neighbors.append(in_nbrs)
                else:
                    batch_in_neighbors.append([src_node])
                if config.use_out_neighbors:
                    out_nbrs = self.dataset.get_historical_out_neighbors(src_node, ts)
                    if not out_nbrs:
                        out_nbrs = [src_node]
                    batch_out_neighbors.append(out_nbrs)
                else:
                    batch_out_neighbors.append([src_node])

            # Pad to equal length
            max_in_len = max(len(nbrs) for nbrs in batch_in_neighbors)
            padded_in_neighbors = []
            for nbrs in batch_in_neighbors:
                if len(nbrs) < max_in_len:
                    nbrs = nbrs + [nbrs[-1]] * (max_in_len - len(nbrs))
                padded_in_neighbors.append(nbrs[:max_in_len])

            max_out_len = max(len(nbrs) for nbrs in batch_out_neighbors)
            padded_out_neighbors = []
            for nbrs in batch_out_neighbors:
                if len(nbrs) < max_out_len:
                    nbrs = nbrs + [nbrs[-1]] * (max_out_len - len(nbrs))
                padded_out_neighbors.append(nbrs[:max_out_len])

            in_neighbors_tensor = torch.LongTensor(padded_in_neighbors).to(config.device)
            out_neighbors_tensor = torch.LongTensor(padded_out_neighbors).to(config.device)

            delta_t_in = self._get_time_delta(src.cpu().numpy(), padded_in_neighbors, timestamps.cpu().numpy(), 'in')
            delta_t_out = self._get_time_delta(src.cpu().numpy(), padded_out_neighbors, timestamps.cpu().numpy(), 'out')

            in_neighbor_lists.append(in_neighbors_tensor)
            out_neighbor_lists.append(out_neighbors_tensor)
            delta_t_in_lists.append(delta_t_in)
            delta_t_out_lists.append(delta_t_out)

        return (src, pos_dst,
                in_neighbor_lists, out_neighbor_lists,
                delta_t_in_lists, delta_t_out_lists,
                timestamps,
                interaction_features,
                common_neighbor_features)

    def train_epoch(self, train_edges, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        random.shuffle(train_edges)
        batch_iter = tqdm(range(0, len(train_edges), config.batch_size), desc=f"Epoch {epoch+1} [Seed {self.seed}]")

        for i, start_idx in enumerate(batch_iter):
            end_idx = start_idx + config.batch_size
            batch = train_edges[start_idx:end_idx]

            # Periodically reset memory for nodes seen in previous batch
            if i > 0 and i % self.reset_interval == 0:
                reset_nodes = torch.LongTensor([e[0] for e in train_edges[start_idx-self.reset_interval:start_idx]]).unique()
                self.model.reset_memory(reset_nodes)

            (src, pos_dst,
             in_neighbors, out_neighbors,
             delta_t_in, delta_t_out,
             timestamps,
             interaction_features,
             common_neighbor_features) = self._prepare_batch(batch)

            src_emb = self.model(
                src,
                in_neighbors, out_neighbors,
                delta_t_in, delta_t_out,
                timestamps,
                interaction_features,
                common_neighbor_features
            )

            # Positive sample loss (no negative sampling)
            pos_scores = (src_emb[:, :config.embedding_dim] * self.model.embedding(pos_dst)).sum(dim=1)
            loss = -F.logsigmoid(pos_scores).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1

            if i % 10 == 0:
                batch_iter.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PosScore': f'{pos_scores.mean().item():.4f}',
                    'IntFreq': f'{interaction_features.mean().item() if interaction_features is not None else 0:.2f}',
                    'CommNeigh': f'{common_neighbor_features.mean().item() if common_neighbor_features is not None else 0:.2f}'
                })

        self.scheduler.step(total_loss)
        model_filename = f"best_model_seed_{self.seed}.pth"
        return total_loss / (len(train_edges)/config.batch_size), model_filename

    def _prepare_single_edge(self, src, dst, ts):
        """Prepare a single edge for evaluation."""
        if config.use_in_neighbors:
            in_nbrs = self.dataset.get_historical_in_neighbors(src, ts)
            if not in_nbrs:
                in_nbrs = [src]
            max_in_len = len(in_nbrs)
            if len(in_nbrs) < max_in_len:
                in_nbrs = in_nbrs + [in_nbrs[-1]] * (max_in_len - len(in_nbrs))
            padded_in_neighbors = [in_nbrs[:max_in_len]]
            in_neighbors_tensor = torch.LongTensor(padded_in_neighbors).to(config.device)
            delta_t_in = self._get_time_delta([src], padded_in_neighbors, [ts], 'in')
        else:
            in_neighbors_tensor = None
            delta_t_in = None

        if config.use_out_neighbors:
            out_nbrs = self.dataset.get_historical_out_neighbors(src, ts)
            if not out_nbrs:
                out_nbrs = [src]
            max_out_len = len(out_nbrs)
            if len(out_nbrs) < max_out_len:
                out_nbrs = out_nbrs + [out_nbrs[-1]] * (max_out_len - len(out_nbrs))
            padded_out_neighbors = [out_nbrs[:max_out_len]]
            out_neighbors_tensor = torch.LongTensor(padded_out_neighbors).to(config.device)
            delta_t_out = self._get_time_delta([src], padded_out_neighbors, [ts], 'out')
        else:
            out_neighbors_tensor = None
            delta_t_out = None

        in_neighbor_lists = [in_neighbors_tensor] * config.num_layers if in_neighbors_tensor is not None else None
        out_neighbor_lists = [out_neighbors_tensor] * config.num_layers if out_neighbors_tensor is not None else None
        delta_t_in_lists = [delta_t_in] * config.num_layers if delta_t_in is not None else None
        delta_t_out_lists = [delta_t_out] * config.num_layers if delta_t_out is not None else None

        timestamps_tensor = torch.LongTensor([ts]).to(config.device)

        if config.use_interaction_frequency:
            interaction_feat = self.dataset.get_interaction_features(src, dst, ts)
            interaction_features = torch.FloatTensor([interaction_feat]).to(config.device)
        else:
            interaction_features = None

        if config.use_common_neighbors:
            common_feat = self.dataset.get_common_neighbor_features(src, dst, ts)
            common_neighbor_features = torch.FloatTensor([common_feat]).to(config.device)
        else:
            common_neighbor_features = None

        return (in_neighbor_lists, out_neighbor_lists,
                delta_t_in_lists, delta_t_out_lists,
                timestamps_tensor,
                interaction_features,
                common_neighbor_features)

    def evaluate(self, edges, mode="val"):
        """Evaluate on given edges (validation or test)."""
        self.model.eval()
        ranks = []
        recall_k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
        recall_at_k = {k: [] for k in recall_k_values}

        with torch.no_grad():
            if mode == "val" or mode == "test":
                self.model.reset_memory()

            for src, dst, ts in tqdm(edges, desc=f"Evaluating {mode} [Seed {self.seed}]"):
                (in_neighbor_lists, out_neighbor_lists,
                 delta_t_in_lists, delta_t_out_lists,
                 timestamps_tensor,
                 interaction_features,
                 common_neighbor_features) = self._prepare_single_edge(src, dst, ts)

                src_emb = self.model(
                    torch.LongTensor([src]).to(config.device),
                    in_neighbor_lists, out_neighbor_lists,
                    delta_t_in_lists, delta_t_out_lists,
                    timestamps_tensor,
                    interaction_features,
                    common_neighbor_features
                )

                all_nodes = torch.arange(self.dataset.num_nodes).to(config.device)
                candidate_embs = self.model.embedding(all_nodes)
                scores = torch.mm(src_emb[:, :config.embedding_dim], candidate_embs.t()).squeeze(0)

                _, sorted_indices = torch.sort(scores, descending=True)
                rank = (sorted_indices == dst).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

                for k in recall_k_values:
                    recall_at_k[k].append(1 if rank <= k else 0)

        ranks = np.array(ranks)
        metrics = {
            "MRR": np.mean(1 / ranks),
            "HITS@1": np.mean(ranks <= 1),
            "HITS@3": np.mean(ranks <= 3),
            "HITS@5": np.mean(ranks <= 5),
            "HITS@10": np.mean(ranks <= 10),
            "HITS@20": np.mean(ranks <= 20),
            "HITS@50": np.mean(ranks <= 50),
            "HITS@100": np.mean(ranks <= 100),
            "HITS@200": np.mean(ranks <= 200),
            "HITS@500": np.mean(ranks <= 500),
            "HITS@1000": np.mean(ranks <= 1000)
        }
        for k in recall_k_values:
            metrics[f"Recall@{k}"] = np.mean(recall_at_k[k])
        return metrics, ranks

    def train(self, train_edges, val_edges, test_edges):
        """Full training loop with early stopping."""
        patience = 5
        early_stop_counter = 0

        logger.info(f"\n=== LAGNN Experiment (No Negative Sampling) - Random Seed: {self.seed} ===")   # 修改日志
        logger.info(f"Device: {config.device}")
        logger.info(f"Total nodes: {self.dataset.num_nodes}")
        logger.info(f"Train edges: {len(train_edges)}, Val edges: {len(val_edges)}, Test edges: {len(test_edges)}")
        logger.info(f"Positional encoding: {config.use_positional_encoding}")
        logger.info(f"Time information: {config.use_time}")
        logger.info(f"In-neighbors: {config.use_in_neighbors}")
        logger.info(f"Out-neighbors: {config.use_out_neighbors}")
        logger.info(f"Interaction frequency encoding: {config.use_interaction_frequency} (integrated into attention)")
        logger.info(f"Common neighbor encoding: {config.use_common_neighbors} (only historical count, integrated into attention)")

        best_mrr = 0
        best_model_path = None

        for epoch in range(config.epochs):
            self.model.reset_memory()
            loss, model_filename = self.train_epoch(train_edges, epoch)
            val_metrics, _ = self.evaluate(val_edges, mode="val")

            logger.info(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val MRR: {val_metrics['MRR']:.4f} | "
                        f"HITS@10: {val_metrics['HITS@10']:.4f}")

            if val_metrics['MRR'] > best_mrr:
                best_mrr = val_metrics['MRR']
                best_model_path = model_filename
                torch.save(self.model.state_dict(), model_filename)
                logger.info(f"Model improved, best val MRR: {best_mrr:.4f}, saved to {model_filename}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"Early stopping after {patience} epochs without improvement.")
                    break

        if best_model_path:
            self.model.load_state_dict(torch.load(best_model_path))
        test_metrics, test_ranks = self.evaluate(test_edges, mode="test")
        self._print_feature_stats(test_edges)
        return test_metrics, test_ranks

    def _print_feature_stats(self, edges):
        """Print statistics of interaction and common neighbor features on the given edges."""
        if config.use_interaction_frequency or config.use_common_neighbors:
            interaction_features = []
            common_neighbor_features = []
            for src, dst, ts in edges:
                if config.use_interaction_frequency:
                    interaction_feat = self.dataset.get_interaction_features(src, dst, ts)
                    interaction_features.append(interaction_feat[0])
                if config.use_common_neighbors:
                    common_feat = self.dataset.get_common_neighbor_features(src, dst, ts)
                    common_neighbor_features.append(common_feat[0])
            logger.info(f"\nFeature statistics (test set):")
            if config.use_interaction_frequency and interaction_features:
                logger.info(f"  Interaction count:")
                logger.info(f"    Mean: {np.mean(interaction_features):.2f}")
                logger.info(f"    Median: {np.median(interaction_features):.2f}")
                logger.info(f"    Max: {np.max(interaction_features):.2f}")
                logger.info(f"    Min: {np.min(interaction_features):.2f}")
            if config.use_common_neighbors and common_neighbor_features:
                logger.info(f"  Common neighbor count (historical):")
                logger.info(f"    Mean: {np.mean(common_neighbor_features):.2f}")
                logger.info(f"    Median: {np.median(common_neighbor_features):.2f}")
                logger.info(f"    Max: {np.max(common_neighbor_features):.2f}")
                logger.info(f"    Min: {np.min(common_neighbor_features):.2f}")
                logger.info(f"    Zero common neighbor ratio: {np.sum(np.array(common_neighbor_features) == 0) / len(common_neighbor_features):.2%}")