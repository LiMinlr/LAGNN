import logging
import sys
import numpy as np
import torch
from config.config import config
from data.dataset import TemporalDataset
from training.trainer import Trainer
from utils.utils import set_random_seed, compute_statistics

def setup_logger(log_file='experiment.log'):
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def run_experiment_with_seed(seed: int, dataset_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment (no negative sampling) - Random seed: {seed}")
    logger.info(f"{'='*60}")

    set_random_seed(seed)
    dataset = TemporalDataset(dataset_path)
    train_edges, val_edges, test_edges = dataset.split_data()
    trainer = Trainer(dataset, train_edges, seed=seed)
    test_metrics, test_ranks = trainer.train(train_edges, val_edges, test_edges)

    logger.info(f"\nSeed {seed} test results (no negative sampling):")
    logger.info(f"Test MRR: {test_metrics['MRR']:.4f}")
    logger.info(f"Test HITS@10: {test_metrics['HITS@10']:.4f}")
    logger.info(f"Test HITS@20: {test_metrics['HITS@20']:.4f}")

    return test_metrics, test_ranks

def print_results_summary(all_results, seeds):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*100)
    logger.info("LAGNN Results Summary (with interaction frequency and common neighbor encoding)")   # 修改
    logger.info("="*100)

    metrics_order = ['MRR', 'HITS@1', 'HITS@3', 'HITS@5', 'HITS@10', 'HITS@20', 'HITS@50', 'HITS@100','HITS@200','HITS@500','HITS@1000']

    header = f"{'Metric':<10} | " + " | ".join([f"Seed {seed:<10}" for seed in seeds])
    logger.info("\n" + header)
    logger.info("-" * (10 + 15 * len(seeds)))

    for metric in metrics_order:
        row_values = []
        for i, seed in enumerate(seeds):
            if i < len(all_results) and metric in all_results[i]:
                row_values.append(f"{all_results[i][metric]:.4f}")
            else:
                row_values.append("N/A")
        logger.info(f"{metric:<10} | " + " | ".join([f"{v:>10}" for v in row_values]))

    logger.info("\n" + "-" * (10 + 15 * len(seeds)))
    logger.info("Summary (mean ± std)")
    logger.info("-" * (10 + 15 * len(seeds)))

    for metric in metrics_order:
        values = []
        for i in range(min(len(all_results), len(seeds))):
            if metric in all_results[i]:
                values.append(all_results[i][metric])
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric:<10} | {mean_val:.4f} ± {std_val:.4f}  [range: {np.min(values):.4f} - {np.max(values):.4f}]")
        else:
            logger.info(f"{metric:<10} | N/A")

def main():
    logger = setup_logger(log_file='lagnn_experiment.log')   # 修改日志文件名
    seeds = [42, 43, 44, 45, 46]
    all_results = []
    all_ranks = []

    logger.info(f"\n{'#'*80}")
    logger.info(f"Starting LAGNN experiment (with interaction frequency and common neighbor encoding) using {len(seeds)} random seeds")   # 修改
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Model configuration:")
    logger.info(f"  Positional encoding: {config.use_positional_encoding}")
    logger.info(f"  Time information: {config.use_time}")
    logger.info(f"  In-neighbors: {config.use_in_neighbors}")
    logger.info(f"  Out-neighbors: {config.use_out_neighbors}")
    logger.info(f"  Interaction frequency encoding: {config.use_interaction_frequency} (integrated into attention)")
    logger.info(f"  Common neighbor encoding: {config.use_common_neighbors} (only historical count, integrated into attention)")
    logger.info(f"  Negative sampling: disabled")
    logger.info(f"{'#'*80}")

    for i, seed in enumerate(seeds):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Running experiment {i+1}/{len(seeds)} - Random seed: {seed}")
        logger.info(f"{'#'*80}")
        test_metrics, test_ranks = run_experiment_with_seed(seed, config.data_path)
        all_results.append(test_metrics)
        all_ranks.append(test_ranks)

    print_results_summary(all_results, seeds)

    # Save detailed results to txt file
    with open("lagnn_results_with_interaction_features.txt", "w") as f:   # 修改输出文件名
        f.write("="*100 + "\n")
        f.write("LAGNN Experiment Results (with interaction frequency and common neighbor encoding)\n")   # 修改
        f.write("="*100 + "\n\n")
        f.write("Model configuration:\n")
        f.write(f"  Positional encoding: {config.use_positional_encoding}\n")
        f.write(f"  Time information: {config.use_time}\n")
        f.write(f"  In-neighbors: {config.use_in_neighbors}\n")
        f.write(f"  Out-neighbors: {config.use_out_neighbors}\n")
        f.write(f"  Interaction frequency encoding: {config.use_interaction_frequency} (integrated into attention)\n")
        f.write(f"  Common neighbor encoding: {config.use_common_neighbors} (only historical count, integrated into attention)\n")
        f.write(f"  Negative sampling: disabled\n\n")
        for i, seed in enumerate(seeds):
            if i < len(all_results):
                f.write(f"\n{'#'*80}\n")
                f.write(f"Random seed: {seed}\n")
                f.write(f"{'#'*80}\n\n")
                for metric in ['MRR', 'HITS@1', 'HITS@5', 'HITS@10', 'HITS@20', 'HITS@50', 'HITS@100','HITS@200','HITS@500','HITS@1000']:
                    if metric in all_results[i]:
                        f.write(f"{metric}: {all_results[i][metric]:.4f}\n")
        f.write(f"\n{'#'*80}\n")
        f.write("Statistical summary (across all seeds)\n")
        f.write(f"{'#'*80}\n\n")
        stats = compute_statistics(all_results)
        for metric in ['MRR', 'HITS@1', 'HITS@5', 'HITS@10', 'HITS@20', 'HITS@50', 'HITS@100','HITS@200','HITS@500','HITS@1000']:
            if f"{metric}_mean" in stats:
                mean = stats[f"{metric}_mean"]
                std = stats[f"{metric}_std"]
                min_val = stats[f"{metric}_min"]
                max_val = stats[f"{metric}_max"]
                f.write(f"{metric}: {mean:.4f} ± {std:.4f}  [range: {min_val:.4f} - {max_val:.4f}]\n")

    logger.info("\n" + "="*100)
    logger.info("LAGNN experiment completed!")   # 修改
    logger.info("Results saved to:")
    logger.info("  - lagnn_results_with_interaction_features.txt (detailed results)")   # 修改
    if len(seeds) > 1:
        logger.info("  - seed_comparison_with_features.png (seed comparison plot)")
    logger.info("="*100)

if __name__ == "__main__":
    main()