# LAGNN: A Lightweight Graph Neural Network with Decoupled Static and Dynamic Representations for Temporal Link Prediction in Social Networks

This repository implements **LAGNN**, a dynamic graph neural network for temporal link prediction. The model decouples static node identity (embedding) from dynamic context (neighbor aggregation) and incorporates **interaction frequency** and **temporal common neighbor** features into an attention-based framework, ensuring no future information leakage.

## Features
- Chronological train/val/test split
- Positional encoding for neighbor sequences
- Multi-head attention with time decay bias
- Integration of interaction frequency and common neighbor features
- No negative sampling (positive-only training)

## Requirements
See `requirements.txt`. Python 3.8+ recommended.

## Usage
1. Place your dataset (e.g., `UCI.txt`) in the `data/` folder, or modify `config.data_path` in `config/config.py`.
2. Run:
   ```bash
   python main.py