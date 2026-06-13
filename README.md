# LAGNN: A Lightweight Graph Neural Network with Decoupled Static and Dynamic Representations for Temporal Link Prediction in Social Networks

This repository implements LAGNN, a dynamic graph neural network for temporal link prediction. The model decouples static node identity (embedding) from dynamic context (neighbor aggregation) and incorporates interaction frequency and temporal common neighbor features into an attention-based framework, ensuring no future information leakage.

## Features
- Chronological train/val/test split
- Positional encoding for neighbor sequences
- Multi-head attention with time decay bias
- Integration of interaction frequency and common neighbor features
- No negative sampling (positive-only training)

## Optimal Hyperparameters (UCI Dataset)
### LAGNN best configuration
- embedding_dim = 128    # Node embedding dimension
- learning_rate = 1e-4   # Adam learning rate
- batch_size = 1024      # Training batch size
- dropout = 0.4          # Dropout rate for regularization

## Requirements
- See requirements.txt.
- Python 3.8+ recommended.

## Usage
Place your dataset (e.g., UCI.txt) in the data/ folder, or modify config.data_path in config/config.py.

Run the command below:
```bash
python main.py
