# Protein Secondary Structure Prediction

A comprehensive machine learning toolkit for predicting protein secondary structure (SST3 and SST8) from amino acid sequences. This project implements multiple modeling approaches including **Transformers**, **Random Forest**, and **XGBoost**, with integrated hyperparameter optimization and experiment tracking.

## Overview

Secondary structure prediction is a fundamental task in bioinformatics. This project provides a robust pipeline to:
- Preprocess protein sequence data and handle standard/non-standard amino acids.
- Train state-of-the-art **Transformer-based models** for sequence-to-sequence prediction.
- Implement efficient baseline models using **Random Forest** and **XGBoost** with window-based feature extraction.
- Track all experiments and model versions using **MLflow**.
- Optimize hyperparameters automatically with **Optuna**.

## Results

| Model | Task | Accuracy |
|-------|------|----------|
| **Random Forest** | SST3 | **70.8%** |
| Transformer | SST3/SST8 | TBD |
| XGBoost | SST3 | TBD |

*Note: The current highest accuracy achieved in this project is **70.8%**.*

## Project Structure

- `model.py`: Transformer architecture implementation using PyTorch.
- `data_utils.py`: Data loading, vocabulary management, and sequence padding.
- `preprocess.py`: Initial data processing and vocabulary generation.
- `train.py`: Main training script for the Transformer model.
- `train_rf.py` / `train_xgb.py`: Training scripts for Random Forest and XGBoost baselines.
- `optimize.py` / `optimize_rf.py` / `optimize_xgb.py`: Hyperparameter optimization scripts using Optuna.
- `data/`: Directory for input CSV datasets.

## Installation

1. Clone the repository.
2. Install the required dependencies:
```bash
pip install torch pandas scikit-learn mlflow xgboost optuna
```

## Usage

### 1. Data Preprocessing
Generate the vocabulary files from your raw protein data:
```bash
python preprocess.py
```

### 2. Training Models
Train the **Transformer** model:
```bash
python train.py --sample_size 10000 --epochs 20
```

Train the **Random Forest** baseline:
```bash
python train_rf.py --sample_size 5000 --window_size 15
```

### 3. Hyperparameter Optimization
To find the best parameters for the Transformer:
```bash
python optimize.py --trials 20
```

## Experiment Tracking
This project uses **MLflow** for tracking experiments. To view the dashboard and compare runs:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser.

## Methodology
- **Transformer**: Utilizes a sequence-to-sequence architecture with Multi-Head Attention to capture long-range dependencies between amino acids.
- **Window-based RF/XGB**: Extracts local context by sliding a window (default size 15) across the sequence, converting amino acids into one-hot encoded features.
