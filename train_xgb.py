import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from data_utils import prepare_data
import mlflow
import os
import argparse
import pickle

def extract_window_features(df, window_size=15):
    """
    Extract window-based features for each residue using vectorized operations.
    """
    half_window = window_size // 2
    
    # Get all amino acids
    all_aas = sorted(list(set("".join(df['seq']))))
    aa_to_idx = {aa: i for i, aa in enumerate(all_aas)}
    n_aas = len(all_aas)
    pad_idx = n_aas
    
    print(f"Extracting features with window size {window_size} (Vectorized)...")
    
    all_X = []
    all_y = []
    
    for _, row in df.iterrows():
        seq = row['seq']
        sst3 = row['sst3']
        
        # Map sequence to indices
        seq_idx = np.array([aa_to_idx.get(aa, 0) for aa in seq])
        
        # Pad with pad_idx
        padded_idx = np.pad(seq_idx, (half_window, half_window), constant_values=pad_idx)
        
        # Create sliding windows of indices
        # This is a cool trick to get windows without copying data
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded_idx, window_size)
        
        # Now one-hot encode the windows: [L, window_size] -> [L, window_size, n_aas + 1]
        # We can do this efficiently using np.eye
        L = len(seq)
        one_hot = np.eye(n_aas + 1)[windows] # [L, window_size, n_aas + 1]
        
        all_X.append(one_hot.reshape(L, -1))
        all_y.extend(list(sst3))
            
    return np.vstack(all_X), np.array(all_y), all_aas

def train_xgb(args, is_nested=False):
    mlflow.set_experiment("Protein_SST_Prediction_XGB")
    
    with mlflow.start_run(nested=is_nested):
        mlflow.log_params(vars(args))
        
        # Load data
        train_df, test_df, _, _, _ = prepare_data(args.csv_path, sample_size=args.sample_size)
        
        # Extract features
        X_train, y_train_raw, all_aas = extract_window_features(train_df, window_size=args.window_size)
        X_test, y_test_raw, _ = extract_window_features(test_df, window_size=args.window_size)
        
        # Encode labels for XGBoost
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"Using GPU device: {args.device}")
        
        # Configure XGBoost for Random Forest mode
        xgb = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=1, # Standard for Random Forest mode
            num_parallel_tree=args.n_estimators, # This makes it act like a Random Forest
            subsample=0.8,
            colsample_bynode=0.8,
            tree_method='hist', # Required for GPU
            device=args.device,
            random_state=42,
            verbosity=1
        )
        
        print("Training XGBoost (Random Forest mode)...")
        xgb.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating...")
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        
        print(f"SST3 Accuracy (XGB RF): {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        mlflow.log_metric("sst3_accuracy", acc)
        
        # Save model and label encoder
        model_data = {
            'model': xgb,
            'label_encoder': le,
            'window_size': args.window_size
        }
        with open("xgb_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        mlflow.log_artifact("xgb_model.pkl")
        
        return acc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    train_xgb(args)
