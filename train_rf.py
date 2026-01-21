import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_utils import prepare_data
import mlflow
import os
import argparse
import pickle

def extract_window_features(df, window_size=15):
    """
    Extract window-based features for each residue.
    """
    half_window = window_size // 2
    X = []
    y = []
    
    # Get all amino acids for one-hot encoding consistency
    all_aas = sorted(list(set("".join(df['seq']))))
    aa_to_idx = {aa: i for i, aa in enumerate(all_aas)}
    n_aas = len(all_aas)
    
    print(f"Extracting features with window size {window_size}...")
    
    for _, row in df.iterrows():
        seq = row['seq']
        sst3 = row['sst3']
        
        # Pad sequence and labels
        padded_seq = "Z" * half_window + seq + "Z" * half_window
        
        for i in range(len(seq)):
            window = padded_seq[i : i + window_size]
            
            # One-hot encode window
            features = np.zeros((window_size, n_aas + 1)) # +1 for padding token 'Z'
            for j, char in enumerate(window):
                if char == 'Z':
                    features[j, n_aas] = 1
                else:
                    features[j, aa_to_idx.get(char, 0)] = 1
            
            X.append(features.flatten())
            y.append(sst3[i])
            
    return np.array(X), np.array(y), all_aas

def train_rf(args, is_nested=False):
    mlflow.set_experiment("Protein_SST_Prediction_RF")
    
    with mlflow.start_run(nested=is_nested):
        mlflow.log_params(vars(args))
        
        train_df, test_df, _, _, _ = prepare_data(args.csv_path, sample_size=args.sample_size)
        
        X_train, y_train, all_aas = extract_window_features(train_df, window_size=args.window_size)
        X_test, y_test, _ = extract_window_features(test_df, window_size=args.window_size)
        
        print(f"X_train shape: {X_train.shape}")
        
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print("Training Random Forest...")
        rf.fit(X_train, y_train)
        
        print("Evaluating...")
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"SST3 Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        mlflow.log_metric("sst3_accuracy", acc)
        
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(rf, f)
        mlflow.log_artifact("rf_model.pkl")
        
        return acc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    
    args = parser.parse_args()
    train_rf(args)
