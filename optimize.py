import optuna
import mlflow
import os
import argparse
from train import train

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def objective(trial):
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    
    args = Args(
        csv_path=csv_path,
        sample_size=5000,
        batch_size=32,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 2,
        dropout=0.1,
        lr=lr,
        epochs=5
    )
    
    accuracy = train(args, is_nested=True)
    
    return accuracy

def run_optimization(n_trials=20):
    mlflow.set_experiment("Protein_SST_Prediction")
    
    # Start a parent run for the entire study
    with mlflow.start_run(run_name="Optuna_Study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        # Log best params to the parent run
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_accuracy", trial.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    
    run_optimization(n_trials=args.trials)
