import optuna
import mlflow
import os
import argparse
from train_xgb import train_xgb

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def objective(trial):
    # Suggest hyperparameters
    window_size = trial.suggest_categorical("window_size", [15, 21, 25])
    n_estimators = trial.suggest_int("n_estimators", 100, 200)
    max_depth = trial.suggest_int("max_depth", 6, 15)
    
    # Other fixed parameters
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    
    args = Args(
        csv_path=default_csv,
        sample_size=1000, 
        window_size=window_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        device="cuda"
    )
    
    # Run training as a nested MLflow run
    accuracy = train_xgb(args, is_nested=True)
    
    return accuracy

def run_optimization(n_trials=10):
    mlflow.set_experiment("Protein_SST_Prediction_XGB")
    
    # Start a parent run for the entire study
    with mlflow.start_run(run_name="XGB_RF_Optuna_Study"):
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
