import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from data_utils import prepare_data, ProteinDataset, ProteinVocabulary
from model import ProteinTransformer
import pickle
import os
import argparse
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

def train_one_epoch(model, loader, optimizer, criterion_sst8, criterion_sst3, device):
    model.train()
    total_loss = 0
    correct_sst8 = 0
    correct_sst3 = 0
    total_residues = 0

    for batch in loader:
        seq = batch['seq'].to(device)
        sst8 = batch['sst8'].to(device)
        sst3 = batch['sst3'].to(device)
        mask = batch['mask'].to(device)
        
        # Invert mask for Transformer (True means padding)
        padding_mask = ~mask

        optimizer.zero_grad()
        sst8_logits, sst3_logits = model(seq, src_key_padding_mask=padding_mask)

        # Flatten for loss calculation
        # sst8_logits: [batch, seq_len, n_classes] -> [batch * seq_len, n_classes]
        # sst8: [batch, seq_len] -> [batch * seq_len]
        active_sst8 = mask.reshape(-1)
        labels_sst8 = sst8.reshape(-1)[active_sst8]
        preds_sst8 = sst8_logits.reshape(-1, sst8_logits.size(-1))[active_sst8]

        active_sst3 = mask.reshape(-1)
        labels_sst3 = sst3.reshape(-1)[active_sst3]
        preds_sst3 = sst3_logits.reshape(-1, sst3_logits.size(-1))[active_sst3]

        loss_sst8 = criterion_sst8(preds_sst8, labels_sst8)
        loss_sst3 = criterion_sst3(preds_sst3, labels_sst3)
        loss = loss_sst8 + loss_sst3

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        _, predicted_sst8 = torch.max(preds_sst8, 1)
        correct_sst8 += (predicted_sst8 == labels_sst8).sum().item()
        
        _, predicted_sst3 = torch.max(preds_sst3, 1)
        correct_sst3 += (predicted_sst3 == labels_sst3).sum().item()
        
        total_residues += labels_sst8.size(0)

    return total_loss / len(loader), correct_sst8 / total_residues, correct_sst3 / total_residues

def evaluate(model, loader, criterion_sst8, criterion_sst3, device):
    model.eval()
    total_loss = 0
    correct_sst8 = 0
    correct_sst3 = 0
    total_residues = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch['seq'].to(device)
            sst8 = batch['sst8'].to(device)
            sst3 = batch['sst3'].to(device)
            mask = batch['mask'].to(device)
            padding_mask = ~mask

            sst8_logits, sst3_logits = model(seq, src_key_padding_mask=padding_mask)

            active = mask.reshape(-1)
            labels_sst8 = sst8.reshape(-1)[active]
            preds_sst8 = sst8_logits.reshape(-1, sst8_logits.size(-1))[active]
            labels_sst3 = sst3.reshape(-1)[active]
            preds_sst3 = sst3_logits.reshape(-1, sst3_logits.size(-1))[active]

            loss_sst8 = criterion_sst8(preds_sst8, labels_sst8)
            loss_sst3 = criterion_sst3(preds_sst3, labels_sst3)
            loss = loss_sst8 + loss_sst3

            total_loss += loss.item()
            _, predicted_sst8 = torch.max(preds_sst8, 1)
            correct_sst8 += (predicted_sst8 == labels_sst8).sum().item()
            _, predicted_sst3 = torch.max(preds_sst3, 1)
            correct_sst3 += (predicted_sst3 == labels_sst3).sum().item()
            total_residues += labels_sst8.size(0)

    return total_loss / len(loader), correct_sst8 / total_residues, correct_sst3 / total_residues

def train(args, is_nested=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabs
    with open('vocabs.pkl', 'rb') as f:
        vocabs = pickle.load(f)
    
    seq_vocab = vocabs['seq']
    sst8_vocab = vocabs['sst8']
    sst3_vocab = vocabs['sst3']

    train_df, test_df, _, _, _ = prepare_data(args.csv_path, sample_size=args.sample_size)
    
    train_ds = ProteinDataset(train_df, seq_vocab, sst8_vocab, sst3_vocab)
    test_ds = ProteinDataset(test_df, seq_vocab, sst8_vocab, sst3_vocab)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    mlflow.set_experiment("Protein_SST_Prediction")
    
    with mlflow.start_run(nested=is_nested):
        mlflow.log_params(vars(args))
        
        model = ProteinTransformer(
            n_tokens=len(seq_vocab),
            n_sst8=len(sst8_vocab),
            n_sst3=len(sst3_vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        ).to(device)

        criterion_sst8 = nn.CrossEntropyLoss()
        criterion_sst3 = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_acc = 0
        for epoch in range(args.epochs):
            train_loss, train_acc8, train_acc3 = train_one_epoch(
                model, train_loader, optimizer, criterion_sst8, criterion_sst3, device
            )
            test_loss, test_acc8, test_acc3 = evaluate(
                model, test_loader, criterion_sst8, criterion_sst3, device
            )

            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, SST8 Acc: {train_acc8:.4f}, SST3 Acc: {train_acc3:.4f}")
            print(f"Test Loss: {test_loss:.4f}, SST8 Acc: {test_acc8:.4f}, SST3 Acc: {test_acc3:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc8", train_acc8, step=epoch)
            mlflow.log_metric("train_acc3", train_acc3, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc8", test_acc8, step=epoch)
            mlflow.log_metric("test_acc3", test_acc3, step=epoch)

            if test_acc8 > best_acc:
                best_acc = test_acc8
                # Specify pip_requirements to avoid the torch version warning
                # and use keyword arguments to avoid deprecation warnings
                mlflow.pytorch.log_model(
                    model, 
                    artifact_path="model", 
                    pip_requirements=[f"torch=={torch.__version__.split('+')[0]}", "mlflow"]
                )
        
        return best_acc
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=default_csv)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    
    args = parser.parse_args()
    train(args)
