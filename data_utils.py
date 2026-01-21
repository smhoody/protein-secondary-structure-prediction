import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os

class ProteinVocabulary:
    def __init__(self, chars, is_label=False):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.is_label = is_label
        
        self.itos = {0: self.pad_token}
        if not is_label:
            self.itos[1] = self.unk_token
            start_idx = 2
        else:
            start_idx = 1
            
        for i, char in enumerate(sorted(list(chars))):
            self.itos[i + start_idx] = char
            
        self.stoi = {s: i for i, s in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def encode(self, sequence):
        if self.is_label:
            return [self.stoi[char] for char in sequence]
        return [self.stoi.get(char, self.stoi[self.unk_token]) for char in sequence]

    def decode(self, indices):
        return [self.itos[idx] for idx in indices]

class ProteinDataset(Dataset):
    def __init__(self, df, seq_vocab, sst8_vocab, sst3_vocab, max_len=512):
        self.df = df
        self.seq_vocab = seq_vocab
        self.sst8_vocab = sst8_vocab
        self.sst3_vocab = sst3_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        sst8 = row['sst8']
        sst3 = row['sst3']

        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            sst8 = sst8[:self.max_len]
            sst3 = sst3[:self.max_len]

        seq_encoded = self.seq_vocab.encode(seq)
        sst8_encoded = self.sst8_vocab.encode(sst8)
        sst3_encoded = self.sst3_vocab.encode(sst3)

        padding_len = self.max_len - len(seq_encoded)
        seq_encoded += [self.seq_vocab.stoi[self.seq_vocab.pad_token]] * padding_len
        sst8_encoded += [self.sst8_vocab.stoi[self.sst8_vocab.pad_token]] * padding_len
        sst3_encoded += [self.sst3_vocab.stoi[self.sst3_vocab.pad_token]] * padding_len

        return {
            'seq': torch.tensor(seq_encoded, dtype=torch.long),
            'sst8': torch.tensor(sst8_encoded, dtype=torch.long),
            'sst3': torch.tensor(sst3_encoded, dtype=torch.long),
            'mask': torch.tensor([1] * len(seq) + [0] * padding_len, dtype=torch.bool)
        }

def prepare_data(csv_path, max_len=512, sample_size=None):
    print(f"Loading data from {csv_path}...")
    if not os.path.isabs(csv_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(base_dir, csv_path)
        if os.path.exists(potential_path):
            csv_path = potential_path
            
    df = pd.read_csv(csv_path)
    df = df[df['has_nonstd_aa'] == False].reset_index(drop=True)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    all_seq_chars = set("".join(df['seq'].unique()))
    all_sst8_chars = set("".join(df['sst8'].unique()))
    all_sst3_chars = set("".join(df['sst3'].unique()))

    seq_vocab = ProteinVocabulary(all_seq_chars)
    sst8_vocab = ProteinVocabulary(all_sst8_chars, is_label=True)
    sst3_vocab = ProteinVocabulary(all_sst3_chars, is_label=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df, seq_vocab, sst8_vocab, sst3_vocab
