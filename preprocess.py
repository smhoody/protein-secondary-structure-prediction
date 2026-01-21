import os
import pickle
from data_utils import prepare_data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Robust path construction
    csv_path = os.path.join(base_dir, "data", "2022-08-03-ss.cleaned.csv")
    
    # We load a sample to build the vocabulary
    train_df, test_df, seq_vocab, sst8_vocab, sst3_vocab = prepare_data(csv_path, sample_size=10000)
    
    # Save vocabs
    with open('vocabs.pkl', 'wb') as f:
        pickle.dump({'seq': seq_vocab, 'sst8': sst8_vocab, 'sst3': sst3_vocab}, f)
    
    print("Vocabularies saved correctly to vocabs.pkl")
    print(f"SST8 labels: {sst8_vocab.stoi}")
    print(f"SST3 labels: {sst3_vocab.stoi}")
