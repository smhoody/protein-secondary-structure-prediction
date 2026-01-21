import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class ProteinTransformer(nn.Module):
    def __init__(self, n_tokens, n_sst8, n_sst3, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(ProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        
        self.sst8_head = nn.Linear(d_model, n_sst8)
        self.sst3_head = nn.Linear(d_model, n_sst3)

    def forward(self, src, src_key_padding_mask=None):
        # src: [batch_size, seq_len]
        # src_key_padding_mask: [batch_size, seq_len] where True means ignore
        
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer expects padding mask where True means padding
        # Our mask is 1 for real, 0 for padding. So we need to invert it.
        # But wait, nn.TransformerEncoderLayer batch_first=True takes src_key_padding_mask
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        sst8_logits = self.sst8_head(output)
        sst3_logits = self.sst3_head(output)
        
        return sst8_logits, sst3_logits
