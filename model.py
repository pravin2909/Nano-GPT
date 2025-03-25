import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiheadAttention(d_model, h, dropout)
        self.feedforward = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_en = self.positional_encoding(seq_len, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(n)])
        self.norm = LayerNormalization(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def causal_mask(self, size):
        return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def forward(self, x):
        mask = self.causal_mask(x.shape[1]).to(x.device)
        x = self.embedding(x) + self.positional_en[:, :x.shape[1], :].to(x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))

   