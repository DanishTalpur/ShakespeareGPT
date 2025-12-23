import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device

# Single self-attention head
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, block_size, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # Causal mask (no future tokens)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        wei = (self.query(x) @ self.key(x).transpose(-2, -1)) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(torch.softmax(wei, dim=-1))
        return wei @ self.value(x)


# Multiple attention heads
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, block_size, head_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Transformer block
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attention
        x = x + self.ff(self.ln2(x))   # MLP
        return x


# Full GPT model
class ShakespeareGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        # Stack transformer blocks
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        T = idx.size(1)
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        return self.head(self.ln(x))
