import torch
import torch.nn as nn
from attention import MultiHeadAttention
from generate import LayerNorm, FeedForward

class TransformerBlock(nn.Module):
    """A single Pre-LN Transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
