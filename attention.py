import math
import torch
import torch.nn as nn

def create_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    """Creates a 4D causal mask of shape (1, 1, seq_len, seq_len)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention."""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k, v shapes: (B, num_heads, T, head_dim)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class SingleHeadAttention(nn.Module):
    """Reference implementation of a single attention head."""
    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (B, T, embed_dim)
        q = self.q_proj(x).unsqueeze(1) # (B, 1, T, head_dim)
        k = self.k_proj(x).unsqueeze(1) # (B, 1, T, head_dim)
        v = self.v_proj(x).unsqueeze(1) # (B, 1, T, head_dim)
        
        out, _ = self.attn(q, k, v, mask)
        out = out.squeeze(1) # (B, T, head_dim)
        return self.out_proj(out)


class MultiHeadAttention(nn.Module):
    """GPT-2 multi-head self-attention module."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Combined projection for query, key, value
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch, Seq_len, Embed_dim
        
        # Compute Q, K, V projections and split them
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        
        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal mask for autoregressive generation
        mask = create_causal_mask(T, x.device)
        
        # Scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted value combination
        y = torch.matmul(attn_weights, v)
        
        # Transpose and concatenate heads back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Project output back and apply residual dropout
        return self.resid_dropout(self.c_proj(y))
