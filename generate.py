import os
import argparse
import math
import torch
import torch.nn as nn
from tokenizer import SentencePieceTokenizer

# --- Layers from layers.py ---

class LayerNorm(nn.Module):
    """Custom LayerNorm implemented from scratch to match PyTorch/GPT-2."""
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class GELU(nn.Module):
    """GELU activation function approximation (standard for GPT-2)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    """The MLP/FeedForward block in GPT-2."""
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """Residual connection helper block."""
    def __init__(self, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer(x)


# --- Embeddings from embeddings.py ---

class TokenEmbedding(nn.Module):
    """Maps token IDs to vectors."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wte(x)


class LearnedPositionEmbedding(nn.Module):
    """Learned absolute positional embeddings (GPT-2 style)."""
    def __init__(self, block_size: int, embed_dim: int):
        super().__init__()
        self.wpe = nn.Embedding(block_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return self.wpe(positions)


class SinusoidalPositionEmbedding(nn.Module):
    """Static sinusoidal positional embeddings."""
    def __init__(self, block_size: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(block_size, embed_dim)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(-1)]


# --- Original generate.py logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_prefix", type=str, default="data/processed/sp", help="Prefix for SentencePiece model file")
    parser.add_argument("--prompt", type=str, default="To be, or not to be, that is the question:", help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=150, help="Number of tokens to generate")
    parser.add_argument("--method", type=str, default="top_k", choices=["greedy", "temperature", "top_k", "top_p"], help="Sampling method")
    parser.add_argument("--temperature", type=float, default=0.8, help="Logits temperature scaling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering constraint")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p filtering constraint")
    parser.add_argument("--out_file", type=str, default="outputs/generated.txt", help="File path to save the generated output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = SentencePieceTokenizer()
    sp_model_file = f"{args.tokenizer_prefix}.model"
    if not os.path.exists(sp_model_file):
        raise FileNotFoundError(f"Tokenizer model not found at {sp_model_file}. Run train.py first to train it.")
    tokenizer.load(sp_model_file)

    if args.checkpoint == "checkpoints/checkpoint_best.pt" and not os.path.exists(args.checkpoint):
        part1 = "checkpoints/checkpoint_best.zip.aa"
        part2 = "checkpoints/checkpoint_best.zip.ab"
        if os.path.exists(part1) and os.path.exists(part2):
            import zipfile
            print("Reassembling model checkpoint from split zip files...")
            os.makedirs("checkpoints", exist_ok=True)
            temp_zip = "checkpoints/temp_checkpoint_best.zip"
            try:
                with open(temp_zip, "wb") as f_out:
                    for part in [part1, part2]:
                        with open(part, "rb") as f_in:
                            f_out.write(f_in.read())
                with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                    zip_ref.extractall(".")
                print("Model checkpoint successfully reassembled.")
            except Exception as e:
                print(f"Error reassembling checkpoint: {e}")
            finally:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")
    
    print(f"Loading model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    from model import GPT2
    
    model_config = checkpoint['config']
    model = GPT2(model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = model.generate(
        idx=x,
        max_new_tokens=args.max_tokens,
        method=args.method,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    output_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated Output:")
    print(output_text)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nSaved generated output to {args.out_file}")

if __name__ == "__main__":
    main()
