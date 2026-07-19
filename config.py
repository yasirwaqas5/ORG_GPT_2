from dataclasses import dataclass, field
from typing import Dict, Any
import torch

@dataclass
class ModelConfig:
    """Configuration class for the GPT-2 model architecture.
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length (context window).
        embedding_dimension (int): Dimensionality of the embeddings and hidden states.
        number_of_heads (int): Number of attention heads.
        number_of_layers (int): Number of Transformer blocks.
        dropout (float): Dropout probability for embeddings, attention, and residual connections.
    """
    # Default parameters set to a lightweight "Mini-GPT" model (11.5M params)
    # suitable for local training on Apple Silicon Macs (M1/M2/M3) without system freezes.
    # For a full GPT-2 (124M params), use: vocab=50257, block_size=1024, embed=768, heads=12, layers=12
    vocab_size: int = 50257
    block_size: int = 128
    embedding_dimension: int = 192
    number_of_heads: int = 4
    number_of_layers: int = 4
    dropout: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary representation."""
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embedding_dimension": self.embedding_dimension,
            "number_of_heads": self.number_of_heads,
            "number_of_layers": self.number_of_layers,
            "dropout": self.dropout,
        }


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters and environment settings.
    
    Attributes:
        batch_size (int): Number of sequences per batch.
        learning_rate (float): Peak learning rate for the optimizer.
        weight_decay (float): L2 regularization/weight decay factor.
        max_iterations (int): Total number of training iterations/steps.
        warmup_steps (int): Number of steps to linearly warm up the learning rate.
        evaluation_interval (int): How often (in steps) to run validation evaluation.
        gradient_clip (float): Maximum norm for gradient clipping.
        device (str): Device to run training on ('cpu', 'cuda', 'mps').
        seed (int): Seed for reproducibility.
        dataset_raw_path (str): Path to raw input text dataset.
        dataset_train_path (str): Path to save/load processed training tokens.
        dataset_val_path (str): Path to save/load processed validation tokens.
        sp_model_prefix (str): Prefix path for saving SentencePiece model and vocab.
        checkpoint_dir (str): Directory where training checkpoints are saved.
        output_dir (str): Directory where generated texts or outputs are saved.
        log_dir (str): Directory where logging files are written.
    """
    # Default parameters set to be memory-efficient.
    # To simulate larger batch sizes, use gradient_accumulation_steps.
    batch_size: int = 16
    learning_rate: float = 4e-4
    weight_decay: float = 0.2
    max_iterations: int = 5000
    warmup_steps: int = 200
    evaluation_interval: int = 500
    gradient_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed: int = 1337
    
    # Memory Optimizations
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 4  # Accumulate grads over N steps (effective batch size = batch_size * steps)
    
    # Dataset and Output Paths
    dataset_raw_path: str = "data/raw/input.txt"
    dataset_train_path: str = "data/processed/train.pt"
    dataset_val_path: str = "data/processed/val.pt"
    sp_model_prefix: str = "data/processed/sp"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary representation."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_iterations": self.max_iterations,
            "warmup_steps": self.warmup_steps,
            "evaluation_interval": self.evaluation_interval,
            "gradient_clip": self.gradient_clip,
            "device": self.device,
            "seed": self.seed,
            "dataset_raw_path": self.dataset_raw_path,
            "dataset_train_path": self.dataset_train_path,
            "dataset_val_path": self.dataset_val_path,
            "sp_model_prefix": self.sp_model_prefix,
            "checkpoint_dir": self.checkpoint_dir,
            "output_dir": self.output_dir,
            "log_dir": self.log_dir,
            "use_amp": self.use_amp,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
