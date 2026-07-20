import os
import random
import logging
import tempfile
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, logger=None) -> None:
    """Print architectural summary and parameter counts."""
    total_params = count_parameters(model)
    summary_str = (
        f"\n========================================\n"
        f"GPT-2 Model Summary\n"
        f"========================================\n"
        f"Embedding Dimension: {model.config.embedding_dimension}\n"
        f"Number of Heads:     {model.config.number_of_heads}\n"
        f"Number of Layers:    {model.config.number_of_layers}\n"
        f"Context Block Size:  {model.config.block_size}\n"
        f"Vocabulary Size:     {model.config.vocab_size}\n"
        f"Total Trainable Params: {total_params:,} ({total_params/1e6:.2f}M)\n"
        f"========================================\n"
    )
    if logger:
        logger.info(summary_str)
    else:
        print(summary_str)


def save_checkpoint(state: dict, path: str) -> None:
    """Atomically save a checkpoint so an interruption cannot leave a partial file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        with open(temporary_path, "wb") as checkpoint_file:
            torch.save(state, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(path: str, device: str = 'cpu') -> dict:
    """Load model and optimizer state dictionaries from disk."""
    return torch.load(path, map_location=device)


def create_logger(log_dir: str) -> logging.Logger:
    """Create a unified logger that prints to console and writes to a log file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("GPT-2")
    logger.setLevel(logging.INFO)
    
    # Check to prevent adding duplicate handlers if method is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Stream Handler (Console)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
    return logger
