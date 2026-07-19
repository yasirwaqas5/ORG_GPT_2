import os
import urllib.request
import torch
from config import ModelConfig, TrainingConfig
from tokenizer import SentencePieceTokenizer
from model import GPT2
from trainer import Trainer
from utils import seed_everything, print_model_summary, create_logger

def download_data(raw_path: str):
    """Downloads Tiny Shakespeare if not present on disk."""
    if not os.path.exists(raw_path):
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Tiny Shakespeare dataset from: {url}")
        urllib.request.urlretrieve(url, raw_path)

def main():
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # Create logger and seed environment
    logger = create_logger(train_config.log_dir)
    seed_everything(train_config.seed)

    # 1. Download dataset
    download_data(train_config.dataset_raw_path)

    # 2. Train and load Tokenizer
    tokenizer = SentencePieceTokenizer()
    sp_model_file = f"{train_config.sp_model_prefix}.model"
    if not os.path.exists(sp_model_file):
        logger.info("Training SentencePiece tokenizer...")
        # Use vocab size of 1000 for Shakespeare dataset size
        tokenizer.train(train_config.dataset_raw_path, train_config.sp_model_prefix, vocab_size=1000)
    else:
        tokenizer.load(sp_model_file)

    # Update model vocab size based on trained tokenizer
    model_config.vocab_size = tokenizer.vocab_size()

    # 3. Tokenize dataset and cache splits
    if not os.path.exists(train_config.dataset_train_path) or not os.path.exists(train_config.dataset_val_path):
        logger.info("Tokenizing dataset text...")
        with open(train_config.dataset_raw_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        ids = tokenizer.encode(text)
        n = int(0.9 * len(ids))
        train_ids = torch.tensor(ids[:n], dtype=torch.long)
        val_ids = torch.tensor(ids[n:], dtype=torch.long)

        os.makedirs(os.path.dirname(train_config.dataset_train_path), exist_ok=True)
        torch.save(train_ids, train_config.dataset_train_path)
        torch.save(val_ids, train_config.dataset_val_path)
    else:
        logger.info("Loading cached tokenized datasets...")
        train_ids = torch.load(train_config.dataset_train_path)
        val_ids = torch.load(train_config.dataset_val_path)

    # 4. Initialize model
    logger.info("Initializing GPT-2 model...")
    model = GPT2(model_config)
    print_model_summary(model, logger)

    # 5. Initialize trainer and start training
    logger.info("Starting training loop...")
    trainer = Trainer(model, train_config, train_ids, val_ids, logger=logger)
    trainer.train()

if __name__ == "__main__":
    main()
