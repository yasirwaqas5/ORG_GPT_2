import argparse
import os
import urllib.request
from dataclasses import fields
from typing import Optional

import torch

from config import ModelConfig, TrainingConfig
from tokenizer import SentencePieceTokenizer
from model import GPT2
from trainer import Trainer
from utils import seed_everything, print_model_summary, create_logger


def download_data(raw_path: str):
    """Download Tiny Shakespeare if it is not already available locally."""
    if not os.path.exists(raw_path):
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Tiny Shakespeare dataset from: {url}")
        urllib.request.urlretrieve(url, raw_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train, resume, or evaluate the local GPT-2 model.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", type=str, help="Resume from a checkpoint file.")
    action.add_argument("--from-best", action="store_true", help="Resume from checkpoint_best.pt in the save directory.")
    action.add_argument("--evaluate", type=str, help="Evaluate an existing checkpoint without training.")
    parser.add_argument("--save-every", type=int, help="Evaluate and save checkpoints every N optimizer steps.")
    parser.add_argument("--save-dir", type=str, help="Directory for new checkpoint files.")
    parser.add_argument("--max-iterations", type=int, help="Total optimizer-step limit, including resumed steps.")
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Allow overwriting an existing checkpoint_step_<N>.pt file.",
    )
    return parser.parse_args()


def load_checkpoint(path: str, device: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def training_config_from_checkpoint(checkpoint: dict, fallback: TrainingConfig) -> tuple[TrainingConfig, bool]:
    """Use saved training settings when available, while keeping this host's device."""
    saved = checkpoint.get("training_config")
    if not isinstance(saved, dict):
        return fallback, False
    valid_fields = {field.name for field in fields(TrainingConfig)}
    values = {key: value for key, value in saved.items() if key in valid_fields}
    values["device"] = fallback.device
    return TrainingConfig(**values), True


def prepare_data(config: TrainingConfig, logger):
    download_data(config.dataset_raw_path)

    tokenizer = SentencePieceTokenizer()
    sp_model_file = f"{config.sp_model_prefix}.model"
    if not os.path.exists(sp_model_file):
        logger.info("Training SentencePiece tokenizer...")
        tokenizer.train(config.dataset_raw_path, config.sp_model_prefix, vocab_size=1000)
    else:
        tokenizer.load(sp_model_file)

    if not os.path.exists(config.dataset_train_path) or not os.path.exists(config.dataset_val_path):
        logger.info("Tokenizing dataset text...")
        with open(config.dataset_raw_path, "r", encoding="utf-8") as input_file:
            token_ids = tokenizer.encode(input_file.read())
        split_index = int(0.9 * len(token_ids))
        train_ids = torch.tensor(token_ids[:split_index], dtype=torch.long)
        val_ids = torch.tensor(token_ids[split_index:], dtype=torch.long)
        os.makedirs(os.path.dirname(config.dataset_train_path), exist_ok=True)
        torch.save(train_ids, config.dataset_train_path)
        torch.save(val_ids, config.dataset_val_path)
    else:
        logger.info("Loading cached tokenized datasets...")
        train_ids = torch.load(config.dataset_train_path, weights_only=False)
        val_ids = torch.load(config.dataset_val_path, weights_only=False)
    return tokenizer, train_ids, val_ids


def apply_cli_overrides(config: TrainingConfig, args, default_save_dir: Optional[str] = None) -> None:
    if args.save_every is not None:
        if args.save_every <= 0:
            raise ValueError("--save-every must be greater than zero.")
        config.evaluation_interval = args.save_every
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            raise ValueError("--max-iterations must be greater than zero.")
        config.max_iterations = args.max_iterations
    if args.save_dir:
        config.checkpoint_dir = args.save_dir
    elif default_save_dir:
        config.checkpoint_dir = default_save_dir
    config.overwrite_step_checkpoints = args.overwrite_checkpoints


def evaluate_checkpoint(checkpoint_path: str, config: TrainingConfig, logger) -> None:
    checkpoint = load_checkpoint(checkpoint_path, config.device)
    model_config = checkpoint["config"]
    tokenizer, train_ids, val_ids = prepare_data(config, logger)
    if tokenizer.vocab_size() != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.vocab_size()}) does not match checkpoint vocabulary ({model_config.vocab_size})."
        )
    model = GPT2(model_config)
    trainer = Trainer(model, config, train_ids, val_ids, logger=logger)
    trainer.restore_checkpoint(checkpoint)
    losses = trainer.estimate_loss()
    logger.info(
        f"Evaluation complete for {checkpoint_path}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}"
    )


def main():
    args = parse_args()
    base_config = TrainingConfig()

    if args.from_best:
        best_dir = args.save_dir or base_config.checkpoint_dir
        args.resume = os.path.join(best_dir, "checkpoint_best.pt")

    checkpoint = None
    checkpoint_path = args.resume or args.evaluate
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, base_config.device)
        train_config, has_saved_training_config = training_config_from_checkpoint(checkpoint, base_config)
        if not has_saved_training_config:
            print("Warning: legacy checkpoint has no training_config metadata; using current TrainingConfig values.")
        default_save_dir = os.path.dirname(os.path.abspath(args.resume)) if args.resume else None
    else:
        train_config = base_config
        default_save_dir = None

    apply_cli_overrides(train_config, args, default_save_dir)
    logger = create_logger(train_config.log_dir)
    seed_everything(train_config.seed)

    if args.evaluate:
        evaluate_checkpoint(args.evaluate, train_config, logger)
        return

    tokenizer, train_ids, val_ids = prepare_data(train_config, logger)
    if checkpoint is not None:
        model_config = checkpoint["config"]
        if tokenizer.vocab_size() != model_config.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary ({tokenizer.vocab_size()}) does not match checkpoint vocabulary ({model_config.vocab_size})."
            )
        model = GPT2(model_config)
        logger.info(f"Resuming model from {args.resume}.")
    else:
        model_config = ModelConfig()
        model_config.vocab_size = tokenizer.vocab_size()
        model = GPT2(model_config)
        logger.info("Initializing a new GPT-2 model.")

    print_model_summary(model, logger)
    trainer = Trainer(model, train_config, train_ids, val_ids, logger=logger)
    if checkpoint is not None:
        trainer.restore_checkpoint(checkpoint)
        if trainer.step >= train_config.max_iterations:
            raise ValueError(
                f"Checkpoint resumes at step {trainer.step}, but --max-iterations is {train_config.max_iterations}. "
                "Increase --max-iterations to run additional steps."
            )

    logger.info("Starting training loop...")
    trainer.train()


if __name__ == "__main__":
    main()
