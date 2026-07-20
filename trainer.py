import os
import time
import math
import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW

from config import TrainingConfig
from dataset import GPTDataset, get_dataloader
from utils import save_checkpoint


class Trainer:
    """Training loop with restart-safe checkpointing for the GPT-2 model."""

    def __init__(self, model: nn.Module, config: TrainingConfig, train_data: torch.Tensor, val_data: torch.Tensor, logger=None):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.logger = logger
        self.device = config.device

        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Checkpoints are written only after a complete optimizer step, so there
        # is never partial gradient-accumulation state to restore.
        self.best_val_loss = float("inf")
        self.step = 0
        self.patience_counter = 0
        self.patience_threshold = 5
        self.last_validation_loss = None

        # A dedicated generator makes shuffled DataLoader order checkpointable.
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(config.seed)

        self.train_dataset = GPTDataset(self.train_data, self.model.config.block_size)
        self.val_dataset = GPTDataset(self.val_data, self.model.config.block_size)
        self.train_loader = get_dataloader(
            self.train_dataset,
            self.config.batch_size,
            shuffle=True,
            generator=self.train_generator,
        )
        self.val_loader = get_dataloader(self.val_dataset, self.config.batch_size, shuffle=False)

        import contextlib
        device_type = "cuda" if "cuda" in self.device else ("cpu" if "cpu" in self.device else "mps")
        amp_enabled = getattr(self.config, "use_amp", True)
        if amp_enabled:
            if device_type == "cuda":
                self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            elif device_type == "mps":
                try:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        pass
                    self.autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
                except Exception:
                    self.autocast_ctx = contextlib.nullcontext()
            elif device_type == "cpu":
                self.autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            else:
                self.autocast_ctx = contextlib.nullcontext()
        else:
            self.autocast_ctx = contextlib.nullcontext()

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def get_learning_rate(self, step: int) -> float:
        """Linear warmup followed by cosine learning rate decay."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / max(1, self.config.warmup_steps)
        if step > self.config.max_iterations:
            return self.config.learning_rate * 0.1

        decay_ratio = (step - self.config.warmup_steps) / max(1, self.config.max_iterations - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        min_lr = self.config.learning_rate * 0.1
        return min_lr + coeff * (self.config.learning_rate - min_lr)

    def _rng_state(self) -> dict:
        state = {
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "train_generator_state": self.train_generator.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, checkpoint: dict) -> bool:
        required = ("python_rng_state", "numpy_rng_state", "torch_rng_state", "train_generator_state")
        if not all(key in checkpoint for key in required):
            return False
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        self.train_generator.set_state(checkpoint["train_generator_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        return True

    def restore_checkpoint(self, checkpoint: dict) -> None:
        """Restore every state needed to continue from the next optimizer step."""
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step = int(checkpoint["step"]) + 1
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self.patience_counter = int(checkpoint.get("patience_counter", 0))
        self.last_validation_loss = checkpoint.get("validation_loss")

        exact_rng_restore = self._restore_rng_state(checkpoint)
        if exact_rng_restore:
            self._log(f"Resumed checkpoint after step {self.step - 1}; next step is {self.step} with restored RNG state.")
        else:
            self._log(
                "Resumed legacy checkpoint after step "
                f"{self.step - 1}; next step is {self.step}. RNG/DataLoader state is unavailable in this legacy checkpoint."
            )

    def _checkpoint_state(self, step: int, train_loss: float, validation_loss: float, learning_rate: float) -> dict:
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
            "best_val_loss": self.best_val_loss,
            "validation_loss": validation_loss,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "patience_counter": self.patience_counter,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "config": self.model.config,
            "training_config": self.config.to_dict(),
        }
        state.update(self._rng_state())
        return state

    def _save_evaluation_checkpoints(self, step: int, train_loss: float, validation_loss: float, learning_rate: float) -> None:
        improved = validation_loss < self.best_val_loss
        if improved:
            self.best_val_loss = validation_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Build after updating best_val_loss so every metadata field describes
        # the checkpoint's actual evaluation result.
        checkpoint = self._checkpoint_state(step, train_loss, validation_loss, learning_rate)
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "checkpoint_last.pt"))
        if improved:
            save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "checkpoint_best.pt"))
            self._log(f"Saved new best checkpoint at step {step}; validation loss {validation_loss:.4f}.")
        else:
            self._log(
                f"Validation loss did not improve (patience {self.patience_counter}/{self.patience_threshold})."
            )

        step_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(step_path) and not self.config.overwrite_step_checkpoints:
            raise FileExistsError(
                f"Refusing to overwrite historical checkpoint: {step_path}. "
                "Use a different --save-dir or --overwrite-checkpoints."
            )
        save_checkpoint(checkpoint, step_path)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate stable loss values for training and validation splits."""
        self.model.eval()
        results = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = []
            loader_iter = iter(loader)
            for _ in range(30):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    x, y = next(loader_iter)
                x, y = x.to(self.device), y.to(self.device)
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            results[split] = sum(losses) / len(losses)
        self.model.train()
        return results

    def train(self) -> None:
        self.model.train()
        t0 = time.time()
        train_iter = iter(self.train_loader)
        grad_accum_steps = self.config.gradient_accumulation_steps

        for step in range(self.step, self.config.max_iterations):
            self.step = step
            learning_rate = self.get_learning_rate(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                x, y = x.to(self.device), y.to(self.device)
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                    loss = loss / grad_accum_steps
                loss_accum += loss.item()
                loss.backward()

            if self.config.gradient_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            else:
                grad_norm = 0.0
            self.optimizer.step()

            if step % 10 == 0 or step == self.config.max_iterations - 1:
                t1 = time.time()
                elapsed = t1 - t0
                t0 = t1
                tokens_per_second = (
                    self.config.batch_size * self.model.config.block_size * grad_accum_steps * 10 / elapsed
                    if step > 0 else 0.0
                )
                perplexity = math.exp(loss_accum) if loss_accum < 20 else float("inf")
                self._log(
                    f"step {step:4d} | loss {loss_accum:.4f} | perplexity {perplexity:.2f} | "
                    f"lr {learning_rate:.2e} | grad_norm {grad_norm:.4f} | tok/sec {tokens_per_second:.1f}"
                )

            if step > 0 and step % self.config.evaluation_interval == 0:
                losses = self.estimate_loss()
                self.last_validation_loss = losses["val"]
                validation_perplexity = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
                self._log(
                    f"--- Eval Step {step} --- Train Loss: {losses['train']:.4f} | "
                    f"Val Loss: {losses['val']:.4f} | Val Perp: {validation_perplexity:.2f}"
                )
                self._save_evaluation_checkpoints(step, losses["train"], losses["val"], learning_rate)

                if self.patience_counter >= self.patience_threshold:
                    self._log(f"Early stopping triggered at step {step}! Best Val Loss: {self.best_val_loss:.4f}")
                    break
