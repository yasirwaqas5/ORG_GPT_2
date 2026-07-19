import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import TrainingConfig
from dataset import GPTDataset, get_dataloader

class Trainer:
    """A clean and simple PyTorch trainer for the GPT-2 model."""
    def __init__(self, model: nn.Module, config: TrainingConfig, train_data: torch.Tensor, val_data: torch.Tensor, logger=None):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.logger = logger
        self.device = config.device
        
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        self.best_val_loss = float('inf')
        self.step = 0
        self.patience_counter = 0
        self.patience_threshold = 5  # Stop training if val loss doesn't improve for 5 consecutive evals

        # Initialize PyTorch datasets and dataloaders
        self.train_dataset = GPTDataset(self.train_data, self.model.config.block_size)
        self.val_dataset = GPTDataset(self.val_data, self.model.config.block_size)
        self.train_loader = get_dataloader(self.train_dataset, self.config.batch_size, shuffle=True)
        self.val_loader = get_dataloader(self.val_dataset, self.config.batch_size, shuffle=False)

        # Autocast setup for memory-efficient training
        import contextlib
        device_type = 'cuda' if 'cuda' in self.device else ('cpu' if 'cpu' in self.device else 'mps')
        amp_enabled = getattr(self.config, 'use_amp', True)
        if amp_enabled:
            if device_type == 'cuda':
                self.autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
            elif device_type == 'mps':
                try:
                    # Test if MPS autocast is supported in this PyTorch version
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        pass
                    self.autocast_ctx = torch.autocast(device_type='mps', dtype=torch.float16)
                except Exception:
                    self.autocast_ctx = contextlib.nullcontext()
            elif device_type == 'cpu':
                self.autocast_ctx = torch.autocast(device_type='cpu', dtype=torch.bfloat16)
            else:
                self.autocast_ctx = contextlib.nullcontext()
        else:
            self.autocast_ctx = contextlib.nullcontext()

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

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate stable loss values for training and validation splits."""
        self.model.eval()
        results = {}
        for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
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

    def train(self):
        self.model.train()
        t0 = time.time()
        
        train_iter = iter(self.train_loader)
        grad_accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        for step in range(self.step, self.config.max_iterations):
            self.step = step
            lr = self.get_learning_rate(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            
            # Gradient accumulation sub-steps
            for micro_step in range(grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                x, y = x.to(self.device), y.to(self.device)
                
                with self.autocast_ctx:
                    logits, loss = self.model(x, y)
                    loss = loss / grad_accum_steps
                    
                loss_accum += loss.item()
                loss.backward()
            
            if self.config.gradient_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            else:
                grad_norm = 0.0
                
            self.optimizer.step()
            
            # Print log messages
            if step % 10 == 0 or step == self.config.max_iterations - 1:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                # Account for gradient accumulation steps in token speed calculation
                tokens_per_sec = (self.config.batch_size * self.model.config.block_size * grad_accum_steps * 10) / dt if step > 0 else 0.0
                
                loss_val = loss_accum
                perplexity = math.exp(loss_val) if loss_val < 20 else float('inf')
                
                log_msg = f"step {step:4d} | loss {loss_val:.4f} | perplexity {perplexity:.2f} | lr {lr:.2e} | grad_norm {grad_norm:.4f} | tok/sec {tokens_per_sec:.1f}"
                if self.logger:
                    self.logger.info(log_msg)
                else:
                    print(log_msg)
                    
            # Run evaluation and save checkpoints
            if step > 0 and step % self.config.evaluation_interval == 0:
                losses = self.estimate_loss()
                val_loss = losses['val']
                val_perp = math.exp(val_loss) if val_loss < 20 else float('inf')
                
                eval_msg = f"--- Eval Step {step} --- Train Loss: {losses['train']:.4f} | Val Loss: {val_loss:.4f} | Val Perp: {val_perp:.2f}"
                if self.logger:
                    self.logger.info(eval_msg)
                else:
                    print(eval_msg)
                    
                checkpoint = {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': self.best_val_loss,
                    'config': self.model.config
                }
                
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "checkpoint_last.pt"))
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt"))
                else:
                    self.patience_counter += 1
                    msg = f"Validation loss did not improve (patience {self.patience_counter}/{self.patience_threshold})"
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                        
                    if self.patience_counter >= self.patience_threshold:
                        stop_msg = f"Early stopping triggered at step {step}! Best Val Loss: {self.best_val_loss:.4f}"
                        if self.logger:
                            self.logger.info(stop_msg)
                        else:
                            print(stop_msg)
                        break
                        
                torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{step}.pt"))
