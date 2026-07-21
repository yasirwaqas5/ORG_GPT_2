# Training Guide

Comprehensive documentation of the training pipeline, configuration, and workflows.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Optimizer](#optimizer)
- [Learning Rate Scheduler](#learning-rate-scheduler)
- [Gradient Accumulation](#gradient-accumulation)
- [Label Smoothing](#label-smoothing)
- [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)
- [Checkpoint Creation](#checkpoint-creation)
- [Evaluation](#evaluation)
- [Early Stopping](#early-stopping)
- [Resuming Training](#resuming-training)
- [Legacy Checkpoint Handling](#legacy-checkpoint-handling)
- [How Max Iterations Work](#how-max-iterations-work)
- [Command Examples](#command-examples)
- [Expected Training Times](#expected-training-times)
- [CPU Considerations](#cpu-considerations)

---

## Overview

The training pipeline is implemented across three files:
- **`train.py`** — CLI entry point, data preparation, argument parsing
- **`trainer.py`** — `Trainer` class with the training loop, evaluation, and checkpoint management
- **`config.py`** — `ModelConfig` and `TrainingConfig` dataclasses

Training follows this sequence:

1. **Data preparation**: Download raw text → train SentencePiece tokenizer → tokenize and cache
2. **Model initialization**: Create `GPT2` model from configuration (or restore from checkpoint)
3. **Training loop**: Forward/backward pass with gradient accumulation → optimizer step → periodic evaluation and checkpointing
4. **Termination**: Training ends when `max_iterations` is reached or early stopping triggers

---

## Quick Start

```bash
# Train from scratch with defaults
python train.py

# Train with custom iterations and save interval
python train.py --max-iterations 10000 --save-every 250

# Resume from last checkpoint
python train.py --resume checkpoints/checkpoint_last.pt

# Resume from best checkpoint
python train.py --from-best

# Evaluate a checkpoint
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

---

## Training Configuration

All training hyperparameters are defined in `config.py` — class `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Sequences per micro-batch |
| `learning_rate` | 4e-4 | Peak learning rate |
| `weight_decay` | 0.2 | L2 regularization coefficient |
| `max_iterations` | 5,000 | Total training steps |
| `warmup_steps` | 200 | Linear warmup phase length |
| `evaluation_interval` | 500 | Steps between evaluations |
| `gradient_clip` | 1.0 | Maximum gradient norm |
| `device` | auto | `"cuda"` > `"mps"` > `"cpu"` |
| `seed` | 1337 | Random seed for reproducibility |
| `use_amp` | True | Enable automatic mixed precision |
| `gradient_accumulation_steps` | 4 | Micro-batches per optimizer step |
| `overwrite_step_checkpoints` | False | Protect historical step checkpoints |

**Effective batch size**: `batch_size × gradient_accumulation_steps = 16 × 4 = 64 sequences per optimizer step`

**Data paths** (also in `TrainingConfig`):

| Parameter | Default |
|-----------|---------|
| `dataset_raw_path` | `data/raw/input.txt` |
| `dataset_train_path` | `data/processed/train.pt` |
| `dataset_val_path` | `data/processed/val.pt` |
| `sp_model_prefix` | `data/processed/sp` |
| `checkpoint_dir` | `checkpoints` |
| `output_dir` | `outputs` |
| `log_dir` | `logs` |

---

## Optimizer

**Algorithm**: AdamW (decoupled weight decay)

```
Configured in: trainer.py, Trainer.__init__
```

| Parameter | Value |
|-----------|-------|
| Learning rate | `config.learning_rate` (4e-4) |
| Weight decay | `config.weight_decay` (0.2) |
| β₁, β₂ | PyTorch defaults (0.9, 0.999) |
| ε | PyTorch default (1e-8) |

AdamW applies weight decay separately from the gradient update (unlike L2 regularization in standard Adam), which provides better generalization for transformer models.

---

## Learning Rate Scheduler

**Type**: Linear warmup + cosine decay

**File**: `trainer.py` — `Trainer.get_learning_rate()` (lines 77–87)

```
Phase 1: Linear Warmup (steps 0 to warmup_steps)
    lr = learning_rate × (step / warmup_steps)

Phase 2: Cosine Decay (steps warmup_steps to max_iterations)
    lr = min_lr + 0.5 × (learning_rate - min_lr) × (1 + cos(π × progress))
    where progress = (step - warmup) / (max_iter - warmup)
    and min_lr = 0.1 × learning_rate

Phase 3: Constant (steps > max_iterations)
    lr = min_lr = 0.1 × learning_rate
```

```
Learning Rate Schedule (default config):
    │
4e-4│          ╱──╲
    │        ╱      ╲
    │      ╱          ╲
    │    ╱              ╲
    │  ╱                  ╲──────────
4e-5│╱                              
    └────────────────────────────────
    0   200              5000
       warmup          max_iter
```

The minimum learning rate is 10% of the peak learning rate, ensuring the model continues to learn at a slow rate even at the end of training.

---

## Gradient Accumulation

**File**: `trainer.py` — `Trainer.train()` (lines 200–259)

Gradient accumulation simulates a larger batch size without requiring more memory:

```
for each training step:
    optimizer.zero_grad()
    for i in range(gradient_accumulation_steps):
        x, y = next(dataloader)
        logits, loss = model(x, y)
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
    clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
```

With `batch_size=16` and `gradient_accumulation_steps=4`, each optimizer step processes `16 × 4 = 64` sequences, but only 16 sequences are in memory at any time.

> **Note**: Loss is divided by `gradient_accumulation_steps` so that the total gradient magnitude is equivalent to processing the full effective batch at once.

---

## Label Smoothing

**File**: `model.py` — `GPT2.forward()` (line 47)

Cross-entropy loss uses `label_smoothing=0.05`:

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1),
                       label_smoothing=0.05)
```

Label smoothing prevents the model from becoming overconfident by redistributing 5% of the probability mass from the ground-truth label to all other tokens uniformly. This acts as a regularizer and can improve generalization.

---

## Automatic Mixed Precision (AMP)

**File**: `trainer.py` — `Trainer.__init__()` (lines 46–65)

AMP is configured per device:

| Device | AMP dtype | Notes |
|--------|-----------|-------|
| CUDA | `float16` | Standard mixed precision |
| MPS | `float16` | With fallback to `nullcontext` on failure |
| CPU | `bfloat16` | Better numerical stability than float16 on CPU |
| Any (if `use_amp=False`) | disabled | Full `float32` training |

AMP keeps model weights in `float32` but performs forward/backward passes in lower precision, reducing memory usage and increasing throughput on supported hardware.

> **Note**: `GradScaler` is not used in the current implementation. The training loop relies on `torch.autocast` only.

---

## Checkpoint Creation

**File**: `trainer.py` — `Trainer._save_evaluation_checkpoints()` (lines 147–176)

At every `evaluation_interval` steps, the trainer saves up to three checkpoint files:

### 1. `checkpoint_last.pt`
Always saved at every evaluation. Represents the most recent model state.

### 2. `checkpoint_best.pt`
Saved only when the current validation loss is lower than all previous validation losses (`best_val_loss`).

### 3. `checkpoint_step_{N}.pt`
A historical snapshot at step N. By default, these are **overwrite-protected** — attempting to save over an existing step checkpoint raises `FileExistsError`. This can be overridden with `--overwrite-checkpoints`.

### Checkpoint Metadata

Each checkpoint file contains:

| Key | Type | Description |
|-----|------|-------------|
| `model_state` | dict | Full `model.state_dict()` |
| `optimizer_state` | dict | Full `optimizer.state_dict()` |
| `step` | int | Completed training step number |
| `best_val_loss` | float | Best validation loss seen so far |
| `validation_loss` | float | Validation loss at this step |
| `train_loss` | float | Training loss at this step |
| `learning_rate` | float | Learning rate at this step |
| `patience_counter` | int | Steps without improvement |
| `gradient_accumulation_steps` | int | Accumulation setting |
| `config` | ModelConfig | Model architecture configuration |
| `training_config` | dict | Serialized training configuration |
| `python_rng_state` | tuple | Python `random` module state |
| `numpy_rng_state` | dict | NumPy RNG state |
| `torch_rng_state` | Tensor | PyTorch CPU RNG state |
| `train_generator_state` | Tensor | DataLoader generator state |
| `cuda_rng_state_all` | list | CUDA RNG states (if available) |

### Atomic Saving

**File**: `utils.py` — `save_checkpoint()` (lines 35–45)

Checkpoints are written atomically to prevent corruption:

1. Create a temporary file in the checkpoint directory via `tempfile.mkstemp`
2. Write the checkpoint data with `torch.save`
3. Flush the file buffer and call `os.fsync` to ensure data reaches disk
4. Use `os.replace` to atomically rename the temp file to the final path

If a crash occurs during step 2 or 3, the original checkpoint file is untouched. The `os.replace` in step 4 is an atomic operation on all modern operating systems.

---

## Evaluation

**File**: `trainer.py` — `Trainer.estimate_loss()` (lines 178–198)

Evaluation runs at every `evaluation_interval` steps (default: 500). It computes the average loss over **30 batches** from both the training and validation DataLoaders:

```
eval_results = {
    "train": average_loss_over_30_train_batches,
    "val":   average_loss_over_30_val_batches
}
```

The model is set to `eval()` mode during evaluation (disabling dropout) and restored to `train()` mode afterward.

### Standalone Evaluation

You can evaluate any checkpoint without training:

```bash
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

This loads the checkpoint, runs `estimate_loss()`, and prints the results.

---

## Early Stopping

**File**: `trainer.py` — `Trainer._save_evaluation_checkpoints()` and `Trainer.train()`

The trainer implements patience-based early stopping:

- **Patience threshold**: 5 evaluations without improvement
- **Improvement**: Any decrease in validation loss
- **Counter reset**: When validation loss improves, `patience_counter` resets to 0
- **Trigger**: When `patience_counter >= patience_threshold`, training stops

With the default `evaluation_interval=500` and `patience_threshold=5`, training stops if validation loss hasn't improved for 2,500 consecutive steps.

---

## Resuming Training

### From a specific checkpoint

```bash
python train.py --resume checkpoints/checkpoint_step_1500.pt
```

### From the best checkpoint

```bash
python train.py --from-best
```

This is equivalent to `--resume checkpoints/checkpoint_best.pt`.

### What happens on resume

1. The checkpoint is loaded from disk
2. `ModelConfig` is extracted from the checkpoint (ensuring architecture matches)
3. `TrainingConfig` is reconstructed from the checkpoint, with the current host's device substituted
4. Model weights and optimizer state are restored
5. Training step is set to `checkpoint_step + 1` (i.e., the next step to execute)
6. `best_val_loss`, `patience_counter`, and RNG states are restored
7. Training continues from where it left off

### Safe resume practices

- **Do not modify `config.py`** between saving and resuming — the model architecture must match the checkpoint.
- **You can change `max_iterations`** via `--max-iterations` to extend training.
- **You can change `evaluation_interval`** via `--save-every`.
- **You can change `checkpoint_dir`** via `--save-dir`.
- **Historical step checkpoints are preserved** — the trainer will not overwrite `checkpoint_step_500.pt` when resuming from step 500 (it starts at step 501).

---

## Legacy Checkpoint Handling

Checkpoints created before the RNG state preservation feature was added are called "legacy checkpoints." They contain model weights, optimizer state, and training metrics but lack:

- `python_rng_state`
- `numpy_rng_state`
- `torch_rng_state`
- `train_generator_state`
- `cuda_rng_state_all`

When a legacy checkpoint is loaded, the trainer:

1. Logs a warning: `"RNG/DataLoader state is unavailable in this legacy checkpoint."`
2. Resumes training normally, but data ordering and dropout patterns may differ from the original run
3. Saves new checkpoints with full RNG state, so future resumes will be fully reproducible

---

## How Max Iterations Work

The `max_iterations` parameter controls the total number of optimizer steps:

- **Step counting** starts at 0 and increments by 1 after each optimizer step
- **On resume**, the step counter is set to `checkpoint_step + 1`
- **Training ends** when `step >= max_iterations`
- **Extending training**: Use `--max-iterations` to set a higher value than the checkpoint's original config

**Example**: A checkpoint saved at step 1500 with `max_iterations=5000`:
- Resume with no override: trains steps 1501 through 5000
- Resume with `--max-iterations 10000`: trains steps 1501 through 10000
- Resume with `--max-iterations 1500`: error — no remaining steps to train

The trainer validates this at startup:

```
If step >= max_iterations:
    → Error: "Checkpoint is at step {step}, but max_iterations is {max_iterations}."
```

---

## Command Examples

### Basic training

```bash
# Default: 5000 steps, eval every 500, save to checkpoints/
python train.py
```

### Custom training run

```bash
# 10K steps, eval every 250, custom save directory
python train.py --max-iterations 10000 --save-every 250 --save-dir my_checkpoints
```

### Resume and extend

```bash
# Continue from step 1500, train up to step 8000
python train.py --resume checkpoints/checkpoint_step_1500.pt --max-iterations 8000
```

### Evaluation only

```bash
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

**Example output:**
```
Evaluation complete for checkpoints/checkpoint_step_1500.pt:
  train_loss=2.6217, val_loss=4.4008
```

### Allow checkpoint overwriting

```bash
python train.py --overwrite-checkpoints
```

---

## Expected Training Times

> Based on actual training logs from this project.

### Medium Model (11.13M parameters)

| Device | Throughput | Time per 10 steps | Est. 5000 steps |
|--------|-----------|-------------------|-----------------|
| CPU | ~47–49 tok/sec | ~57 minutes | ~4.7 days |
| CUDA GPU (est.) | ~5000+ tok/sec | < 1 minute | < 1 hour |

### Mini Model (2.00M parameters, default config)

The mini model trains significantly faster due to its smaller size. Expect roughly 5× faster throughput than the medium model on the same hardware.

### Throughput Metrics

Training logs report tokens/second (`tok/sec`), which is calculated as:

```
tok/sec = (batch_size × block_size × gradient_accumulation_steps) / elapsed_seconds
```

---

## CPU Considerations

Training on CPU is functional but slow. Key points:

- **AMP dtype**: CPU uses `bfloat16` (better numerical stability than `float16`)
- **Throughput**: ~47–49 tokens/second for the 11.13M model
- **Memory**: CPU training uses more memory than GPU due to lack of memory management optimizations
- **Recommendation**: Use the Mini model configuration (2M params) for CPU experimentation; use GPU for the Medium model and above

### Tips for CPU training

1. **Reduce model size**: Use the default Mini config (`embed_dim=192, heads=4, layers=4, block_size=128`)
2. **Reduce batch size**: Lower `batch_size` to 8 or 4 if memory is limited
3. **Increase save frequency**: Use `--save-every 100` to get checkpoints more often
4. **Reduce max iterations**: Start with `--max-iterations 500` to verify everything works
5. **Disable AMP**: Set `use_amp=False` in `TrainingConfig` if you experience numerical issues
