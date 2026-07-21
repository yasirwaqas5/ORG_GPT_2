# Checkpoints

Complete documentation of the checkpoint system: structure, metadata, save/load mechanics, resume logic, and best practices.

---

## Table of Contents

- [Overview](#overview)
- [Checkpoint Types](#checkpoint-types)
- [Checkpoint Structure](#checkpoint-structure)
- [Saved Metadata](#saved-metadata)
- [Model State](#model-state)
- [Optimizer State](#optimizer-state)
- [Training Configuration](#training-configuration)
- [RNG State Preservation](#rng-state-preservation)
- [Atomic Saving](#atomic-saving)
- [Overwrite Protection](#overwrite-protection)
- [Legacy Checkpoints](#legacy-checkpoints)
- [Current Checkpoint Format](#current-checkpoint-format)
- [How Resume Works](#how-resume-works)
- [How Evaluation Works](#how-evaluation-works)
- [Startup Auto-Detection](#startup-auto-detection)
- [Checkpoint Reassembly](#checkpoint-reassembly)
- [Safe Checkpoint Practices](#safe-checkpoint-practices)
- [Inspecting Checkpoints](#inspecting-checkpoints)

---

## Overview

The checkpoint system provides crash-safe, reproducible training with three checkpoint types, atomic writes, overwrite protection, and full RNG state preservation. It is implemented across:

- **`trainer.py`** — `Trainer._checkpoint_state()`, `_save_evaluation_checkpoints()`, `restore_checkpoint()`
- **`utils.py`** — `save_checkpoint()` (atomic write), `load_checkpoint()`
- **`train.py`** — Resume logic, CLI arguments
- **`app.py`** — Startup checkpoint selection, split-zip reassembly

---

## Checkpoint Types

| File | When Saved | Purpose |
|------|-----------|---------|
| `checkpoint_best.pt` | Only when validation loss improves | The best model found during training |
| `checkpoint_last.pt` | At every evaluation interval | Most recent model state |
| `checkpoint_step_{N}.pt` | At every evaluation interval | Immutable historical snapshot at step N |

### Naming Convention

- **Best**: `checkpoint_best.pt` — single file, overwritten on improvement
- **Last**: `checkpoint_last.pt` — single file, overwritten at every evaluation
- **Step**: `checkpoint_step_{N}.pt` — one file per evaluation, never overwritten (by default)

---

## Checkpoint Structure

Each checkpoint is a Python dictionary saved with `torch.save`. The complete key layout:

```python
{
    # Model weights
    "model_state":       OrderedDict,    # model.state_dict()

    # Optimizer state
    "optimizer_state":   dict,            # optimizer.state_dict()

    # Training progress
    "step":              int,             # Completed step number (0-indexed)
    "best_val_loss":     float,           # Best validation loss seen so far
    "validation_loss":   float,           # Validation loss at this step
    "train_loss":        float,           # Training loss at this step
    "learning_rate":     float,           # Learning rate at time of save
    "patience_counter":  int,             # Evals without improvement

    # Configuration
    "gradient_accumulation_steps": int,
    "config":            ModelConfig,     # Model architecture config (dataclass)
    "training_config":   dict,            # Serialized TrainingConfig

    # RNG states (for exact reproducibility)
    "python_rng_state":       tuple,      # random.getstate()
    "numpy_rng_state":        dict,       # numpy.random.get_state()
    "torch_rng_state":        Tensor,     # torch.random.get_rng_state()
    "train_generator_state":  Tensor,     # DataLoader generator state
    "cuda_rng_state_all":     list,       # torch.cuda.get_rng_state_all() (if CUDA)
}
```

---

## Saved Metadata

### Training Progress Fields

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | The completed step number. On resume, training starts at `step + 1`. |
| `best_val_loss` | `float` | The lowest validation loss observed during the entire training run. Used to decide whether to update `checkpoint_best.pt`. |
| `validation_loss` | `float` | The validation loss at this evaluation step. |
| `train_loss` | `float` | The training loss at this evaluation step. |
| `learning_rate` | `float` | The learning rate value at the time of saving. |
| `patience_counter` | `int` | Number of consecutive evaluations without validation loss improvement. |
| `gradient_accumulation_steps` | `int` | The gradient accumulation setting used during training. |

---

## Model State

The `model_state` key contains the full output of `model.state_dict()` — all learnable parameters of the GPT-2 model.

For the Medium configuration (11.13M params), this includes:

| Parameter Group | Shape | Count |
|----------------|-------|-------|
| `transformer.wte.wte.weight` | (1000, 384) | 384,000 |
| `transformer.wpe.wpe.weight` | (256, 384) | 98,304 |
| `transformer.h.{0-5}.ln_1.weight/bias` | (384,) | 4,608 |
| `transformer.h.{0-5}.attn.c_attn.weight/bias` | (384, 1152) / (1152,) | 2,660,352 |
| `transformer.h.{0-5}.attn.c_proj.weight/bias` | (384, 384) / (384,) | 887,040 |
| `transformer.h.{0-5}.ln_2.weight/bias` | (384,) | 4,608 |
| `transformer.h.{0-5}.mlp.c_fc.weight/bias` | (384, 1536) / (1536,) | 3,548,160 |
| `transformer.h.{0-5}.mlp.c_proj.weight/bias` | (1536, 384) / (384,) | 3,548,160 |
| `transformer.ln_f.weight/bias` | (384,) | 768 |
| `lm_head.weight` | *(tied to wte)* | 0 (shared) |

> **Note**: `lm_head.weight` is weight-tied to `transformer.wte.wte.weight`, so it does not occupy additional storage in the state dict.

---

## Optimizer State

The `optimizer_state` key contains the full AdamW optimizer state, including:

- Per-parameter momentum (`exp_avg`) and squared momentum (`exp_avg_sq`) buffers
- Step counts for each parameter group
- Learning rate, weight decay, and beta values

This state is essential for seamless training resumption — without it, the optimizer would restart from scratch, causing a sudden jump in training dynamics.

---

## Training Configuration

Two configuration objects are saved:

### `config` — ModelConfig (dataclass)

Stored as a Python dataclass instance. Contains:
- `vocab_size`, `block_size`, `embedding_dimension`, `number_of_heads`, `number_of_layers`, `dropout`

This is used to reconstruct the model architecture on load.

### `training_config` — TrainingConfig (dict)

Stored as a plain dictionary via `TrainingConfig.to_dict()`. Contains all training hyperparameters. On resume, `TrainingConfig` is reconstructed from this dict, with the `device` field overridden to match the current host.

---

## RNG State Preservation

For exact reproducibility of training results, all random number generator states are captured:

| Key | Source | Purpose |
|-----|--------|---------|
| `python_rng_state` | `random.getstate()` | Python's built-in random module |
| `numpy_rng_state` | `numpy.random.get_state()` | NumPy random state |
| `torch_rng_state` | `torch.random.get_rng_state()` | PyTorch CPU RNG |
| `train_generator_state` | DataLoader's `torch.Generator` | Ensures identical batch ordering |
| `cuda_rng_state_all` | `torch.cuda.get_rng_state_all()` | All CUDA device RNG states (GPU only) |

When these states are restored, training produces **bit-exact identical** results to an uninterrupted run from the same starting point.

---

## Atomic Saving

**File**: `utils.py` — `save_checkpoint()`

Checkpoints are saved atomically to prevent corruption from crashes, power failures, or interrupted processes:

```
Step 1: Create temp file in the same directory
        tempfile.mkstemp(dir=checkpoint_dir)

Step 2: Write checkpoint data
        torch.save(state, temp_file)

Step 3: Flush to disk
        file.flush()
        os.fsync(file_descriptor)

Step 4: Atomic rename
        os.replace(temp_path, final_path)
```

**Why this matters**: If a crash occurs during Step 2 or 3, only the temporary file is corrupted — the original checkpoint remains intact. `os.replace` in Step 4 is guaranteed to be atomic by the operating system, so the checkpoint file is either the old version or the new version, never a partial write.

---

## Overwrite Protection

**File**: `trainer.py` — `_save_evaluation_checkpoints()` (lines 171–175)

By default, step checkpoints (`checkpoint_step_{N}.pt`) are **overwrite-protected**:

```python
if os.path.exists(step_path) and not self.config.overwrite_step_checkpoints:
    raise FileExistsError(f"{step_path} already exists")
```

This prevents accidental data loss when:
- Resuming training from an earlier checkpoint
- Running multiple training experiments
- A bug causes the same step to be evaluated twice

To disable protection, use `--overwrite-checkpoints` on the CLI.

> **Note**: `checkpoint_best.pt` and `checkpoint_last.pt` are always overwritten — they represent the current state, not historical records.

---

## Legacy Checkpoints

Checkpoints created before the RNG state feature was added lack the following keys:
- `python_rng_state`
- `numpy_rng_state`
- `torch_rng_state`
- `train_generator_state`
- `cuda_rng_state_all`

### Detection

The `Trainer._restore_rng_state()` method checks for the presence of `torch_rng_state` in the checkpoint. If missing, it returns `False` and the trainer logs:

```
Resumed legacy checkpoint after step N; next step is N+1.
RNG/DataLoader state is unavailable in this legacy checkpoint.
```

### Behavior

- Training resumes correctly — model weights and optimizer state are fully restored
- Batch ordering and dropout patterns may differ from the original run
- New checkpoints saved during the resumed run will include full RNG state
- No data loss or training instability occurs

### Existing Legacy Checkpoints

The checkpoints in this repository (`checkpoint_step_500.pt`, `checkpoint_step_1000.pt`, `checkpoint_step_1500.pt`) are legacy checkpoints from the 11.13M parameter model training run.

---

## Current Checkpoint Format

The current format includes all fields listed in the [Checkpoint Structure](#checkpoint-structure) section. Any checkpoint saved by the current codebase will include full RNG state and be fully reproducible on resume.

---

## How Resume Works

### Step-by-step process

1. **Load checkpoint from disk**:
   ```python
   checkpoint = torch.load(path, map_location=device, weights_only=False)
   ```

2. **Extract model config** — reconstruct the model architecture from `checkpoint["config"]`

3. **Extract training config** — reconstruct from `checkpoint["training_config"]`, overriding the `device` field with the current host's device

4. **Create model and trainer** — using the extracted configs

5. **Restore state** (`Trainer.restore_checkpoint()`):
   - Load model weights: `model.load_state_dict(checkpoint["model_state"], strict=True)`
   - Load optimizer state: `optimizer.load_state_dict(checkpoint["optimizer_state"])`
   - Set step: `self.step = checkpoint["step"] + 1`
   - Restore `best_val_loss`, `patience_counter`, `last_validation_loss`
   - Restore all RNG states (if available)

6. **Validate** — ensure `step < max_iterations`

7. **Resume training** — the training loop starts from the restored step

### What can be changed on resume

| Parameter | Changeable? | How |
|-----------|------------|-----|
| `max_iterations` | Yes | `--max-iterations` |
| `evaluation_interval` | Yes | `--save-every` |
| `checkpoint_dir` | Yes | `--save-dir` |
| `overwrite_step_checkpoints` | Yes | `--overwrite-checkpoints` |
| `device` | Automatic | Auto-detected from hardware |
| Model architecture | No | Must match checkpoint |
| `batch_size` | No | Stored in checkpoint |
| `learning_rate` | No | Restored from LR schedule |

---

## How Evaluation Works

### During training

Every `evaluation_interval` steps, the trainer:
1. Puts the model in `eval()` mode
2. Computes average loss over 30 batches from train and val DataLoaders
3. Logs the results: `Train Loss: X.XXXX | Val Loss: X.XXXX | Val Perp: X.XX`
4. Saves checkpoints (best, last, step)
5. Updates patience counter
6. Returns model to `train()` mode

### Standalone evaluation

```bash
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

This runs `estimate_loss()` and exits without training. Output:
```
Evaluation complete for checkpoints/checkpoint_step_1500.pt:
  train_loss=2.6217, val_loss=4.4008
```

---

## Startup Auto-Detection

**File**: `app.py` — `find_startup_checkpoint()`

The FastAPI server and CLI automatically find the best checkpoint to load:

1. Check for `checkpoint_best.pt` — use if exists
2. Otherwise, scan for `checkpoint_step_*.pt` files
3. Sort by step number (extracted from filename)
4. Use the highest-step checkpoint
5. If no checkpoints exist, start without a model (API returns `"status": "not_trained"`)

---

## Checkpoint Reassembly

For GitHub distribution (100 MB file size limit), checkpoints can be split:

```bash
# Split (done by the developer)
zip checkpoint_best.zip checkpoint_best.pt
split -b 50m checkpoint_best.zip checkpoint_best.zip.

# Reassemble (done automatically by app.py and generate.py)
cat checkpoint_best.zip.aa checkpoint_best.zip.ab > checkpoint_best.zip
unzip checkpoint_best.zip
```

The `.gitignore` excludes `checkpoints/*.pt` but includes `checkpoint_best.zip.aa` and `.ab`.

Both `app.py` and `generate.py` call a reassembly function on startup that:
1. Checks if `checkpoint_best.pt` already exists
2. If not, looks for `checkpoint_best.zip.aa` + `.ab`
3. Concatenates them into `checkpoint_best.zip`
4. Extracts `checkpoint_best.pt`
5. Removes the intermediate zip

---

## Safe Checkpoint Practices

### Do

- **Keep historical step checkpoints** — they let you resume from any point
- **Use `--from-best`** when resuming to continue from the strongest model
- **Back up `checkpoint_best.pt`** before extended training runs
- **Verify checkpoints** with `python train.py --evaluate <path>` before relying on them
- **Use the default overwrite protection** — it prevents accidental data loss

### Don't

- **Don't modify `config.py` model parameters** between saving and loading — architecture must match
- **Don't manually rename checkpoint files** during training — the trainer tracks them by convention
- **Don't delete `checkpoint_last.pt`** during training — it's the recovery point if the process crashes
- **Don't load checkpoints across different `vocab_size` values** — the embedding dimensions won't match

### Recovery scenarios

| Scenario | Action |
|----------|--------|
| Training crashed mid-step | Resume from `checkpoint_last.pt` — no data lost (atomic saves protect checkpoint files) |
| Model overfitting | Resume from `checkpoint_best.pt` with lower learning rate or more regularization |
| Want to try different hyperparameters | Resume from any `checkpoint_step_N.pt` with `--max-iterations` |
| Corrupted checkpoint file | Use the next-most-recent step checkpoint |

---

## Inspecting Checkpoints

You can examine checkpoint contents in Python:

```python
import torch

ckpt = torch.load("checkpoints/checkpoint_step_1500.pt",
                   map_location="cpu",
                   weights_only=False)

# Training progress
print(f"Step: {ckpt['step']}")
print(f"Training loss: {ckpt['train_loss']:.4f}")
print(f"Validation loss: {ckpt['validation_loss']:.4f}")
print(f"Best val loss: {ckpt['best_val_loss']:.4f}")
print(f"Learning rate: {ckpt['learning_rate']:.6f}")
print(f"Patience counter: {ckpt['patience_counter']}")

# Model architecture
config = ckpt['config']
print(f"Embed dim: {config.embedding_dimension}")
print(f"Heads: {config.number_of_heads}")
print(f"Layers: {config.number_of_layers}")
print(f"Block size: {config.block_size}")
print(f"Vocab size: {config.vocab_size}")

# Check for legacy vs. current format
has_rng = "torch_rng_state" in ckpt
print(f"RNG state preserved: {has_rng}")

# File size
import os
size_mb = os.path.getsize("checkpoints/checkpoint_step_1500.pt") / 1e6
print(f"File size: {size_mb:.1f} MB")
```
