# Developer Guide

A practical guide for developers who want to modify, extend, or experiment with this GPT-2 implementation.

---

## Table of Contents

- [Getting Started](#getting-started)
- [How to Train from Scratch](#how-to-train-from-scratch)
- [How to Resume Training](#how-to-resume-training)
- [How to Evaluate a Checkpoint](#how-to-evaluate-a-checkpoint)
- [How to Modify Model Size](#how-to-modify-model-size)
- [How to Add a New Sampling Method](#how-to-add-a-new-sampling-method)
- [How to Use a Different Dataset](#how-to-use-a-different-dataset)
- [How to Add a New API Endpoint](#how-to-add-a-new-api-endpoint)
- [Repository Conventions](#repository-conventions)
- [Coding Conventions](#coding-conventions)
- [Common Debugging Tips](#common-debugging-tips)
- [Running Verification Tests](#running-verification-tests)

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Recommended) A CUDA-capable GPU for training

### Environment Setup

```bash
# Clone and enter the repository
git clone https://github.com/yasirwaqas5/GPT_2_CELEBAL.git
cd GPT_2_CELEBAL

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Everything Works

```bash
# Run the training smoke test (15 steps in a temp directory)
python verify_train.py

# Run the full checkpoint verification suite
python verify_infrastructure.py
```

---

## How to Train from Scratch

### 1. Default configuration

```bash
python train.py
```

This uses the defaults from `config.py`:
- Mini model (192 embed, 4 heads, 4 layers, 128 context, ~2M params)
- 5,000 iterations, eval every 500 steps
- Downloads Tiny Shakespeare automatically

### 2. Custom configuration

Edit `config.py` to change model or training parameters, then run:

```bash
python train.py --max-iterations 10000 --save-every 250
```

### 3. What to expect

```
========================================
GPT-2 Model Summary
========================================
Embedding Dimension: 192
Number of Heads:     4
Number of Layers:    4
Context Block Size:  128
Vocabulary Size:     1000
Total Trainable Params: 1,996,416 (2.00M)
========================================

step    0 | loss 6.9426 | perplexity 1035.51 | lr 0.00e+00 | ...
step   10 | loss 6.2780 | perplexity  532.72 | lr 3.00e-04 | ...
...
```

---

## How to Resume Training

### From the last checkpoint

```bash
python train.py --resume checkpoints/checkpoint_last.pt
```

### From the best checkpoint

```bash
python train.py --from-best
```

### Extend max iterations

```bash
python train.py --resume checkpoints/checkpoint_step_1500.pt --max-iterations 10000
```

### What happens internally

1. Model architecture is reconstructed from `checkpoint["config"]`
2. Model weights and optimizer state are loaded
3. Step counter is set to `saved_step + 1`
4. RNG states are restored (if available) for exact reproducibility
5. Training loop continues from the restored step

---

## How to Evaluate a Checkpoint

```bash
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

Output:
```
Evaluation complete for checkpoints/checkpoint_step_1500.pt:
  train_loss=2.6217, val_loss=4.4008
```

---

## How to Modify Model Size

Edit `config.py` — class `ModelConfig`:

### Preset configurations

| Config | embed | heads | layers | block_size | ~Params |
|--------|-------|-------|--------|------------|---------|
| Tiny | 128 | 2 | 2 | 64 | ~500K |
| **Mini (default)** | **192** | **4** | **4** | **128** | **~2M** |
| Small | 384 | 6 | 6 | 256 | ~11M |
| Medium | 512 | 8 | 8 | 512 | ~30M |
| GPT-2 Small | 768 | 12 | 12 | 1024 | ~124M |

### Example: Switching to the Small configuration

```python
@dataclass
class ModelConfig:
    vocab_size: int = 50257     # Or keep 1000 for Tiny Shakespeare
    block_size: int = 256
    embedding_dimension: int = 384
    number_of_heads: int = 6
    number_of_layers: int = 6
    dropout: float = 0.2
```

### Constraints

- `embedding_dimension` must be divisible by `number_of_heads`
- `block_size` determines maximum context length and memory usage
- Larger models require more VRAM; the GPT-2 Small config needs ~4-6 GB on GPU
- `vocab_size` must match your tokenizer; the default SentencePiece model uses 1,000

> **Warning**: After changing model size, you must train from scratch. Existing checkpoints will not load because the state dict shapes won't match.

---

## How to Add a New Sampling Method

### Step 1: Implement the method in `sampling.py`

Add a new branch to `sample_next_token()`:

```python
def sample_next_token(logits, method="greedy", temperature=1.0,
                      top_k=None, top_p=None):
    # ... existing methods ...

    elif method == "min_p":
        # Min-P sampling: keep tokens above min_p * max_probability
        probs = torch.softmax(logits / temperature, dim=-1)
        max_prob = probs.max()
        threshold = min_p * max_prob    # min_p passed via top_p param
        mask = probs < threshold
        logits[mask] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)
```

### Step 2: Add the method to the CLI

In `generate.py`, update the `--method` choices:

```python
parser.add_argument('--method', type=str, default='top_k',
                    choices=['greedy', 'temperature', 'top_k', 'top_p', 'min_p'])
```

### Step 3: Update the API

In `app.py`, no code changes are needed — the `method` field in `GenerateRequest` is a free string that gets passed directly to `model.generate()`.

### Step 4: (Optional) Update the frontend

In `static/app.js`, the sampling method is currently hardcoded to `"top_k"`. To make it selectable, add a dropdown in `index.html` and update the fetch call in `app.js`.

---

## How to Use a Different Dataset

### Step 1: Prepare your text file

Place your text file at `data/raw/input.txt` (or update the path in `config.py`).

Requirements:
- Plain text (UTF-8)
- The larger the better; minimum ~100 KB for meaningful training
- Remove any binary or special formatting

### Step 2: Clear cached data

Delete the processed files so they get regenerated:

```bash
# Windows
del data\processed\train.pt data\processed\val.pt data\processed\sp.model data\processed\sp.vocab

# macOS/Linux
rm data/processed/train.pt data/processed/val.pt data/processed/sp.model data/processed/sp.vocab
```

### Step 3: Adjust vocabulary size (optional)

For larger or more diverse datasets, increase the vocabulary:

In `config.py`, the vocab_size in `ModelConfig` is overridden at runtime by `tokenizer.vocab_size()`. To change the SentencePiece vocabulary size, edit the `train()` call in `tokenizer.py` or in `train.py` — `prepare_data()`:

```python
tokenizer.train(config.dataset_raw_path, config.sp_model_prefix, vocab_size=4000)
```

### Step 4: Train

```bash
python train.py
```

The pipeline will automatically retrain the tokenizer and re-tokenize the dataset.

---

## How to Add a New API Endpoint

### Example: Add a `/api/tokenize` endpoint

In `app.py`:

```python
@app.post("/api/tokenize")
async def tokenize_text(payload: dict):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "text": text,
        "token_ids": token_ids,
        "tokens": tokens,
        "count": len(token_ids)
    }
```

---

## Repository Conventions

### File Organization

| Category | Files |
|----------|-------|
| Model definition | `model.py`, `attention.py`, `transformer.py`, `generate.py` |
| Training | `trainer.py`, `train.py`, `config.py` |
| Data | `dataset.py`, `tokenizer.py` |
| Inference | `app.py`, `sampling.py`, `generate.py` (CLI) |
| Frontend | `index.html`, `static/style.css`, `static/app.js` |
| Utilities | `utils.py` |
| Tests | `verify_infrastructure.py`, `verify_train.py` |

### Import Pattern

Internal imports use relative module names:

```python
from config import ModelConfig, TrainingConfig
from model import GPT2
from tokenizer import SentencePieceTokenizer
from utils import save_checkpoint, load_checkpoint
```

### Configuration Pattern

All configurable values are centralized in `config.py` dataclasses. CLI arguments override specific values at runtime, never by modifying the dataclass defaults.

---

## Coding Conventions

### Style

- **Python version**: 3.10+ (uses modern type hints)
- **Type annotations**: Used on function signatures, especially in `config.py` and `app.py`
- **Dataclasses**: Used for configuration (`ModelConfig`, `TrainingConfig`)
- **Pydantic models**: Used for API request validation (`GenerateRequest`)

### Naming

| Convention | Example |
|-----------|---------|
| Classes | `PascalCase`: `GPT2`, `TransformerBlock`, `SentencePieceTokenizer` |
| Functions | `snake_case`: `create_causal_mask`, `sample_next_token`, `get_learning_rate` |
| Constants | Not explicitly used; defaults are dataclass fields |
| Private methods | `_prefix`: `_log`, `_rng_state`, `_checkpoint_state` |
| Files | `snake_case.py`: `attention.py`, `transformer.py` |

### Module Dependencies

The codebase avoids circular imports through a clear dependency hierarchy:

```
config.py         (no internal imports)
    ↓
generate.py       (imports: tokenizer)
attention.py      (no internal imports)
    ↓
transformer.py    (imports: attention, generate)
    ↓
model.py          (imports: config, generate, transformer, sampling)
sampling.py       (no internal imports)
    ↓
dataset.py        (no internal imports)
utils.py          (no internal imports)
    ↓
trainer.py        (imports: config, dataset, utils)
    ↓
train.py          (imports: config, tokenizer, model, trainer, utils)
app.py            (imports: config, tokenizer, model, utils)
```

---

## Common Debugging Tips

### 1. Model produces gibberish

- **Cause**: Insufficient training, or loading a checkpoint trained with different hyperparameters.
- **Fix**: Train for more iterations. The loss should drop below ~4.0 before outputs become recognizable. Check that the checkpoint was saved with the same model architecture.

### 2. Loss is NaN or explodes

- **Cause**: Learning rate too high, or gradient explosion.
- **Fix**: Reduce `learning_rate` in `config.py`. The default gradient clipping (1.0) should prevent most explosions; if it persists, try a lower clip value.

### 3. "RuntimeError: size mismatch" on checkpoint load

- **Cause**: Model architecture in `config.py` doesn't match the checkpoint.
- **Fix**: The checkpoint stores its own `ModelConfig`. When resuming, the model is built from the checkpoint's config, not from `config.py`. If you changed `config.py`, train from scratch.

### 4. CUDA out of memory

- **Cause**: Model or batch size too large for GPU VRAM.
- **Fix**:
  - Reduce `batch_size` (e.g., from 16 to 8 or 4)
  - Reduce model dimensions
  - Reduce `block_size` (context length has a quadratic effect on attention memory)
  - Increase `gradient_accumulation_steps` to maintain effective batch size

### 5. Training is very slow on CPU

- **Cause**: Expected behavior — transformer training is compute-intensive.
- **Fix**: Use a GPU, or use the Mini model config (2M params). See [docs/TRAINING.md](TRAINING.md#cpu-considerations) for tips.

### 6. SentencePiece errors on import

- **Cause**: `sentencepiece` package not installed or version incompatible.
- **Fix**: `pip install sentencepiece>=0.1.99`

### 7. FastAPI won't start

- **Cause**: Port 8000 already in use, or missing dependencies.
- **Fix**: Kill the existing process on port 8000, or check that all requirements are installed. See [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### 8. Perplexity is high but loss is decreasing

- **Cause**: Normal early in training. Perplexity = `exp(loss)`, so a loss of 4.0 means perplexity ~55.
- **Fix**: Keep training. Expected progression for this dataset:
  - Loss ~7.0 → perplexity ~1,000 (random initialization)
  - Loss ~4.0 → perplexity ~55 (learning patterns)
  - Loss ~2.5 → perplexity ~12 (coherent outputs)

---

## Running Verification Tests

### Training Smoke Test

```bash
python verify_train.py
```

Runs 15 training steps in an isolated temp directory. Verifies:
- Model initializes correctly
- Training loop runs without errors
- Checkpoint files are created
- Production checkpoints are not affected

### Checkpoint System Verification

```bash
python verify_infrastructure.py
```

Comprehensive test suite covering:
- Legacy checkpoint loading with SHA-256 weight verification
- Fresh training → checkpoint creation → resume → continuation
- Metadata correctness (step numbers, loss values, config)
- Overwrite protection enforcement
- Best checkpoint update logic

Both tests are safe to run — they use isolated temporary directories and never modify existing checkpoints or data.
