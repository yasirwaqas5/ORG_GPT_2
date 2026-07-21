# Troubleshooting

Solutions for common issues encountered when setting up, training, and running the GPT-2 project.

---

## Table of Contents

- [Installation Issues](#installation-issues)
  - [SentencePiece Installation Errors](#sentencepiece-installation-errors)
  - [PyTorch Installation](#pytorch-installation)
  - [Windows-Specific Issues](#windows-specific-issues)
- [Training Issues](#training-issues)
  - [Missing Checkpoint](#missing-checkpoint)
  - [Legacy Checkpoint Warning](#legacy-checkpoint-warning)
  - [Resume Failures](#resume-failures)
  - [Model Loading Failures](#model-loading-failures)
  - [Out of Memory (OOM)](#out-of-memory-oom)
  - [Slow CPU Training](#slow-cpu-training)
  - [Loss is NaN](#loss-is-nan)
  - [Training Stuck or Not Converging](#training-stuck-or-not-converging)
- [Server Issues](#server-issues)
  - [FastAPI Startup Failures](#fastapi-startup-failures)
  - [Port Conflicts](#port-conflicts)
  - [Model Not Loaded Error](#model-not-loaded-error)
- [Generation Issues](#generation-issues)
  - [Gibberish Output](#gibberish-output)
  - [Repetitive Output](#repetitive-output)
  - [Empty or Very Short Output](#empty-or-very-short-output)

---

## Installation Issues

### SentencePiece Installation Errors

**Symptom**: `ModuleNotFoundError: No module named 'sentencepiece'` or build errors during `pip install sentencepiece`.

**Solutions**:

1. Install with pip:
   ```bash
   pip install sentencepiece>=0.1.99
   ```

2. On Windows, if the build fails:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install sentencepiece
   ```

3. If you get a C++ compiler error on Windows, install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

4. On Linux, you may need:
   ```bash
   sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
   pip install sentencepiece
   ```

---

### PyTorch Installation

**Symptom**: `ModuleNotFoundError: No module named 'torch'` or CUDA not detected.

**Solutions**:

1. For **CPU-only** (simplest):
   ```bash
   pip install torch>=2.0.0
   ```

2. For **CUDA** (GPU acceleration):
   Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and select your platform. Example for CUDA 12.1:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

4. For **Apple Silicon** (MPS):
   ```bash
   pip install torch>=2.0.0
   ```
   MPS is automatically detected. Verify:
   ```python
   import torch
   print(torch.backends.mps.is_available())  # Should be True
   ```

---

### Windows-Specific Issues

**Symptom**: Various path or permission errors on Windows.

**Solutions**:

1. **Long path names**: Enable long paths in Windows:
   - Run as admin: `reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f`
   - Or: Settings → System → Developer Settings → Enable long paths

2. **Permission denied on checkpoint save**: Run your terminal as administrator, or ensure the `checkpoints/` directory is not read-only.

3. **`os.replace` errors**: This can happen if another process (antivirus, file indexer) has a lock on the checkpoint file. Close any programs that might be scanning the checkpoint directory.

4. **Unicode errors in dataset**: Ensure `input.txt` is saved with UTF-8 encoding.

---

## Training Issues

### Missing Checkpoint

**Symptom**: `FileNotFoundError: No such file or directory: 'checkpoints/checkpoint_best.pt'`

**Cause**: No checkpoint exists at the specified path.

**Solutions**:

1. List available checkpoints:
   ```bash
   # Windows
   dir checkpoints\*.pt

   # macOS/Linux
   ls checkpoints/*.pt
   ```

2. If no checkpoints exist, train from scratch first:
   ```bash
   python train.py
   ```

3. If using `--from-best` and no best checkpoint exists, use `--resume` with a specific step checkpoint instead.

4. If the checkpoint was split for GitHub distribution, the server should reassemble it automatically. Verify that `checkpoint_best.zip.aa` and `.ab` exist in the checkpoints directory.

---

### Legacy Checkpoint Warning

**Symptom**: `Resumed legacy checkpoint after step N; next step is N+1. RNG/DataLoader state is unavailable in this legacy checkpoint.`

**Cause**: The checkpoint was created before the RNG state preservation feature was added.

**Impact**: Training resumes correctly, but:
- Batch ordering may differ from the original run
- Dropout patterns may differ
- Results won't be bit-for-bit identical to an uninterrupted run

**Is this a problem?** Generally, no. The model weights and optimizer state are fully restored. The training quality is not affected — only the exact reproducibility of the sequence of random numbers.

**Prevention**: New checkpoints saved during training will include full RNG state, so future resumes from these checkpoints will be fully reproducible.

---

### Resume Failures

**Symptom**: Various errors when resuming from a checkpoint.

#### "Checkpoint is at step N, but max_iterations is M"

**Cause**: The checkpoint's step is already at or past `max_iterations`.

**Fix**: Increase max iterations:
```bash
python train.py --resume checkpoints/checkpoint_step_1500.pt --max-iterations 10000
```

#### "size mismatch for transformer.h.0.attn.c_attn.weight"

**Cause**: The model architecture doesn't match the checkpoint. This happens when `config.py` was modified between saving and loading.

**Fix**: Don't change model architecture parameters when resuming. The model is reconstructed from the checkpoint's saved `ModelConfig`, but if the code imports have changed, there can be mismatches. Verify the checkpoint's config:

```python
import torch
ckpt = torch.load("checkpoints/checkpoint_step_1500.pt", map_location="cpu", weights_only=False)
print(ckpt["config"])
```

#### "KeyError: 'model_state'" or similar

**Cause**: Corrupted or incompatible checkpoint file.

**Fix**: Try a different checkpoint. If all checkpoints are corrupted, you'll need to train from scratch.

---

### Model Loading Failures

**Symptom**: Errors when loading a model from a checkpoint.

**Common causes**:
1. **Wrong device**: The checkpoint was saved on GPU but you're loading on CPU (or vice versa). This is handled automatically via `map_location`, but verify:
   ```python
   ckpt = torch.load(path, map_location="cpu", weights_only=False)
   ```

2. **PyTorch version mismatch**: Checkpoints saved with a very different PyTorch version may not load. Ensure you're using `torch>=2.0.0`.

3. **Missing `config` key**: Very old or manually created checkpoints may not have the `config` key. The code expects `checkpoint["config"]` to be a `ModelConfig` dataclass.

---

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory` or system memory exhaustion.

**Solutions** (in order of impact):

1. **Reduce batch size**:
   Edit `config.py`:
   ```python
   batch_size: int = 8  # or 4
   ```
   Increase `gradient_accumulation_steps` to maintain effective batch size:
   ```python
   gradient_accumulation_steps: int = 8  # if batch_size is halved
   ```

2. **Reduce context length**:
   ```python
   block_size: int = 64  # default is 128
   ```
   > Attention memory scales quadratically with `block_size`.

3. **Reduce model size**:
   ```python
   embedding_dimension: int = 128
   number_of_heads: int = 2
   number_of_layers: int = 2
   ```

4. **Disable AMP** (if it's causing issues):
   ```python
   use_amp: bool = False
   ```

5. **Close other GPU applications** that may be consuming VRAM.

---

### Slow CPU Training

**Symptom**: Training runs at ~47–49 tokens/second or slower.

**This is expected.** Transformer training is compute-intensive and CPUs are much slower than GPUs for matrix operations.

**Mitigation strategies**:

1. **Use the Mini model** (default config, ~2M params) instead of larger configurations
2. **Reduce `max_iterations`** for quick experiments:
   ```bash
   python train.py --max-iterations 500
   ```
3. **Use a GPU** — even a modest GPU is 10–100x faster than CPU for this workload
4. **Use Google Colab** (free GPU access) for longer training runs
5. **Save frequently** with `--save-every 100` so you don't lose progress

---

### Loss is NaN

**Symptom**: `step N | loss nan | perplexity nan`

**Causes and fixes**:

1. **Learning rate too high**: Reduce `learning_rate` in `config.py` (try 1e-4 or lower)
2. **Gradient explosion**: The default `gradient_clip=1.0` should prevent this, but try a lower value (0.5 or 0.1)
3. **AMP issues on CPU**: Try `use_amp: bool = False`
4. **Data issues**: Verify `data/processed/train.pt` loads correctly:
   ```python
   import torch
   data = torch.load("data/processed/train.pt")
   print(data.shape, data.dtype, data.min(), data.max())
   ```

---

### Training Stuck or Not Converging

**Symptom**: Loss doesn't decrease significantly after many steps.

**Solutions**:

1. **Check learning rate**: If `warmup_steps` is too high relative to `max_iterations`, the model may spend too long at a low LR. Try `warmup_steps: int = 100`.
2. **Verify data**: Ensure the dataset is loaded correctly and contains meaningful text.
3. **Check model size vs. data**: A very large model on a very small dataset may not converge well. Try a smaller model.
4. **Reset and try again**: Sometimes a different random seed helps:
   ```python
   seed: int = 42  # Try different values
   ```

---

## Server Issues

### FastAPI Startup Failures

**Symptom**: Server crashes or won't start.

**Common causes**:

1. **Missing dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic python-multipart
   ```

2. **No checkpoint available**: The server still starts but reports `"status": "not_trained"`. Train a model first.

3. **Import errors**: Verify all project files are present and not corrupted:
   ```bash
   python -c "from config import ModelConfig, TrainingConfig; print('OK')"
   python -c "from model import GPT2; print('OK')"
   python -c "from tokenizer import SentencePieceTokenizer; print('OK')"
   ```

4. **Missing tokenizer model**: Ensure `data/processed/sp.model` exists. If not, run `python train.py` to generate it.

---

### Port Conflicts

**Symptom**: `OSError: [Errno 98] Address already in use` or `[WinError 10048] Only one usage of each socket address is normally permitted`

**Solutions**:

1. **Kill the existing process**:
   ```bash
   # Find the process
   # Windows:
   netstat -ano | findstr :8000
   # Then kill it:
   taskkill /PID <PID> /F

   # macOS/Linux:
   lsof -i :8000
   kill <PID>
   ```

2. **Change the port**: Edit the `uvicorn.run()` call in `app.py`:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8001)
   ```

3. **Wait**: If the server recently crashed, the port may take a few seconds to be released.

---

### Model Not Loaded Error

**Symptom**: `POST /api/generate` returns `{"detail": "Model not loaded"}`

**Cause**: No checkpoint was found at startup, or the loaded checkpoint failed to initialize the model.

**Solutions**:

1. Verify checkpoints exist:
   ```bash
   # Windows
   dir checkpoints\*.pt

   # macOS/Linux
   ls checkpoints/*.pt
   ```

2. Train a model if no checkpoints exist:
   ```bash
   python train.py
   ```

3. Check server logs for error messages during startup.

4. Try loading a checkpoint manually via the API:
   ```bash
   curl -X POST http://localhost:8000/api/load-checkpoint \
     -H "Content-Type: application/json" \
     -d '{"checkpoint": "checkpoint_step_1500.pt"}'
   ```

---

## Generation Issues

### Gibberish Output

**Symptom**: Generated text is random characters or nonsensical.

**Causes**:

1. **Model undertrained**: Train for more iterations. Loss should be below ~3.5 for coherent outputs with Tiny Shakespeare.
2. **Wrong checkpoint**: Loading an early-step checkpoint (e.g., step 500) may produce lower quality than a later one. Try the highest-step or best checkpoint.
3. **Temperature too high**: Values above 1.5 produce increasingly random text. Try `temperature=0.7`.

---

### Repetitive Output

**Symptom**: Generated text repeats the same phrase over and over.

**Solutions**:

1. **Increase temperature**: Try `temperature=0.9` or higher
2. **Use nucleus sampling**: `method="top_p"` with `top_p=0.9` gives more diverse outputs than greedy or low-temperature sampling
3. **Reduce top_k**: A very low `top_k` (e.g., 5) can cause repetition. Try `top_k=50`

---

### Empty or Very Short Output

**Symptom**: Generation returns the prompt with no or very few additional tokens.

**Causes**:

1. **`max_tokens` too low**: Increase `max_tokens` in the request
2. **Prompt exceeds block_size**: If the encoded prompt is longer than `block_size`, it gets truncated. Use shorter prompts or increase `block_size` (requires retraining)
3. **Model issue**: Check if the model loaded correctly via `/api/model-info`
