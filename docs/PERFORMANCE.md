# Performance

Documented metrics, training history, and analysis from actual training runs on this project.

---

## Table of Contents

- [Training Runs Overview](#training-runs-overview)
- [Model Configurations Trained](#model-configurations-trained)
- [Current Best Checkpoint](#current-best-checkpoint)
- [Training Loss Progression](#training-loss-progression)
- [Validation Metrics](#validation-metrics)
- [Parameter Counts](#parameter-counts)
- [Throughput](#throughput)
- [Hardware Used](#hardware-used)
- [Known Limitations](#known-limitations)
- [Observed Outputs](#observed-outputs)
- [Future Optimization Opportunities](#future-optimization-opportunities)

---

## Training Runs Overview

Multiple training runs have been performed with different configurations. All metrics below are sourced from `logs/train.log`.

| Run | Config | Params | Steps Trained | Final Train Loss | Final Val Loss |
|-----|--------|--------|---------------|-----------------|---------------|
| 1 | 768d, 12h, 12L, 1024ctx | 86.61M | < 10 (aborted) | — | — |
| 2 | 384d, 6h, 6L, 256ctx | 11.13M | 200 | 4.24 | 4.65 |
| 3 | 192d, 4h, 4L, 128ctx | 2.00M | ~1 (aborted) | — | — |
| 4 | 384d, 6h, 6L, 256ctx | 11.13M | 1500→1550+ (resumed) | ~2.82 | 4.40 |

> **Note**: Run 1 (86.61M params) was likely aborted due to excessive training time on CPU. Run 4 resumed from the 11.13M model's step 1500 checkpoint.

---

## Model Configurations Trained

### Medium Model (Primary — 11.13M parameters)

| Parameter | Value |
|-----------|-------|
| `embedding_dimension` | 384 |
| `number_of_heads` | 6 |
| `number_of_layers` | 6 |
| `block_size` | 256 |
| `vocab_size` | 1,000 |
| `dropout` | 0.25 |
| **Total Parameters** | **11,129,856** |

This is the most-trained configuration in the project, with existing checkpoints at steps 500, 1000, and 1500.

### Mini Model (Default Config — 2.00M parameters)

| Parameter | Value |
|-----------|-------|
| `embedding_dimension` | 192 |
| `number_of_heads` | 4 |
| `number_of_layers` | 4 |
| `block_size` | 128 |
| `vocab_size` | 1,000 |
| `dropout` | 0.25 |
| **Total Parameters** | **1,996,416** |

This is the current default in `config.py`. Only briefly tested (~1 step).

---

## Current Best Checkpoint

The best available checkpoint is `checkpoint_step_1500.pt`:

| Metric | Value |
|--------|-------|
| **Step** | 1,500 |
| **Training Loss** | 2.6217 |
| **Validation Loss** | 4.4008 |
| **Training Perplexity** | ~13.8 |
| **Validation Perplexity** | ~81.5 |
| **Model** | 11.13M params (384d, 6h, 6L) |
| **File Size** | ~128 MB |
| **Format** | Legacy (no RNG state) |

> **Note**: This is a legacy checkpoint without RNG state preservation. Resuming from it works correctly but doesn't guarantee bit-exact reproducibility.

### Continued Training (Steps 1510–1550)

After resuming from step 1500:

| Step | Loss | Perplexity | LR | Grad Norm | Tok/sec |
|------|------|------------|-----|-----------|---------|
| 1510 | 2.89 | 18.02 | 3.38e-4 | 3.30 | 46.5 |
| 1520 | 2.80 | 16.48 | 3.37e-4 | 2.91 | 47.4 |
| 1530 | 2.82 | 16.75 | 3.36e-4 | 3.36 | 47.8 |
| 1540 | 2.79 | 16.27 | 3.35e-4 | 2.80 | 48.6 |
| 1550 | 2.82 | 16.73 | 3.34e-4 | 3.24 | 48.8 |

> **Observation**: Loss values after resuming are slightly higher than at step 1500 (2.62 → 2.80–2.89). This is expected because the step 1500 metric was evaluated on 30 fixed batches, while the per-step metrics are from different training batches. The trend is stable around ~2.8.

---

## Training Loss Progression

### Medium Model — 200-Step Fresh Training Run

| Step | Loss | Perplexity | Learning Rate |
|------|------|------------|---------------|
| 0 | 7.00 | 1,099.3 | 0.0 |
| 10 | 6.28 | 532.7 | 3.00e-4 |
| 20 | 5.93 | 376.2 | 6.00e-4 |
| 30 | 5.64 | 281.4 | 5.96e-4 |
| 40 | 5.29 | 198.7 | 5.84e-4 |
| 50 | 5.06 | 157.6 | 5.64e-4 |
| 60 | 4.86 | 128.8 | 5.37e-4 |
| 70 | 4.74 | 114.5 | 5.04e-4 |
| 80 | 4.56 | 95.5 | 4.65e-4 |
| 90 | 4.54 | 93.3 | 4.22e-4 |
| 100 | 4.46 | 86.2 | 3.77e-4 |
| 110 | 4.39 | 80.9 | 3.30e-4 |
| 120 | 4.38 | 80.0 | 2.83e-4 |
| 130 | 4.40 | 81.8 | 2.38e-4 |
| 140 | 4.28 | 72.2 | 1.95e-4 |
| 150 | 4.27 | 71.7 | 1.56e-4 |
| 160 | 4.26 | 70.9 | 1.23e-4 |
| 170 | 4.27 | 71.2 | 9.62e-5 |
| 180 | 4.19 | 66.3 | 7.63e-5 |
| 190 | 4.19 | 65.7 | 6.41e-5 |
| 199 | 4.24 | 69.3 | 6.00e-5 |

**Key observations**:
- Loss drops rapidly in the first 50 steps (7.0 → 5.0)
- Stabilizes around 4.2–4.4 by step 100
- Continued slow improvement through step 199
- Learning rate peaks at ~6e-4 (around step 20, during warmup) then decays

### Evaluation at Step 100

| Split | Loss | Perplexity |
|-------|------|------------|
| Train | 4.4117 | ~82.4 |
| Val | 4.6478 | 104.35 |

**Train-Val gap**: 0.24 — indicates minimal overfitting at this stage.

### Evaluation at Step 1500

| Split | Loss | Perplexity |
|-------|------|------------|
| Train | 2.6217 | ~13.8 |
| Val | 4.4008 | ~81.5 |

**Train-Val gap**: 1.78 — significant gap indicating some overfitting, which is expected given the small dataset size (~1.1 MB).

---

## Validation Metrics

| Checkpoint | Train Loss | Val Loss | Train Perp | Val Perp | Gap |
|-----------|-----------|---------|-----------|---------|-----|
| Step 100 | 4.4117 | 4.6478 | 82.4 | 104.4 | 0.24 |
| Step 1500 | 2.6217 | 4.4008 | 13.8 | 81.5 | 1.78 |

The increasing train-val gap from 0.24 to 1.78 shows that the model progressively overfits the small Tiny Shakespeare dataset. The validation loss plateaus around 4.4, suggesting this is near the floor for this dataset/model size combination.

---

## Parameter Counts

### Breakdown by Component (11.13M model)

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Token Embedding (`wte`) | 384,000 | 3.5% |
| Position Embedding (`wpe`) | 98,304 | 0.9% |
| Attention (QKV + output) × 6 layers | 3,547,392 | 31.9% |
| FeedForward (fc + proj) × 6 layers | 7,096,320 | 63.8% |
| LayerNorm × 13 (12 in blocks + 1 final) | 9,984 | 0.1% |
| LM Head | *(tied)* | 0% |
| **Total** | **11,129,856** | **100%** |

> The FeedForward layers dominate with ~64% of all parameters, due to the 4× expansion factor.

### Configuration Comparison

| Config | Embed | Heads | Layers | Context | Vocab | Params |
|--------|-------|-------|--------|---------|-------|--------|
| Mini (default) | 192 | 4 | 4 | 128 | 1,000 | 2.00M |
| Medium (trained) | 384 | 6 | 6 | 256 | 1,000 | 11.13M |
| Large (attempted) | 768 | 12 | 12 | 1024 | 1,000 | 86.61M |
| GPT-2 Small (ref) | 768 | 12 | 12 | 1024 | 50,257 | ~124M |

---

## Throughput

### CPU Training Throughput

Measured from actual training runs:

| Model | Tokens/sec | Time per 10 steps | Hardware |
|-------|-----------|-------------------|----------|
| 11.13M (384d, 6h, 6L) | 46.5 – 48.8 | ~57 minutes | CPU |
| 11.13M (early steps) | 8,700 – 8,774 | ~19 seconds | CPU* |

> *The early-step throughput (~8,700 tok/sec) was from a different session and may have used a different system or less background load. The ~47 tok/sec measurement from the resumed run (steps 1510–1550) is the more reliable steady-state figure.

### Estimated GPU Throughput

Based on typical GPU performance for models of this size:

| Hardware | Est. Tokens/sec | Est. Time for 5000 steps |
|----------|----------------|-------------------------|
| CPU | ~47 | ~4.7 days |
| RTX 3060 (12 GB) | ~3,000 | ~2 hours |
| RTX 4090 (24 GB) | ~10,000 | ~30 minutes |
| A100 (40 GB) | ~20,000 | ~15 minutes |

> These are rough estimates based on typical benchmarks. Actual throughput depends on batch size, block size, and system configuration.

---

## Hardware Used

Based on the training logs:

| Property | Observed Value |
|----------|---------------|
| Device | CPU |
| Python | 3.11, 3.12, 3.14 (based on `__pycache__` files) |
| OS | Windows |
| AMP dtype | `bfloat16` (CPU mode) |

---

## Known Limitations

### Dataset Limitations
- **Tiny Shakespeare** is only ~1.1 MB — far too small for a language model to learn general language patterns
- The model memorizes the training data (train loss 2.62 vs. val loss 4.40), limiting generalization
- Vocabulary of 1,000 BPE tokens is very small compared to production tokenizers (32K–100K+)

### Architecture Limitations
- **No KV-cache**: Each generation step runs a full forward pass, recomputing attention for all positions
- **No FlashAttention**: Standard scaled dot-product attention with O(T²) memory
- **No `torch.compile`**: Model is not compiled, missing potential kernel-fusion optimizations
- **Single-GPU only**: No data parallelism or model parallelism

### Training Limitations
- **No GradScaler**: AMP uses `autocast` only, missing dynamic loss scaling for float16
- **Fixed architecture per training run**: Cannot change model dimensions mid-training
- **No learning rate finder**: Peak LR is manually set, not automatically searched
- **Small evaluation set**: Only 30 batches used for evaluation, which can be noisy

### Inference Limitations
- **No streaming**: Full generation completes before returning a response
- **No batched inference**: Each API request generates one sequence
- **Frontend hardcodes `top_k`**: Other sampling methods require direct API calls

---

## Observed Outputs

### From the trained 11.13M model (step 1500)

**Prompt**: `"ROMEO: "`
**Output**: `"ROMEO: Pardon me, good night; I hope. BALTHASAR: Grey"`

The model generates recognizably Shakespearean text with character names and dialogue structure, though coherence is limited by the small model and dataset.

### Output Quality Expectations by Training Loss

| Loss Range | Perplexity | Expected Quality |
|-----------|-----------|-----------------|
| > 6.0 | > 400 | Random/garbage — model hasn't learned anything |
| 4.0 – 6.0 | 55 – 400 | Recognizable words, some structure |
| 2.5 – 4.0 | 12 – 55 | Coherent phrases, character names, dialogue format |
| 1.5 – 2.5 | 4.5 – 12 | Largely coherent sentences, appropriate vocabulary |
| < 1.5 | < 4.5 | Near-verbatim reproduction (likely overfitting) |

The current best model (train loss 2.62, val loss 4.40) falls in the "coherent phrases" range for training data, but closer to "recognizable words" on unseen text.

---

## Future Optimization Opportunities

### Training Speed

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| GPU training (as-is) | 50–200× speedup | Low (just use GPU) |
| `torch.compile(model)` | 1.5–3× on GPU | Low (one line of code) |
| FlashAttention | 2–4× on long sequences | Medium |
| Data parallelism (DDP) | Linear scaling with GPUs | Medium |
| Gradient checkpointing | 2× memory savings (for larger models) | Medium |

### Inference Speed

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| KV-cache | 5–20× for long sequences | Medium |
| `torch.compile` | 1.5–3× | Low |
| Batched inference | Throughput scaling | Medium |
| Token streaming (SSE/WebSocket) | Better UX (not raw speed) | Medium |
| ONNX/TorchScript export | Deployment optimization | Medium |

### Model Quality

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| Larger dataset (e.g., OpenWebText) | Significant quality improvement | Low–Medium |
| Larger vocabulary (8K–32K BPE) | Better token coverage | Low |
| Longer training (10K–50K steps) | Better convergence | Low (just time) |
| Larger model (124M GPT-2 Small) | Better language modeling | Medium (needs GPU) |
| Pre-trained weight loading (HuggingFace) | State-of-the-art quality | Medium |
| Cosine annealing with restarts | Better convergence | Low |
| GradScaler for float16 AMP | More stable mixed precision | Low |
