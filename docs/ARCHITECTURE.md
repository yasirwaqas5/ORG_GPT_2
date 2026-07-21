# Architecture

This document provides a detailed walkthrough of the GPT-2 model architecture, component relationships, and data flow as implemented in this repository.

---

## Table of Contents

- [High-Level System Overview](#high-level-system-overview)
- [Component Dependency Graph](#component-dependency-graph)
- [Model Architecture](#model-architecture)
  - [Token Embedding](#token-embedding)
  - [Positional Embedding](#positional-embedding)
  - [Multi-Head Causal Attention](#multi-head-causal-attention)
  - [Feed-Forward Network](#feed-forward-network)
  - [Transformer Block](#transformer-block)
  - [Layer Normalization](#layer-normalization)
  - [GELU Activation](#gelu-activation)
  - [Residual Connections](#residual-connections)
  - [Language Modeling Head](#language-modeling-head)
- [Training Flow](#training-flow)
- [Inference Flow](#inference-flow)
- [Checkpoint Flow](#checkpoint-flow)
- [FastAPI Flow](#fastapi-flow)
- [Frontend Flow](#frontend-flow)
- [Tokenizer](#tokenizer)
- [Dataset Pipeline](#dataset-pipeline)
- [Sampling Pipeline](#sampling-pipeline)
- [Module Reference](#module-reference)

---

## High-Level System Overview

```mermaid
graph TB
    subgraph Data Pipeline
        RAW[Raw Text<br>input.txt] --> TOK[SentencePiece<br>Tokenizer]
        TOK --> TRAIN_DATA[train.pt]
        TOK --> VAL_DATA[val.pt]
    end

    subgraph Training
        TRAIN_DATA --> DATASET[GPTDataset]
        VAL_DATA --> DATASET
        DATASET --> LOADER[DataLoader]
        LOADER --> TRAINER[Trainer]
        TRAINER --> MODEL[GPT-2 Model]
        TRAINER --> CKPT[Checkpoints]
    end

    subgraph Inference
        CKPT --> API[FastAPI Server]
        API --> GEN[Generation Pipeline]
        GEN --> SAMP[Sampling]
        SAMP --> OUTPUT[Generated Text]
    end

    subgraph Frontend
        API --> HTML[Web Playground]
        HTML --> |HTTP POST| API
    end
```

---

## Component Dependency Graph

```mermaid
graph LR
    config[config.py] --> model[model.py]
    config --> trainer[trainer.py]
    config --> app[app.py]

    generate_mod[generate.py<br>LayerNorm, GELU,<br>FeedForward, Embeddings] --> model
    generate_mod --> transformer[transformer.py]

    attention[attention.py<br>MultiHeadAttention] --> transformer

    transformer --> model

    sampling[sampling.py] --> model

    tokenizer[tokenizer.py] --> train[train.py]
    tokenizer --> app
    tokenizer --> generate_mod

    dataset[dataset.py] --> trainer
    utils[utils.py] --> trainer
    utils --> train
    utils --> app

    model --> trainer
    model --> app
    trainer --> train
```

---

## Model Architecture

The model follows the GPT-2 architecture with **Pre-Layer Normalization** (Pre-LN). This means LayerNorm is applied *before* each sublayer, which improves training stability compared to the original Post-LN Transformer.

### Full Architecture Diagram

```mermaid
graph TD
    INPUT[Input Token IDs<br>shape: B × T] --> WTE[Token Embedding<br>nn.Embedding vocab × d_model]
    INPUT --> WPE[Positional Embedding<br>nn.Embedding block_size × d_model]
    WTE --> SUM((+))
    WPE --> SUM
    SUM --> TB1[Transformer Block 1]
    TB1 --> TB2[Transformer Block 2]
    TB2 --> TBN[Transformer Block N]
    TBN --> LN_F[Final LayerNorm]
    LN_F --> LM_HEAD[LM Head<br>Linear d_model → vocab]
    LM_HEAD --> LOGITS[Output Logits<br>shape: B × T × vocab]

    style WTE fill:#2d3748,color:#e2e8f0
    style WPE fill:#2d3748,color:#e2e8f0
    style LM_HEAD fill:#2d3748,color:#e2e8f0
```

> **Weight tying**: `lm_head.weight` is set to the same parameter object as `wte.weight`, meaning the token embedding matrix is reused as the output projection. This reduces total parameter count and has been shown to improve language model quality.

---

### Token Embedding

**File**: `generate.py` — class `TokenEmbedding` (lines 58–65)

Wraps `nn.Embedding(vocab_size, embed_dim)`. Converts integer token IDs to dense vectors of dimension `embed_dim`.

```
Input:  [  42,  17, 831,   5 ]        shape: (B, T)
Output: [[ 0.12, -0.34, ...],         shape: (B, T, embed_dim)
         [ 0.56,  0.78, ...],
         [-0.23,  0.91, ...],
         [ 0.45, -0.67, ...]]
```

---

### Positional Embedding

**File**: `generate.py` — class `LearnedPositionEmbedding` (lines 68–77)

The default positional embedding uses a **learned** embedding table of shape `(block_size, embed_dim)`. Position indices `[0, 1, ..., T-1]` are mapped to dense vectors and added element-wise to the token embeddings.

A `SinusoidalPositionEmbedding` (lines 80–93) is also implemented but not used in the default model. It uses the standard sinusoidal formula from "Attention Is All You Need":

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### Multi-Head Causal Attention

**File**: `attention.py` — class `MultiHeadAttention` (lines 51–97)

This is the core attention mechanism. It implements **masked (causal) multi-head self-attention** with a combined QKV projection.

```mermaid
graph TD
    X[Input x<br>B × T × d_model] --> C_ATN[Combined QKV Projection<br>Linear d_model → 3·d_model]
    C_ATN --> SPLIT[Split into Q K V]
    SPLIT --> Q[Q: B × n_heads × T × head_dim]
    SPLIT --> K[K: B × n_heads × T × head_dim]
    SPLIT --> V[V: B × n_heads × T × head_dim]
    Q --> SCORE[Scaled Dot-Product<br>Q·Kᵀ / √d_k]
    K --> SCORE
    SCORE --> MASK[Apply Causal Mask<br>upper triangle → -∞]
    MASK --> SOFTMAX[Softmax]
    SOFTMAX --> DROP1[Attention Dropout]
    DROP1 --> ATTN_V[Attention × V]
    V --> ATTN_V
    ATTN_V --> CONCAT[Concatenate Heads<br>B × T × d_model]
    CONCAT --> C_PROJ[Output Projection<br>Linear d_model → d_model]
    C_PROJ --> DROP2[Residual Dropout]
    DROP2 --> OUT[Output<br>B × T × d_model]
```

**Causal mask** (`create_causal_mask` in `attention.py`, lines 5–7): A lower-triangular matrix that prevents each position from attending to future positions. Implemented via `torch.tril`.

```
Mask for T=4:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**Scaling**: Attention scores are divided by `√(head_dim)` to prevent softmax saturation for large embedding dimensions.

A `SingleHeadAttention` class (lines 30–48) is also implemented as a reference but is not used in the model. It uses separate projection matrices for Q, K, and V.

---

### Feed-Forward Network

**File**: `generate.py` — class `FeedForward` (lines 30–43)

A position-wise two-layer MLP with GELU activation and 4× inner expansion:

```
Input (d_model) → Linear (d_model → 4·d_model) → GELU → Linear (4·d_model → d_model) → Dropout → Output
```

The 4× expansion is standard in GPT-2 and allows the network to learn more complex transformations while keeping the residual stream dimension constant.

---

### Transformer Block

**File**: `transformer.py` — class `TransformerBlock` (lines 6–19)

Each block applies Pre-LN attention and Pre-LN MLP with residual connections:

```python
x = x + attn(ln_1(x))    # Attention sublayer with residual
x = x + mlp(ln_2(x))     # MLP sublayer with residual
```

```mermaid
graph TD
    IN[Input x] --> LN1[LayerNorm 1]
    LN1 --> ATTN[Multi-Head Attention]
    ATTN --> ADD1((+))
    IN --> ADD1

    ADD1 --> LN2[LayerNorm 2]
    LN2 --> MLP[FeedForward MLP]
    MLP --> ADD2((+))
    ADD1 --> ADD2

    ADD2 --> OUT[Output]
```

---

### Layer Normalization

**File**: `generate.py` — class `LayerNorm` (lines 11–21)

Custom implementation matching PyTorch's `nn.LayerNorm` behavior:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
```

- `γ` (weight): initialized to ones
- `β` (bias): initialized to zeros
- `ε`: 1e-5 (default)
- Uses `unbiased=False` variance (population variance, not sample variance)

---

### GELU Activation

**File**: `generate.py` — class `GELU` (lines 24–27)

Uses the **tanh approximation** of the Gaussian Error Linear Unit:

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

This is the same approximation used in the original GPT-2 implementation.

---

### Residual Connections

Residual (skip) connections are used around both the attention and MLP sublayers. They allow gradients to flow directly through the network during backpropagation and enable training of deeper models.

A `ResidualBlock` helper class exists in `generate.py` (lines 46–53) but is not directly used — the `TransformerBlock` implements residual connections inline.

---

### Language Modeling Head

**File**: `model.py` — `self.lm_head` (line 22)

A linear projection from `d_model` to `vocab_size` without bias:

```
logits = Linear(x)    # shape: (B, T, vocab_size)
```

The weight matrix is **tied** to the token embedding weight (line 24), so there is no separate weight matrix for the LM head.

---

## Weight Initialization

**File**: `model.py` — `__init__` (lines 26–30)

All weights are initialized using a specific scheme:

| Module | Initialization |
|--------|---------------|
| `nn.Linear` weights | Normal(μ=0, σ=0.02) |
| `nn.Linear` biases | Zeros |
| `nn.Embedding` weights | Normal(μ=0, σ=0.02) |
| `c_proj.weight` (output projections) | Normal(μ=0, σ=0.02/√(2·N)) where N = number of layers |

The special initialization for `c_proj` follows the GPT-2 convention: the residual stream accumulates contributions from `2N` sublayers (N attention + N MLP), so each output projection's initial contribution is scaled down by `√(2N)` to maintain stable activations at initialization.

---

## Training Flow

```mermaid
sequenceDiagram
    participant CLI as train.py
    participant Prep as Data Preparation
    participant Model as GPT2
    participant Trainer as Trainer
    participant Disk as Checkpoint Files

    CLI->>Prep: download_data() + prepare_data()
    Prep->>Prep: Train SentencePiece tokenizer
    Prep->>Prep: Tokenize → train.pt, val.pt
    CLI->>Model: Initialize GPT2(config)
    CLI->>Trainer: Create Trainer(model, config, data)

    alt Resume mode
        CLI->>Disk: load_checkpoint(path)
        Disk-->>CLI: checkpoint dict
        CLI->>Trainer: restore_checkpoint(ckpt)
    end

    loop Each training step
        Trainer->>Trainer: Update learning rate (cosine schedule)
        loop gradient_accumulation_steps
            Trainer->>Model: Forward pass (logits, loss)
            Trainer->>Model: Backward pass (loss / accum_steps)
        end
        Trainer->>Trainer: Clip gradients (max_norm=1.0)
        Trainer->>Trainer: Optimizer step

        alt Every evaluation_interval steps
            Trainer->>Trainer: estimate_loss() on train + val
            Trainer->>Disk: Save checkpoint_last.pt
            Trainer->>Disk: Save checkpoint_best.pt (if improved)
            Trainer->>Disk: Save checkpoint_step_N.pt
        end

        alt Early stopping triggered
            Trainer->>CLI: Return (patience exceeded)
        end
    end
```

---

## Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant Gen as generate.py / API
    participant Tok as SentencePieceTokenizer
    participant Model as GPT2
    participant Samp as sampling.py

    User->>Gen: prompt text + params
    Gen->>Tok: encode(prompt) → token IDs
    Gen->>Model: model.generate(ids, max_tokens, method, ...)

    loop max_new_tokens times
        Model->>Model: Forward pass → logits
        Model->>Samp: sample_next_token(logits[-1])
        Samp-->>Model: next token ID
        Model->>Model: Append token to sequence
    end

    Model-->>Gen: full token sequence
    Gen->>Tok: decode(sequence) → text
    Gen-->>User: generated text
```

---

## Checkpoint Flow

```mermaid
graph TD
    SAVE[Save Checkpoint] --> TEMP[Write to temp file]
    TEMP --> FLUSH[Flush + fsync]
    FLUSH --> RENAME[Atomic os.replace]
    RENAME --> DONE[Checkpoint on disk]

    subgraph Checkpoint Contents
        MS[model_state_dict]
        OS[optimizer_state_dict]
        STEP[step number]
        BVL[best_val_loss]
        VL[validation_loss]
        TL[train_loss]
        LR[learning_rate]
        PC[patience_counter]
        GAS[gradient_accumulation_steps]
        MC[ModelConfig]
        TC[TrainingConfig dict]
        RNG[RNG states<br>Python + NumPy + Torch + CUDA]
    end
```

---

## FastAPI Flow

```mermaid
sequenceDiagram
    participant Browser
    participant FastAPI as app.py
    participant Model as GPT2

    Note over FastAPI: Startup: auto-detect best checkpoint
    FastAPI->>FastAPI: check_and_reassemble_checkpoint()
    FastAPI->>FastAPI: find_startup_checkpoint()
    FastAPI->>Model: Load model from checkpoint

    Browser->>FastAPI: GET /
    FastAPI-->>Browser: index.html

    Browser->>FastAPI: GET /api/model-info
    FastAPI-->>Browser: {status, dims, params, ...}

    Browser->>FastAPI: GET /api/checkpoints
    FastAPI-->>Browser: {checkpoints: [...]}

    Browser->>FastAPI: POST /api/generate {prompt, params}
    FastAPI->>Model: encode → generate → decode
    FastAPI-->>Browser: {prompt, generated}
```

---

## Frontend Flow

```mermaid
graph TD
    PAGE[Page Load] --> FETCH_INFO[Fetch /api/model-info]
    PAGE --> FETCH_CKPTS[Fetch /api/checkpoints]
    FETCH_INFO --> DISPLAY_STATS[Display Model Stats]
    FETCH_CKPTS --> POPULATE[Populate Checkpoint Dropdown]

    USER[User Action] --> CHANGE_CKPT{Change Checkpoint?}
    CHANGE_CKPT -->|Yes| LOAD[POST /api/load-checkpoint]
    LOAD --> UPDATE_STATS[Update Stats Display]

    USER --> GENERATE{Click Generate?}
    GENERATE -->|Yes| DISABLE[Disable Button]
    DISABLE --> POST[POST /api/generate]
    POST --> TYPEWRITER[Typewriter Text Effect]
    TYPEWRITER --> ENABLE[Re-enable Button]

    USER --> COPY{Click Copy?}
    COPY -->|Yes| CLIPBOARD[Copy to Clipboard]
    CLIPBOARD --> FEEDBACK[Visual Feedback]
```

---

## Tokenizer

**File**: `tokenizer.py`

Two tokenizer implementations:

### SentencePieceTokenizer (Primary)

- **Algorithm**: Byte-Pair Encoding (BPE)
- **Vocabulary size**: 1,000 tokens
- **Character coverage**: 100%
- **Special tokens**:

| Token | ID | Symbol |
|-------|-----|--------|
| Padding | 0 | `<pad>` |
| Unknown | 1 | `<unk>` |
| Beginning of Sequence | 2 | `<s>` |
| End of Sequence | 3 | `</s>` |

- **Model file**: `data/processed/sp.model`
- **Vocabulary file**: `data/processed/sp.vocab`

### CharacterTokenizer (Alternative)

- Character-level tokenizer mapping each unique character to an integer
- Not used in the default pipeline
- Available for experimentation with smaller datasets

---

## Dataset Pipeline

**File**: `dataset.py`

```mermaid
graph LR
    TEXT[Raw Text] --> TOK[SentencePiece Encode]
    TOK --> IDS[Token ID Tensor]
    IDS --> SPLIT[90/10 Split]
    SPLIT --> TRAIN[GPTDataset train]
    SPLIT --> VAL[GPTDataset val]
    TRAIN --> DL[DataLoader<br>batch, shuffle, pin_memory]
```

**`GPTDataset`**: Implements a sliding window over the token sequence. For each index `i`, returns:
- `x = tokens[i : i + block_size]` (input)
- `y = tokens[i+1 : i + block_size + 1]` (target, shifted by one position)

This is the standard next-token prediction setup for autoregressive language models.

---

## Sampling Pipeline

**File**: `sampling.py`

The generation pipeline supports four sampling strategies:

```mermaid
graph TD
    LOGITS[Logits from model] --> METHOD{Sampling Method}

    METHOD -->|greedy| ARGMAX[argmax]
    METHOD -->|temperature| TEMP[Scale by 1/T] --> SOFTMAX1[Softmax] --> SAMPLE1[Multinomial Sample]
    METHOD -->|top_k| TOPK[Keep top K logits<br>rest → -∞] --> SOFTMAX2[Softmax] --> SAMPLE2[Multinomial Sample]
    METHOD -->|top_p| SORT[Sort by probability] --> CUMSUM[Cumulative sum] --> MASK_P[Mask above threshold p] --> SOFTMAX3[Softmax] --> SAMPLE3[Multinomial Sample]
```

| Method | Parameter | Effect |
|--------|-----------|--------|
| `greedy` | — | Always selects the highest-probability token. Deterministic. |
| `temperature` | `temperature` (0, ∞) | Values < 1.0 sharpen the distribution (more conservative). Values > 1.0 flatten it (more creative). |
| `top_k` | `top_k` (int) | Only the top K most probable tokens are considered. |
| `top_p` | `top_p` (0, 1] | Keeps the smallest set of tokens whose cumulative probability exceeds p. |

---

## Module Reference

| Module | File | Key Classes/Functions | Purpose |
|--------|------|----------------------|---------|
| `ModelConfig` | `config.py` | `ModelConfig` dataclass | Model hyperparameters |
| `TrainingConfig` | `config.py` | `TrainingConfig` dataclass | Training hyperparameters |
| `GPT2` | `model.py` | `GPT2(nn.Module)` | Top-level language model |
| `MultiHeadAttention` | `attention.py` | `MultiHeadAttention`, `ScaledDotProductAttention`, `create_causal_mask` | Attention mechanisms |
| `TransformerBlock` | `transformer.py` | `TransformerBlock` | Single transformer layer |
| `Layers` | `generate.py` | `LayerNorm`, `GELU`, `FeedForward`, `TokenEmbedding`, `LearnedPositionEmbedding`, `SinusoidalPositionEmbedding`, `ResidualBlock` | Fundamental building blocks |
| `Sampling` | `sampling.py` | `sample_next_token()` | Decoding strategies |
| `Tokenizer` | `tokenizer.py` | `SentencePieceTokenizer`, `CharacterTokenizer` | Text tokenization |
| `Dataset` | `dataset.py` | `GPTDataset`, `get_dataloader()`, `create_datasets()`, `get_batch()` | Data loading |
| `Trainer` | `trainer.py` | `Trainer` | Training loop + checkpointing |
| `Training CLI` | `train.py` | `main()`, `prepare_data()`, `download_data()` | CLI entry point |
| `Generation CLI` | `generate.py` | `main()` | CLI text generation |
| `API Server` | `app.py` | FastAPI app, `GenerateRequest`, endpoints | REST API |
| `Utilities` | `utils.py` | `seed_everything()`, `save_checkpoint()`, `load_checkpoint()`, `count_parameters()`, `create_logger()` | Helper functions |
