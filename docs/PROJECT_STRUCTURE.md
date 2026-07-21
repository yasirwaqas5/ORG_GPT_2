# Project Structure

A walkthrough of every file and directory in the repository, explaining its role and responsibilities.

---

## Directory Tree

```
GPT_2_CELEBAL/
│
│── config.py                  Model + training configuration
│── model.py                   GPT-2 top-level model
│── attention.py               Attention mechanisms
│── transformer.py             Transformer block
│── generate.py                Layer modules + CLI generation
│── tokenizer.py               Tokenizer implementations
│── dataset.py                 Dataset and DataLoader utilities
│── trainer.py                 Training loop and checkpoint management
│── train.py                   CLI entry point for training
│── sampling.py                Decoding strategies
│── app.py                     FastAPI inference server
│── utils.py                   Helper utilities
│── verify_infrastructure.py   Checkpoint verification suite
│── verify_train.py            Training pipeline smoke test
│── index.html                 Web playground HTML
│── requirements.txt           Python dependencies
│── .gitignore                 Git exclusion rules
│── README.md                  Project documentation
│
├── static/
│   ├── style.css              Frontend stylesheet
│   └── app.js                 Frontend JavaScript
│
├── data/
│   ├── raw/
│   │   └── input.txt          Tiny Shakespeare dataset
│   └── processed/
│       ├── sp.model           SentencePiece BPE model
│       ├── sp.vocab           SentencePiece vocabulary
│       ├── train.pt           Tokenized training data
│       └── val.pt             Tokenized validation data
│
├── checkpoints/               Saved model checkpoints
├── logs/                      Training log files
├── outputs/                   Generated text samples
└── docs/                      Documentation
```

---

## Core Source Files

### `config.py` — Configuration

**Purpose**: Defines all model and training hyperparameters as Python dataclasses.

| Class | Description |
|-------|-------------|
| `ModelConfig` | Architecture parameters: `vocab_size`, `block_size`, `embedding_dimension`, `number_of_heads`, `number_of_layers`, `dropout` |
| `TrainingConfig` | Training parameters: `batch_size`, `learning_rate`, `weight_decay`, `max_iterations`, `warmup_steps`, `evaluation_interval`, `gradient_clip`, device selection, AMP, gradient accumulation, file paths |

Both classes provide a `to_dict()` method for serialization into checkpoints.

**Key defaults**:
- Mini model: 192 embed, 4 heads, 4 layers, 128 context (~2M params)
- Peak learning rate: 4e-4 with cosine decay
- Effective batch size: 64 (16 × 4 accumulation steps)

---

### `model.py` — GPT-2 Model

**Purpose**: The top-level language model, assembling all components into a complete GPT-2.

| Component | Description |
|-----------|-------------|
| `GPT2.__init__()` | Builds the model: token embedding, positional embedding, N transformer blocks, final LayerNorm, LM head with weight tying |
| `GPT2.forward()` | Forward pass: embeddings → transformer blocks → LayerNorm → logits. Computes cross-entropy loss with label smoothing if targets provided |
| `GPT2.generate()` | Autoregressive generation loop. Truncates to `block_size`, gets logits, samples next token, repeats |

**Weight initialization**: `nn.Linear` and `nn.Embedding` weights use Normal(0, 0.02). Output projection weights (`c_proj`) are scaled by `1/√(2N)` where N is the number of layers.

---

### `attention.py` — Attention Mechanisms

**Purpose**: Implements the attention computations used inside transformer blocks.

| Component | Description |
|-----------|-------------|
| `create_causal_mask()` | Creates a lower-triangular boolean mask to prevent attending to future positions |
| `ScaledDotProductAttention` | Computes `softmax(QK^T/√d) × V` with masking and dropout |
| `SingleHeadAttention` | Reference implementation with separate Q/K/V projections (not used in the model) |
| `MultiHeadAttention` | Production implementation with combined QKV projection, multi-head splitting, causal masking, and output projection |

---

### `transformer.py` — Transformer Block

**Purpose**: A single Pre-LN transformer layer combining attention and MLP sublayers.

| Component | Description |
|-----------|-------------|
| `TransformerBlock` | Pre-LN block: `x = x + attn(ln_1(x))` then `x = x + mlp(ln_2(x))` |

Each block contains: LayerNorm × 2, MultiHeadAttention × 1, FeedForward × 1.

---

### `generate.py` — Layer Modules + CLI Generation

**Purpose**: Dual-purpose file containing fundamental building-block modules and a CLI text generation script.

**Modules defined** (lines 1–93):

| Component | Description |
|-----------|-------------|
| `LayerNorm` | Custom implementation with learnable weight/bias, `eps=1e-5` |
| `GELU` | Gaussian Error Linear Unit (tanh approximation) |
| `FeedForward` | Two-layer MLP with 4× expansion: Linear → GELU → Linear → Dropout |
| `ResidualBlock` | Helper for `x + sublayer(x)` (available but not directly used) |
| `TokenEmbedding` | Wraps `nn.Embedding` for token-to-vector lookup |
| `LearnedPositionEmbedding` | Learned position embeddings (used by default) |
| `SinusoidalPositionEmbedding` | Static sinusoidal embeddings (available but not used) |

**CLI generation** (lines 98–181): `main()` function that loads a checkpoint and generates text from the command line with configurable sampling parameters.

---

### `tokenizer.py` — Tokenizers

**Purpose**: Text-to-token-ID conversion and back.

| Component | Description |
|-----------|-------------|
| `CharacterTokenizer` | Maps each unique character to an integer. Not used in the default pipeline, available for experimentation |
| `SentencePieceTokenizer` | Wraps Google SentencePiece with BPE. Trains with `character_coverage=1.0`, `vocab_size=1000`. Special tokens: `<pad>` (0), `<unk>` (1), `<s>` (2), `</s>` (3) |

Both tokenizers share the same interface: `train()`, `load()`, `save()`, `encode()`, `decode()`, `batch_encode()`, `batch_decode()`, `vocab_size()`.

---

### `dataset.py` — Data Loading

**Purpose**: Dataset and DataLoader creation for training.

| Component | Description |
|-----------|-------------|
| `GPTDataset` | PyTorch Dataset implementing a sliding window: `x[i:i+block_size]` → `y[i+1:i+block_size+1]` |
| `create_datasets()` | Splits a token ID tensor into train/val datasets (90/10 split) |
| `get_dataloader()` | Creates a DataLoader with shuffling, pinned memory, and optional seeded generator |
| `get_batch()` | Random batch sampling helper (alternative to DataLoader-based iteration) |

---

### `trainer.py` — Training Loop

**Purpose**: The main training orchestrator handling the forward/backward loop, evaluation, checkpoint management, and early stopping.

| Component | Description |
|-----------|-------------|
| `Trainer.__init__()` | Sets up optimizer (AdamW), DataLoaders, AMP context, RNG generator |
| `Trainer.get_learning_rate()` | Computes LR for a given step (linear warmup + cosine decay) |
| `Trainer.restore_checkpoint()` | Loads model/optimizer state, step counter, RNG states from a checkpoint |
| `Trainer.estimate_loss()` | Evaluates average loss over 30 batches from train and val sets |
| `Trainer.train()` | Main loop: gradient accumulation → clip → step → log → eval → checkpoint → early stop |
| `Trainer._save_evaluation_checkpoints()` | Saves best/last/step checkpoints, manages patience counter |
| `Trainer._checkpoint_state()` | Assembles the full checkpoint dictionary |
| `Trainer._rng_state()` | Captures all RNG states (Python, NumPy, PyTorch, CUDA) |
| `Trainer._restore_rng_state()` | Restores all RNG states from a checkpoint |

---

### `train.py` — Training CLI Entry Point

**Purpose**: Command-line interface for starting, resuming, and evaluating training.

| Component | Description |
|-----------|-------------|
| `download_data()` | Downloads Tiny Shakespeare from Karpathy's char-rnn repository |
| `parse_args()` | Parses CLI arguments (`--resume`, `--from-best`, `--evaluate`, etc.) |
| `load_checkpoint()` | Loads a checkpoint file with `weights_only=False` |
| `training_config_from_checkpoint()` | Reconstructs `TrainingConfig` from a checkpoint's saved dict |
| `prepare_data()` | Orchestrates data pipeline: download → tokenize → cache |
| `apply_cli_overrides()` | Applies CLI flag overrides to the training configuration |
| `evaluate_checkpoint()` | Standalone evaluation of a checkpoint |
| `main()` | Entry point: resolve mode → prepare data → create/load model → train or evaluate |

---

### `sampling.py` — Decoding Strategies

**Purpose**: Implements next-token sampling for text generation.

| Component | Description |
|-----------|-------------|
| `sample_next_token()` | Dispatches to one of four sampling strategies based on the `method` parameter |

Strategies: `greedy` (argmax), `temperature` (scaled softmax sampling), `top_k` (top-K filtering + sampling), `top_p` (nucleus/top-P filtering + sampling).

---

### `app.py` — FastAPI Server

**Purpose**: REST API for inference and the web playground.

| Component | Description |
|-----------|-------------|
| `check_and_reassemble_checkpoint()` | Reassembles `checkpoint_best.pt` from split zip parts if needed |
| `find_startup_checkpoint()` | Finds the best checkpoint to load at startup |
| `GenerateRequest` | Pydantic model for generation request validation |
| `GET /` | Serves the web playground (`index.html`) |
| `GET /api/model-info` | Returns model architecture details |
| `GET /api/checkpoints` | Lists available checkpoint files |
| `POST /api/load-checkpoint` | Loads a different checkpoint at runtime |
| `POST /api/generate` | Generates text from a prompt |

Static files are served from the `static/` directory at the `/static` URL prefix.

---

### `utils.py` — Utilities

**Purpose**: Shared helper functions used across the project.

| Component | Description |
|-----------|-------------|
| `seed_everything()` | Seeds Python `random`, NumPy, PyTorch CPU, and CUDA RNGs |
| `count_parameters()` | Counts trainable parameters in a model |
| `print_model_summary()` | Prints a formatted model architecture summary |
| `save_checkpoint()` | Atomic checkpoint saving (temp file → flush → fsync → replace) |
| `load_checkpoint()` | Loads a checkpoint with device mapping |
| `create_logger()` | Creates a configured logger with file and console handlers |

---

### `verify_infrastructure.py` — Verification Suite

**Purpose**: End-to-end tests for the checkpoint system.

| Test | Description |
|------|-------------|
| `verify_legacy_checkpoint_load` | Loads a legacy checkpoint, verifies weights match (SHA-256), optimizer state, step counter, best val loss, LR schedule |
| `verify_checkpoint_system` | Runs a fresh 25-step training, verifies all checkpoint files exist with correct metadata, resumes to step 35, verifies historical preservation |
| `verify_overwrite_protection` | Verifies `FileExistsError` when attempting to overwrite an existing step checkpoint |

---

### `verify_train.py` — Training Smoke Test

**Purpose**: Quick end-to-end training validation (15 iterations in a temp directory).

Verifies: checkpoint files are created, production checkpoints are not modified.

---

## Frontend Files

### `index.html` — Web Playground

**Purpose**: Interactive HTML page for text generation.

Features:
- Sidebar with checkpoint selector, temperature slider (0.1–2.0), top-k slider (1–100), max tokens slider (10–500)
- Main area with prompt textarea, Generate/Clear buttons, output display with copy functionality
- Stats grid showing Model Size, Embedding Dim, Layers/Heads, Context Size
- Accessibility features: `aria-live`, `aria-busy`, screen-reader-only labels
- Loads Inter and JetBrains Mono fonts from Google Fonts

### `static/style.css` — Stylesheet

**Purpose**: Dark glassmorphism theme with CSS custom properties.

Features:
- Dark theme with emerald (#10b981), cyan (#06b6d4), and purple (#8b5cf6) accents
- Glassmorphism effects (backdrop-filter, semi-transparent backgrounds)
- Responsive breakpoints at 900px and 560px
- `prefers-reduced-motion` support
- Stats grid transitions from 4-column to 2-column to 1-column

### `static/app.js` — Frontend JavaScript

**Purpose**: Client-side logic for the web playground.

Features:
- Fetches model info and checkpoint list on page load
- Handles checkpoint switching via `/api/load-checkpoint`
- Posts generation requests to `/api/generate`
- Typewriter text output effect (10ms per character)
- Prompt text highlighted in secondary color
- Copy-to-clipboard with visual feedback
- Respects `prefers-reduced-motion` for accessibility

---

## Data Directory

### `data/raw/input.txt`

**Source**: [Karpathy's char-rnn repository](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
**Size**: ~1.1 MB, ~40,000 lines
**Content**: Complete works of Shakespeare in plain text
**Download**: Automatic on first `python train.py` run

### `data/processed/sp.model` and `sp.vocab`

- **sp.model** (246 KB): Trained SentencePiece BPE model file
- **sp.vocab** (11 KB): Human-readable vocabulary with log probabilities
- **Vocab size**: 1,000 tokens
- **Training**: Automatic from `input.txt` on first run

### `data/processed/train.pt` and `val.pt`

- **train.pt** (2.8 MB): Tokenized training data (90% of the full dataset)
- **val.pt** (317 KB): Tokenized validation data (10% of the full dataset)
- **Format**: PyTorch long tensors saved with `torch.save`
- **Generation**: Automatic from `input.txt` + `sp.model` on first run

---

## Output Directories

### `checkpoints/`

Contains model checkpoint files. See [docs/CHECKPOINTS.md](CHECKPOINTS.md) for details.

Current contents:
- `checkpoint_step_500.pt` (~128 MB) — 11.13M model at step 500
- `checkpoint_step_1000.pt` (~128 MB) — 11.13M model at step 1000
- `checkpoint_step_1500.pt` (~128 MB) — 11.13M model at step 1500

### `logs/`

Contains `train.log` — append-only training log with timestamps, step metrics, and evaluation results across all training runs.

### `outputs/`

Contains `generated.txt` — sample text generation output.

---

## Configuration Files

### `requirements.txt`

Python package dependencies with minimum version constraints:
```
torch>=2.0.0
sentencepiece>=0.1.99
numpy>=1.20.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

### `.gitignore`

Excludes:
- Virtual environments (`.venv/`, `venv/`, `ENV/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Runtime outputs (`logs/`, `outputs/`)
- Large data files (`data/raw/input.txt`, `data/processed/train.pt`, `data/processed/val.pt`)
- Checkpoint files (`checkpoints/*.pt`) — except split zip parts (`checkpoint_best.zip.aa`, `.ab`) which are kept for model distribution
