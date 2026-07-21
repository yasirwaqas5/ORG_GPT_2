<p align="center">
  <h1 align="center">GPT-2: From-Scratch Transformer Language Model</h1>
  <p align="center">
    A clean, educational implementation of the GPT-2 architecture built entirely from scratch using PyTorch — with training, checkpoint management, inference API, and an interactive playground frontend.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#api-playground">API &amp; Playground</a> •
  <a href="#documentation">Docs</a>
</p>

---

## Overview

This project is a **ground-up implementation** of the GPT-2 transformer language model. Every component — attention, layer normalization, positional embeddings, feed-forward networks, the training loop, checkpoint system, tokenizer integration, and inference pipeline — is written from scratch in Python and PyTorch. No pre-trained weights or high-level model libraries are used.

The model is trained on the **Tiny Shakespeare** dataset (~1.1 MB of Shakespeare's complete works) and generates coherent Shakespearean-style text through a FastAPI backend and a glassmorphism-styled web playground.

### Motivation

- **Deep understanding**: Building every transformer component from first principles to truly understand the GPT-2 architecture.
- **Production-grade engineering**: Implementing professional checkpoint management, atomic saves, resume-from-checkpoint, early stopping, and gradient accumulation.
- **End-to-end delivery**: Not just a model — a complete system with API, frontend, and documentation.

### Project Goals

| Goal | Status |
|------|--------|
| Implement GPT-2 architecture from scratch | Done |
| Custom attention, LayerNorm, GELU, FeedForward | Done |
| SentencePiece BPE tokenizer integration | Done |
| Full training pipeline with cosine LR schedule | Done |
| Checkpoint save/resume with RNG state preservation | Done |
| Multiple sampling strategies (greedy, temperature, top-k, top-p) | Done |
| FastAPI inference server | Done |
| Interactive web playground | Done |
| Atomic checkpoint writes for crash safety | Done |
| Early stopping with patience | Done |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT-2 Model                          │
│                                                             │
│  ┌──────────────┐  ┌───────────────────────┐               │
│  │   Token       │  │  Learned Positional   │               │
│  │  Embedding    │  │     Embedding         │               │
│  └──────┬───────┘  └──────────┬────────────┘               │
│         │                     │                             │
│         └─────────┬───────────┘                             │
│                   │  (sum)                                  │
│                   ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Transformer Block  ×N                     │   │
│  │  ┌─────────┐   ┌────────────────────────────────┐   │   │
│  │  │LayerNorm│──▶│  Multi-Head Causal Attention   │   │   │
│  │  └─────────┘   └──────────────┬─────────────────┘   │   │
│  │       │  (residual)           │                      │   │
│  │       └───────────────────────┤                      │   │
│  │                               ▼                      │   │
│  │  ┌─────────┐   ┌────────────────────────────────┐   │   │
│  │  │LayerNorm│──▶│   FeedForward (4× expansion)   │   │   │
│  │  └─────────┘   └──────────────┬─────────────────┘   │   │
│  │       │  (residual)           │                      │   │
│  │       └───────────────────────┘                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                   │                                         │
│                   ▼                                         │
│           ┌──────────────┐                                  │
│           │  LayerNorm   │                                  │
│           └──────┬───────┘                                  │
│                  ▼                                          │
│           ┌──────────────┐                                  │
│           │   LM Head    │  (weight-tied with token embed)  │
│           └──────┬───────┘                                  │
│                  ▼                                          │
│              [logits]                                       │
└─────────────────────────────────────────────────────────────┘
```

**Design choices:**
- **Pre-LN architecture**: Layer normalization is applied *before* each sublayer (not after), following the GPT-2 convention for improved training stability.
- **Weight tying**: The language modeling head shares weights with the token embedding layer, reducing parameter count.
- **Combined QKV projection**: Multi-head attention uses a single linear projection for all queries, keys, and values, then splits — matching the original GPT-2 implementation.
- **Label smoothing**: Cross-entropy loss uses `label_smoothing=0.05` to improve generalization.

---

## Repository Structure

```
GPT_2_CELEBAL/
├── config.py                 # Model and training configuration dataclasses
├── model.py                  # GPT2 nn.Module (top-level model)
├── attention.py              # Causal mask, single-head, multi-head attention
├── transformer.py            # Pre-LN TransformerBlock
├── generate.py               # Layer modules (LayerNorm, GELU, FeedForward, embeddings) + CLI generation
├── tokenizer.py              # CharacterTokenizer + SentencePieceTokenizer
├── dataset.py                # GPTDataset, DataLoader utilities
├── trainer.py                # Training loop, checkpoint save/restore, evaluation
├── train.py                  # CLI entry point for training / resume / evaluation
├── sampling.py               # Greedy / temperature / top-k / top-p decoding
├── app.py                    # FastAPI inference server
├── utils.py                  # Seeding, checkpoint I/O, logging, parameter counting
├── index.html                # Web playground frontend
├── requirements.txt          # Python dependencies
├── verify_infrastructure.py  # Checkpoint system verification suite
├── verify_train.py           # Training pipeline smoke test
│
├── static/
│   ├── style.css             # Dark glassmorphism theme
│   └── app.js                # Frontend JavaScript (API calls, typewriter effect)
│
├── data/
│   ├── raw/
│   │   └── input.txt         # Tiny Shakespeare dataset (~1.1 MB)
│   └── processed/
│       ├── sp.model           # Trained SentencePiece BPE model
│       ├── sp.vocab           # SentencePiece vocabulary (1,000 tokens)
│       ├── train.pt           # Tokenized training data
│       └── val.pt             # Tokenized validation data
│
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs
├── outputs/                  # Generated text samples
└── docs/                     # Project documentation
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    ├── INFERENCE.md
    ├── CHECKPOINTS.md
    ├── PROJECT_STRUCTURE.md
    ├── API_REFERENCE.md
    ├── DEVELOPER_GUIDE.md
    ├── TROUBLESHOOTING.md
    └── PERFORMANCE.md
```

---

## Features

### Model Architecture
- Custom **LayerNorm** with learnable weight and bias
- **GELU** activation (tanh approximation)
- **Multi-Head Causal Self-Attention** with combined QKV projection
- **FeedForward** network with 4× inner expansion
- **Pre-LN TransformerBlock** with residual connections
- **Learned positional embeddings** (sinusoidal also available)
- **Weight tying** between token embedding and LM head
- Configurable depth, width, heads, context length, and dropout

### Training Pipeline
- **AdamW** optimizer with weight decay
- **Cosine learning rate schedule** with linear warmup
- **Gradient accumulation** for effective batch size scaling
- **Gradient clipping** (max norm = 1.0)
- **Automatic Mixed Precision (AMP)** for CUDA, MPS, and CPU
- **Early stopping** with configurable patience
- **Label smoothing** (0.05) on cross-entropy loss
- Real-time logging of loss, perplexity, learning rate, gradient norm, and throughput

### Checkpoint System
- **Atomic saves** using temp files + `os.replace` to prevent corruption
- **Three checkpoint types**: `best`, `last`, and per-step historical
- **Full state preservation**: model weights, optimizer state, RNG states, training config
- **Overwrite protection** for historical step checkpoints
- **Resume training** from any checkpoint with exact reproducibility
- **Legacy checkpoint detection** with graceful fallback

### Inference & API
- **FastAPI** server with REST endpoints
- **Dynamic checkpoint loading** via API
- **Four sampling methods**: greedy, temperature, top-k, top-p (nucleus)
- **Interactive web playground** with dark glassmorphism UI
- Real-time model info display and typewriter text generation effect

### Tokenizer
- **SentencePiece BPE** tokenizer (1,000 token vocabulary)
- Character-level tokenizer also available as an alternative
- Automatic training and caching of tokenized datasets

---

## Current Capabilities

| Capability | Details |
|-----------|---------|
| Architecture | Pre-LN GPT-2 Transformer |
| Tokenizer | SentencePiece BPE (vocab = 1,000) |
| Dataset | Tiny Shakespeare (~1.1 MB, 40K lines) |
| Training | Full pipeline with resume, eval, early stopping |
| Inference | FastAPI server + web playground |
| Sampling | Greedy, temperature, top-k, top-p |
| Checkpointing | Atomic saves, 3 checkpoint types, RNG preservation |
| Device support | CUDA, MPS (Apple Silicon), CPU |

---

## Model Configurations

The codebase supports flexible model sizing. Two configurations have been trained:

| Config | Embed Dim | Heads | Layers | Context | Vocab | Params |
|--------|-----------|-------|--------|---------|-------|--------|
| **Default (Mini)** | 192 | 4 | 4 | 128 | 1,000 | **2.00M** |
| **Medium** | 384 | 6 | 6 | 256 | 1,000 | **11.13M** |
| Full GPT-2 (reference) | 768 | 12 | 12 | 1,024 | 50,257 | ~124M |

> **Note**: The `config.py` defaults use the Mini configuration. The existing checkpoints (`checkpoint_step_500/1000/1500.pt`) were trained with the Medium configuration (11.13M parameters).

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yasirwaqas5/GPT_2_CELEBAL.git
cd GPT_2_CELEBAL

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0.0 | Deep learning framework |
| `sentencepiece` | >= 0.1.99 | BPE tokenizer |
| `numpy` | >= 1.20.0 | Numerical operations |
| `fastapi` | >= 0.100.0 | REST API framework |
| `uvicorn` | >= 0.22.0 | ASGI server |
| `pydantic` | >= 2.0.0 | Data validation |
| `python-multipart` | >= 0.0.6 | Form data parsing |

> **Tip**: For GPU training, install PyTorch with CUDA support. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for platform-specific instructions.

---

## Training

### Train from Scratch

```bash
python train.py
```

This will:
1. Download the Tiny Shakespeare dataset (if not present)
2. Train or load the SentencePiece tokenizer (vocab = 1,000)
3. Tokenize and cache the dataset (90/10 train/val split)
4. Initialize a fresh GPT-2 model
5. Train for 5,000 iterations (default) with periodic evaluation and checkpointing

### Resume Training

```bash
# Resume from the last checkpoint
python train.py --resume checkpoints/checkpoint_last.pt

# Resume from the best checkpoint
python train.py --from-best

# Resume with custom max iterations
python train.py --resume checkpoints/checkpoint_step_1500.pt --max-iterations 10000
```

### Evaluate a Checkpoint

```bash
python train.py --evaluate checkpoints/checkpoint_step_1500.pt
```

### Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--resume PATH` | Resume training from a specific checkpoint | — |
| `--from-best` | Resume from `checkpoint_best.pt` | — |
| `--evaluate PATH` | Evaluate a checkpoint without training | — |
| `--save-every N` | Override evaluation/checkpoint interval (steps) | 500 |
| `--save-dir DIR` | Override checkpoint directory | `checkpoints` |
| `--max-iterations N` | Override maximum training steps | 5,000 |
| `--overwrite-checkpoints` | Allow overwriting existing step checkpoints | `False` |

> **Note**: `--resume`, `--from-best`, and `--evaluate` are mutually exclusive.

### Expected Training Behavior

Training logs every 10 steps with: loss, perplexity, learning rate, gradient norm, and tokens/second.

Evaluation runs every `evaluation_interval` steps (default: 500), saving:
- `checkpoint_last.pt` — always
- `checkpoint_best.pt` — only on validation loss improvement
- `checkpoint_step_{N}.pt` — historical snapshot

---

## Inference

### CLI Text Generation

```bash
python generate.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --prompt "To be, or not to be" \
  --max_tokens 150 \
  --method top_k \
  --temperature 0.8 \
  --top_k 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Path to model checkpoint |
| `--tokenizer_prefix` | `data/processed/sp` | SentencePiece model prefix |
| `--prompt` | `"To be, or not to be, that is the question:"` | Input prompt |
| `--max_tokens` | 150 | Maximum tokens to generate |
| `--method` | `top_k` | Sampling method: `greedy`, `temperature`, `top_k`, `top_p` |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 50 | Top-k filtering threshold |
| `--top_p` | 0.9 | Top-p (nucleus) filtering threshold |
| `--out_file` | `outputs/generated.txt` | Output file for generated text |

---

## API & Playground

### Start the Server

```bash
python app.py
```

The server starts at `http://localhost:8000`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web playground |
| `GET` | `/api/model-info` | Returns model architecture details |
| `GET` | `/api/checkpoints` | Lists available checkpoints |
| `POST` | `/api/load-checkpoint` | Loads a different checkpoint |
| `POST` | `/api/generate` | Generates text from a prompt |

### Example API Request

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ROMEO: ",
    "max_tokens": 100,
    "method": "top_k",
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9
  }'
```

**Response:**
```json
{
  "prompt": "ROMEO: ",
  "generated": "ROMEO: Pardon me, good night; I hope..."
}
```

### Web Playground

The playground at `http://localhost:8000` provides:
- Prompt input with real-time generation
- Checkpoint selector dropdown
- Temperature, top-k, and max tokens sliders
- Model statistics display (parameters, dimensions, layers, context size)
- Typewriter text output effect
- Dark glassmorphism UI with responsive design

> **Screenshot placeholder**: *Add screenshots of the playground UI to this section.*

---

## Project Workflow

```
                    ┌──────────────────┐
                    │  Raw Text Data   │
                    │  (input.txt)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  SentencePiece   │
                    │  BPE Training    │
                    │  (vocab=1000)    │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼──────┐  ┌───▼────────┐
     │  sp.model   │  │  train.pt   │  │  val.pt    │
     │  sp.vocab   │  │  (90%)      │  │  (10%)     │
     └─────────────┘  └──────┬──────┘  └───┬────────┘
                             │              │
                    ┌────────▼──────────────▼┐
                    │    Training Loop       │
                    │  (AdamW + Cosine LR)   │
                    └────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼──────┐  ┌───▼────────┐
     │  best.pt   │  │  last.pt    │  │  step_N.pt │
     └────────────┘  └─────────────┘  └────────────┘
                             │
                    ┌────────▼─────────┐
                    │  FastAPI Server   │
                    │  (app.py)         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Web Playground   │
                    │  (index.html)     │
                    └──────────────────┘
```

---

## How Checkpoints Work

The checkpoint system is designed for reliability and reproducibility:

1. **Atomic writes**: Checkpoints are first written to a temporary file, flushed to disk, then atomically renamed. This prevents corruption from interrupted saves.

2. **Three types of checkpoints**:
   - `checkpoint_best.pt` — updated only when validation loss improves
   - `checkpoint_last.pt` — updated at every evaluation interval
   - `checkpoint_step_{N}.pt` — immutable historical snapshots (overwrite-protected by default)

3. **Full state preservation**: Each checkpoint stores model weights, optimizer state, training configuration, and all RNG states (Python, NumPy, PyTorch, CUDA, DataLoader generator) for bit-exact training resumption.

4. **Legacy compatibility**: Older checkpoints without RNG state are detected and loaded with a warning; training resumes but without exact reproducibility.

For detailed checkpoint documentation, see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

---

## Known Limitations

- **Small dataset**: Tiny Shakespeare (~1.1 MB) limits what the model can learn; outputs are recognizably Shakespearean but not always coherent.
- **Small vocabulary**: BPE vocabulary of 1,000 tokens is far below production GPT-2's 50,257, limiting expressiveness.
- **CPU training speed**: On CPU, training runs at ~47–48 tokens/second (the 11.13M model takes ~57 minutes per 10 steps). GPU is strongly recommended.
- **No distributed training**: Single-GPU only; no data or model parallelism.
- **Fixed sampling method in frontend**: The web playground hardcodes `top_k` sampling; other methods are only available via the API or CLI.
- **No streaming**: Text generation returns the full result; there is no token-by-token streaming API.

---

## Future Improvements

- [ ] Add GPU-optimized training (FlashAttention, `torch.compile`)
- [ ] Implement streaming token generation via WebSocket or SSE
- [ ] Support larger datasets and vocabularies
- [ ] Add sampling method selector to the web playground
- [ ] Implement distributed training (DDP)
- [ ] Add model export (ONNX, TorchScript)
- [ ] Implement KV-cache for faster autoregressive generation
- [ ] Add perplexity-based evaluation benchmarks
- [ ] Support loading pre-trained GPT-2 weights from HuggingFace

---

## Performance Metrics

> These metrics are from actual training runs documented in `logs/train.log`.

### Medium Model (11.13M parameters)

| Metric | Value |
|--------|-------|
| Architecture | 384 embed, 6 heads, 6 layers, 256 context |
| Parameters | 11,129,856 (11.13M) |
| Training Loss (step 1500) | 2.62 |
| Validation Loss (step 1500) | 4.40 |
| Training Perplexity (step 1550) | ~16.7 |
| Throughput (CPU) | ~47–49 tokens/sec |

### Training Progression (Medium Model, 200-step run)

| Step | Loss | Perplexity | Learning Rate |
|------|------|------------|---------------|
| 0 | 7.00 | 1,099 | 0.0 |
| 50 | 5.06 | 158 | 5.64e-4 |
| 100 | 4.46 | 86 | 3.77e-4 |
| 150 | 4.27 | 72 | 1.56e-4 |
| 199 | 4.24 | 69 | 6.00e-5 |

---

## Example Outputs

From the trained 11.13M parameter model:

```
Prompt:  "ROMEO: "
Output:  "ROMEO: Pardon me, good night; I hope. BALTHASAR: Grey"
```

> **Note**: Output quality is constrained by the small dataset and vocabulary size. Longer training and larger models produce more coherent text.

---

## FAQ

**Q: Can I train the full 124M parameter GPT-2?**
A: Yes, by changing the configuration in `config.py` to `embedding_dimension=768`, `number_of_heads=12`, `number_of_layers=12`, `block_size=1024`, and `vocab_size=50257`. You will need a GPU with sufficient memory and a larger dataset with a matching tokenizer vocabulary.

**Q: How long does training take?**
A: On CPU, the 11.13M model processes ~47 tokens/second. At the default 5,000 iterations with gradient accumulation of 4, expect several days on CPU. With a modern GPU, training completes in hours.

**Q: Can I use my own dataset?**
A: Yes. Replace `data/raw/input.txt` with your text file and delete the processed files (`data/processed/train.pt`, `val.pt`). The pipeline will retrain the tokenizer and re-tokenize automatically.

**Q: What is the checkpoint split zip (`checkpoint_best.zip.aa`, `.ab`)?**
A: GitHub has a 100 MB file size limit. The best checkpoint is split into zip parts for distribution. The `app.py` and `generate.py` scripts automatically reassemble it on first run.

**Q: Why does resume training show a "legacy checkpoint" warning?**
A: Older checkpoints don't include RNG state. Training resumes correctly, but results won't be bit-for-bit identical to an uninterrupted run. New checkpoints saved during training will include full RNG state.

**Q: Can I run this on Apple Silicon (M1/M2/M3)?**
A: Yes. The code auto-detects MPS devices and uses `float16` AMP. All features work on Apple Silicon.

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the verification suite (`python verify_infrastructure.py`)
5. Commit your changes (`git commit -m "Add your feature"`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

Please see [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for coding conventions and development workflows.

---

## License

This project is provided as-is for educational purposes. See the repository for license details.

---

## Acknowledgements

- **Andrej Karpathy** — [nanoGPT](https://github.com/karpathy/nanoGPT) and the Tiny Shakespeare dataset, which inspired this implementation.
- **OpenAI** — The original [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ("Language Models are Unsupervised Multitask Learners", Radford et al., 2019).
- **Google** — [SentencePiece](https://github.com/google/sentencepiece) tokenizer library.
- **Celebal Technologies** — Project context and mentorship.
