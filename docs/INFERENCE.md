# Inference Guide

This document covers how to generate text using the trained GPT-2 model — via the FastAPI server, the web playground, or the CLI.

---

## Table of Contents

- [Overview](#overview)
- [FastAPI Server](#fastapi-server)
- [API Endpoints](#api-endpoints)
- [Model Loading](#model-loading)
- [Checkpoint Loading at Startup](#checkpoint-loading-at-startup)
- [Dynamic Checkpoint Loading](#dynamic-checkpoint-loading)
- [Generation Request Flow](#generation-request-flow)
- [Tokenizer in Inference](#tokenizer-in-inference)
- [Sampling Methods](#sampling-methods)
- [Response Generation](#response-generation)
- [Frontend Interaction](#frontend-interaction)
- [CLI Generation](#cli-generation)
- [Example API Requests](#example-api-requests)

---

## Overview

The inference system supports three interfaces:

| Interface | Entry Point | Description |
|-----------|------------|-------------|
| **FastAPI Server** | `python app.py` | REST API at `http://localhost:8000` |
| **Web Playground** | `http://localhost:8000` | Interactive browser UI |
| **CLI** | `python generate.py` | Command-line text generation |

All three share the same generation pipeline: tokenize the prompt, run autoregressive generation with the selected sampling method, decode the output tokens back to text.

---

## FastAPI Server

### Starting the Server

```bash
python app.py
```

The server starts on `0.0.0.0:8000` using uvicorn. On startup, it:

1. Auto-detects the compute device (`cuda` > `mps` > `cpu`)
2. Checks for and reassembles split checkpoint archives (if applicable)
3. Finds the best available checkpoint
4. Loads the SentencePiece tokenizer from `data/processed/sp.model`
5. Loads the model from the selected checkpoint

### Server Configuration

| Setting | Value |
|---------|-------|
| Host | `0.0.0.0` (all interfaces) |
| Port | `8000` |
| ASGI Server | uvicorn |
| Static files | Served from `static/` at `/static` |

---

## API Endpoints

### `GET /` — Serve Playground

Returns the `index.html` file as an HTML response.

**Response**: HTML page (the web playground)

---

### `GET /api/model-info` — Model Information

Returns the current model's architecture details and status.

**Response (model loaded):**
```json
{
  "status": "ready",
  "embedding_dimension": 384,
  "number_of_heads": 6,
  "number_of_layers": 6,
  "block_size": 256,
  "vocab_size": 1000,
  "total_params": 11129856
}
```

**Response (no model loaded):**
```json
{
  "status": "not_trained",
  "message": "No model or checkpoint available."
}
```

---

### `GET /api/checkpoints` — List Checkpoints

Returns all `.pt` files in the `checkpoints/` directory, sorted with `checkpoint_best.pt` first, then `checkpoint_last.pt`, then step checkpoints in ascending order.

**Response:**
```json
{
  "checkpoints": [
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "checkpoint_step_500.pt",
    "checkpoint_step_1000.pt",
    "checkpoint_step_1500.pt"
  ]
}
```

---

### `POST /api/load-checkpoint` — Load a Checkpoint

Dynamically loads a different checkpoint into the running server.

**Request:**
```json
{
  "checkpoint": "checkpoint_step_1000.pt"
}
```

**Response (success):**
```json
{
  "status": "ok",
  "message": "Loaded checkpoint_step_1000.pt",
  "total_params": 11129856
}
```

**Response (file not found):**

HTTP 404:
```json
{
  "detail": "Checkpoint not found"
}
```

---

### `POST /api/generate` — Generate Text

The primary generation endpoint.

**Request Body** (`GenerateRequest` schema):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | The input text to continue |
| `max_tokens` | integer | 100 | Maximum number of tokens to generate |
| `method` | string | `"top_k"` | Sampling method: `"greedy"`, `"temperature"`, `"top_k"`, `"top_p"` |
| `temperature` | float | 0.8 | Sampling temperature (higher = more random) |
| `top_k` | integer | 50 | Number of top tokens to consider (for `top_k` method) |
| `top_p` | float | 0.9 | Cumulative probability threshold (for `top_p` method) |

**Response:**
```json
{
  "prompt": "ROMEO: ",
  "generated": "ROMEO: Pardon me, good night; I hope. BALTHASAR: Grey"
}
```

> **Note**: The `generated` field contains the full text including the original prompt, not just the new tokens.

**Error (no model loaded):**

HTTP 400:
```json
{
  "detail": "Model not loaded"
}
```

---

## Model Loading

### From Checkpoint

The model is reconstructed from the checkpoint's saved `ModelConfig`:

```
checkpoint = torch.load(path, map_location=device)
config = checkpoint["config"]         # ModelConfig dataclass
model = GPT2(config)                  # Create fresh model with matching architecture
model.load_state_dict(checkpoint["model_state"])  # Load trained weights
model.eval()                          # Set to inference mode
```

This ensures the model architecture always matches the saved weights, regardless of any changes to `config.py`.

---

## Checkpoint Loading at Startup

**File**: `app.py` — `find_startup_checkpoint()` (lines 77–91)

The startup checkpoint selection follows this priority:

1. **`checkpoint_best.pt`** — preferred if it exists
2. **Highest-step `checkpoint_step_*.pt`** — fallback if no best checkpoint

### Checkpoint Reassembly

**File**: `app.py` — `check_and_reassemble_checkpoint()` (lines 62–75)

For distribution via GitHub (which has a 100 MB file size limit), the best checkpoint can be split into zip parts:

- `checkpoints/checkpoint_best.zip.aa`
- `checkpoints/checkpoint_best.zip.ab`

On startup, if `checkpoint_best.pt` does not exist but the split zip parts do, the server:
1. Concatenates the parts into `checkpoint_best.zip`
2. Extracts `checkpoint_best.pt` from the archive
3. Cleans up the intermediate zip file

---

## Dynamic Checkpoint Loading

The `/api/load-checkpoint` endpoint allows switching models at runtime without restarting the server. This is useful for:

- Comparing outputs from different training stages
- A/B testing different checkpoints
- Loading the best checkpoint after additional training

The global `model` and `current_config` variables are updated, making the new model immediately available for subsequent `/api/generate` requests.

---

## Generation Request Flow

```
1. Receive POST /api/generate with prompt + parameters
2. Encode prompt → token IDs via SentencePieceTokenizer
3. Convert to tensor, move to device
4. Call model.generate(ids, max_tokens, method, temperature, top_k, top_p)
   └─ Autoregressive loop:
      a. Truncate input to block_size (sliding window)
      b. Forward pass → logits for last position
      c. sample_next_token(logits, method, temperature, top_k, top_p)
      d. Append sampled token to sequence
      e. Repeat for max_tokens iterations
5. Decode full token sequence → text via SentencePieceTokenizer
6. Return {"prompt": original, "generated": full_text}
```

> **Important**: Generation uses `@torch.no_grad()` to disable gradient computation, reducing memory usage and improving speed.

---

## Tokenizer in Inference

The SentencePiece tokenizer (`data/processed/sp.model`) handles text-to-tokens and tokens-to-text conversion:

```
"ROMEO: To be"  →  encode()  →  [742, 8, 47, 156]
[742, 8, 47, 156, 331, 89]  →  decode()  →  "ROMEO: To be or"
```

The tokenizer is loaded once at server startup and shared across all requests.

---

## Sampling Methods

The `sampling.py` module provides four decoding strategies:

### Greedy

```json
{"method": "greedy"}
```

Always selects the token with the highest probability. Deterministic — the same prompt always produces the same output.

### Temperature

```json
{"method": "temperature", "temperature": 0.8}
```

Scales logits by `1/temperature` before sampling:
- `temperature < 1.0` → more focused, conservative outputs
- `temperature = 1.0` → standard sampling from the model's distribution
- `temperature > 1.0` → more random, creative outputs

### Top-K

```json
{"method": "top_k", "top_k": 50, "temperature": 0.8}
```

Restricts sampling to the `top_k` most probable tokens. All other tokens have their logits set to `-inf`. Temperature scaling is applied after filtering.

### Top-P (Nucleus Sampling)

```json
{"method": "top_p", "top_p": 0.9, "temperature": 0.8}
```

Sorts tokens by probability, then keeps the smallest set whose cumulative probability exceeds `top_p`. At least one token is always kept. This dynamically adjusts the number of candidate tokens based on the model's confidence.

---

## Response Generation

### Context Window Handling

The model has a fixed context length (`block_size`). During autoregressive generation, if the accumulated sequence exceeds `block_size`, it is truncated from the left:

```
If len(sequence) > block_size:
    input = sequence[-block_size:]   # Keep only the last block_size tokens
```

This means the model can generate indefinitely, but only "sees" the most recent `block_size` tokens.

### Token-by-Token Generation

Each generation step:
1. Runs a full forward pass through all transformer layers
2. Extracts logits for the **last** position only
3. Samples the next token
4. Appends it to the sequence

There is no KV-cache — each forward pass recomputes attention for all positions. This is simpler but slower than cached generation.

---

## Frontend Interaction

### Page Load

On load, the frontend (`static/app.js`) makes two parallel API calls:
1. `GET /api/model-info` → populates the stats grid (model size, dimensions, layers, context)
2. `GET /api/checkpoints` → populates the checkpoint selector dropdown

### Checkpoint Selection

When the user selects a different checkpoint from the dropdown:
1. `POST /api/load-checkpoint` → loads the selected model
2. `GET /api/model-info` → refreshes the stats display

### Text Generation

When the user clicks "Generate":
1. The generate button is disabled and shows a loading state
2. `POST /api/generate` with the prompt and current slider values
3. The response text is displayed with a **typewriter effect**:
   - The original prompt portion is highlighted in a secondary color
   - New generated text is typed character-by-character at 10ms intervals
   - Users with `prefers-reduced-motion` see the text appear instantly
4. The generate button is re-enabled

### Controls

| Control | Range | Default | API Parameter |
|---------|-------|---------|---------------|
| Temperature slider | 0.1 – 2.0 | 0.8 | `temperature` |
| Top-K slider | 1 – 100 | 50 | `top_k` |
| Max Tokens slider | 10 – 500 | 150 | `max_tokens` |
| Checkpoint dropdown | Available checkpoints | Best/highest | `checkpoint` (via load endpoint) |

> **Note**: The frontend hardcodes `method: "top_k"` in its API requests. Other sampling methods are only accessible via direct API calls or the CLI.

---

## CLI Generation

The `generate.py` script provides command-line text generation:

```bash
python generate.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --prompt "To be, or not to be" \
  --max_tokens 150 \
  --method top_k \
  --temperature 0.8 \
  --top_k 50
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/checkpoint_best.pt` | Path to model checkpoint |
| `--tokenizer_prefix` | `data/processed/sp` | SentencePiece model prefix |
| `--prompt` | `"To be, or not to be, that is the question:"` | Input text |
| `--max_tokens` | 150 | Tokens to generate |
| `--method` | `top_k` | Sampling: `greedy`, `temperature`, `top_k`, `top_p` |
| `--temperature` | 0.8 | Temperature |
| `--top_k` | 50 | Top-K threshold |
| `--top_p` | 0.9 | Top-P threshold |
| `--out_file` | `outputs/generated.txt` | Output file |

The CLI also supports automatic checkpoint reassembly from split zip files.

---

## Example API Requests

### Generate with Top-K sampling

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ROMEO: What light through yonder window breaks?",
    "max_tokens": 100,
    "method": "top_k",
    "temperature": 0.8,
    "top_k": 50
  }'
```

### Generate with Greedy decoding

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "HAMLET: ",
    "max_tokens": 50,
    "method": "greedy"
  }'
```

### Generate with Top-P (Nucleus) sampling

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 200,
    "method": "top_p",
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

### Get model info

```bash
curl http://localhost:8000/api/model-info
```

### List available checkpoints

```bash
curl http://localhost:8000/api/checkpoints
```

### Switch checkpoint

```bash
curl -X POST http://localhost:8000/api/load-checkpoint \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "checkpoint_step_500.pt"}'
```

### Python example

```python
import requests

response = requests.post("http://localhost:8000/api/generate", json={
    "prompt": "JULIET: ",
    "max_tokens": 100,
    "method": "top_k",
    "temperature": 0.7,
    "top_k": 40
})

data = response.json()
print(data["generated"])
```
