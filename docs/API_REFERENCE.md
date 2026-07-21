# API Reference

Complete reference for the GPT-2 FastAPI inference server.

---

## Table of Contents

- [Server](#server)
- [Endpoints](#endpoints)
  - [GET / — Playground](#get----playground)
  - [GET /api/model-info — Model Information](#get-apimodel-info--model-information)
  - [GET /api/checkpoints — List Checkpoints](#get-apicheckpoints--list-checkpoints)
  - [POST /api/load-checkpoint — Load Checkpoint](#post-apiload-checkpoint--load-checkpoint)
  - [POST /api/generate — Generate Text](#post-apigenerate--generate-text)
- [Request Schemas](#request-schemas)
- [Response Schemas](#response-schemas)
- [Status Codes](#status-codes)
- [Error Responses](#error-responses)
- [Static Files](#static-files)
- [Examples](#examples)

---

## Server

| Setting | Value |
|---------|-------|
| Framework | FastAPI |
| ASGI Server | uvicorn |
| Host | `0.0.0.0` |
| Port | `8000` |
| Base URL | `http://localhost:8000` |
| Title | "GPT-2 Playground" |

### Starting the Server

```bash
python app.py
```

---

## Endpoints

---

### `GET /` — Playground

Serves the interactive web playground.

**Response**: HTML page (`index.html`)

**Content-Type**: `text/html`

```bash
curl http://localhost:8000/
```

---

### `GET /api/model-info` — Model Information

Returns the currently loaded model's architecture details and parameter count.

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

**Status Code**: `200 OK`

```bash
curl http://localhost:8000/api/model-info
```

---

### `GET /api/checkpoints` — List Checkpoints

Returns all `.pt` checkpoint files in the `checkpoints/` directory.

**Sorting order**:
1. `checkpoint_best.pt` (first, if exists)
2. `checkpoint_last.pt` (second, if exists)
3. `checkpoint_step_*.pt` sorted by step number ascending

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

**Response (no checkpoints):**

```json
{
  "checkpoints": []
}
```

**Status Code**: `200 OK`

```bash
curl http://localhost:8000/api/checkpoints
```

---

### `POST /api/load-checkpoint` — Load Checkpoint

Loads a different checkpoint into the running server, replacing the currently loaded model.

**Request Body:**

```json
{
  "checkpoint": "checkpoint_step_1000.pt"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `checkpoint` | string | Yes | Filename of the checkpoint in `checkpoints/` |

**Response (success):**

```json
{
  "status": "ok",
  "message": "Loaded checkpoint_step_1000.pt",
  "total_params": 11129856
}
```

**Response (not found):**

HTTP `404 Not Found`:
```json
{
  "detail": "Checkpoint not found"
}
```

```bash
curl -X POST http://localhost:8000/api/load-checkpoint \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "checkpoint_step_1000.pt"}'
```

---

### `POST /api/generate` — Generate Text

Generates text continuation from a prompt using the currently loaded model.

**Request Body** (`GenerateRequest`):

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `prompt` | string | — | **Yes** | Input text to continue from |
| `max_tokens` | integer | `100` | No | Maximum number of new tokens to generate |
| `method` | string | `"top_k"` | No | Sampling method: `"greedy"`, `"temperature"`, `"top_k"`, `"top_p"` |
| `temperature` | float | `0.8` | No | Sampling temperature (must be > 0) |
| `top_k` | integer | `50` | No | Number of top tokens for top-k filtering |
| `top_p` | float | `0.9` | No | Cumulative probability threshold for nucleus sampling |

**Response (success):**

```json
{
  "prompt": "ROMEO: ",
  "generated": "ROMEO: Pardon me, good night; I hope. BALTHASAR: Grey"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | The original input prompt |
| `generated` | string | Full generated text *including* the original prompt |

**Response (no model):**

HTTP `400 Bad Request`:
```json
{
  "detail": "Model not loaded"
}
```

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

---

## Request Schemas

### `GenerateRequest`

Pydantic model defined in `app.py`:

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    method: Optional[str] = "top_k"
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
```

### Load Checkpoint Request

Plain dictionary (no Pydantic model):

```python
{"checkpoint": str}
```

---

## Response Schemas

### Model Info Response

```python
{
    "status": str,              # "ready" or "not_trained"
    "embedding_dimension": int, # (only when status="ready")
    "number_of_heads": int,
    "number_of_layers": int,
    "block_size": int,
    "vocab_size": int,
    "total_params": int
}
```

### Checkpoints Response

```python
{
    "checkpoints": List[str]    # Sorted list of .pt filenames
}
```

### Generate Response

```python
{
    "prompt": str,              # Original prompt
    "generated": str            # Full text (prompt + generated tokens)
}
```

### Load Checkpoint Response

```python
{
    "status": str,              # "ok"
    "message": str,             # Human-readable message
    "total_params": int         # Total trainable parameters
}
```

---

## Status Codes

| Code | Meaning | When |
|------|---------|------|
| `200 OK` | Success | All successful requests |
| `400 Bad Request` | Client error | Model not loaded when trying to generate |
| `404 Not Found` | Resource missing | Requested checkpoint file does not exist |
| `422 Unprocessable Entity` | Validation error | Invalid request body (missing required fields, wrong types) |

---

## Error Responses

All error responses follow FastAPI's standard format:

```json
{
  "detail": "Error description string"
}
```

For validation errors (422):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "prompt"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

---

## Static Files

Static assets are served from the `static/` directory:

| URL | File | Description |
|-----|------|-------------|
| `/static/style.css` | `static/style.css` | Glassmorphism dark theme |
| `/static/app.js` | `static/app.js` | Frontend JavaScript |

---

## Examples

### Greedy Generation

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "HAMLET: To be, or not to be", "max_tokens": 50, "method": "greedy"}'
```

```json
{
  "prompt": "HAMLET: To be, or not to be",
  "generated": "HAMLET: To be, or not to be..."
}
```

### Temperature Sampling

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 200, "method": "temperature", "temperature": 1.2}'
```

### Top-P / Nucleus Sampling

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "JULIET: O Romeo, Romeo!", "max_tokens": 150, "method": "top_p", "top_p": 0.95, "temperature": 0.9}'
```

### Minimal Request (all defaults)

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The king"}'
```

Uses defaults: `max_tokens=100`, `method="top_k"`, `temperature=0.8`, `top_k=50`, `top_p=0.9`.

### Python Client

```python
import requests

BASE = "http://localhost:8000"

# Check model status
info = requests.get(f"{BASE}/api/model-info").json()
print(f"Model status: {info['status']}")

if info["status"] == "ready":
    print(f"Parameters: {info['total_params']:,}")
    print(f"Architecture: {info['embedding_dimension']}d, "
          f"{info['number_of_heads']}h, {info['number_of_layers']}L")

# List checkpoints
ckpts = requests.get(f"{BASE}/api/checkpoints").json()
print(f"Available: {ckpts['checkpoints']}")

# Switch checkpoint
requests.post(f"{BASE}/api/load-checkpoint",
              json={"checkpoint": "checkpoint_step_1000.pt"})

# Generate text
response = requests.post(f"{BASE}/api/generate", json={
    "prompt": "KING RICHARD: ",
    "max_tokens": 100,
    "method": "top_k",
    "temperature": 0.7,
    "top_k": 40
})
print(response.json()["generated"])
```

### JavaScript (Frontend)

```javascript
async function generate(prompt) {
  const response = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      max_tokens: 150,
      method: 'top_k',
      temperature: 0.8,
      top_k: 50
    })
  });
  const data = await response.json();
  return data.generated;
}
```
