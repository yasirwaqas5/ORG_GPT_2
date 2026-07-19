import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from config import ModelConfig, TrainingConfig
from tokenizer import SentencePieceTokenizer
from model import GPT2
from utils import count_parameters

app = FastAPI(title="GPT-2 Playground")

def check_and_reassemble_checkpoint():
    checkpoint_path = "checkpoints/checkpoint_best.pt"
    if not os.path.exists(checkpoint_path):
        part1 = "checkpoints/checkpoint_best.zip.aa"
        part2 = "checkpoints/checkpoint_best.zip.ab"
        if os.path.exists(part1) and os.path.exists(part2):
            import zipfile
            print("Reassembling model checkpoint from split zip files...")
            os.makedirs("checkpoints", exist_ok=True)
            temp_zip = "checkpoints/temp_checkpoint_best.zip"
            try:
                with open(temp_zip, "wb") as f_out:
                    for part in [part1, part2]:
                        with open(part, "rb") as f_in:
                            f_out.write(f_in.read())
                with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                    zip_ref.extractall(".")
                print("Model checkpoint successfully reassembled.")
            except Exception as e:
                print(f"Error reassembling checkpoint: {e}")
            finally:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)

check_and_reassemble_checkpoint()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Web server using device: {device}")

model = None
tokenizer = SentencePieceTokenizer()
model_config = None

checkpoint_path = "checkpoints/checkpoint_best.pt"
sp_model_path = "data/processed/sp.model"

if os.path.exists(sp_model_path):
    tokenizer.load(sp_model_path)
else:
    print(f"Warning: Tokenizer not found at {sp_model_path}")

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint['config']
        model = GPT2(model_config)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        print("Successfully loaded model checkpoint.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
else:
    print(f"Warning: Model checkpoint not found at {checkpoint_path}")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    method: Optional[str] = "top_k"
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9


@app.get("/api/checkpoints")
def list_checkpoints():
    if not os.path.exists("checkpoints"):
        return {"checkpoints": []}
    files = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
    
    def sort_key(f):
        if f == "checkpoint_best.pt":
            return (0, 0)
        if f == "checkpoint_last.pt":
            return (0, 1)
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            try:
                step = int(f.split("_")[-1].split(".")[0])
                return (1, step)
            except ValueError:
                pass
        return (2, f)
        
    files.sort(key=sort_key)
    return {"checkpoints": files}


@app.post("/api/load-checkpoint")
def load_checkpoint_api(payload: dict):
    global model, model_config
    filename = payload.get("checkpoint")
    if not filename:
        raise HTTPException(status_code=400, detail="Checkpoint filename is required")
        
    path = os.path.join("checkpoints", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found at {path}")
        
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_config = checkpoint['config']
        model = GPT2(model_config)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        return {"status": "success", "message": f"Successfully loaded checkpoint: {filename}", "total_params": count_parameters(model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")


@app.get("/api/model-info")
def get_model_info():
    if model is None:
        return {
            "status": "not_trained",
            "message": "Model checkpoint not found. Please run train.py first."
        }
    
    return {
        "status": "ready",
        "embedding_dimension": model_config.embedding_dimension,
        "number_of_heads": model_config.number_of_heads,
        "number_of_layers": model_config.number_of_layers,
        "block_size": model_config.block_size,
        "vocab_size": model_config.vocab_size,
        "total_params": count_parameters(model)
    }


@app.post("/api/generate")
def generate(request: GenerateRequest):
    global model, tokenizer
    if model is None:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model_config = checkpoint['config']
                model = GPT2(model_config)
                model.load_state_dict(checkpoint['model_state'])
                model.to(device)
                model.eval()
                if not tokenizer.model_path and os.path.exists(sp_model_path):
                    tokenizer.load(sp_model_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading model checkpoint: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Model is not trained yet. Run train.py first.")

    try:
        prompt_ids = tokenizer.encode(request.prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = model.generate(
            idx=x,
            max_new_tokens=request.max_tokens,
            method=request.method,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        output_text = tokenizer.decode(generated_ids[0].tolist())
        return {"prompt": request.prompt, "generated": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
