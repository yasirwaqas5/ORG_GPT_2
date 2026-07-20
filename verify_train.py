import os
import shutil
import tempfile
import torch
from config import ModelConfig, TrainingConfig
from tokenizer import SentencePieceTokenizer
from model import GPT2
from trainer import Trainer
from utils import seed_everything, print_model_summary, create_logger
from train import download_data

def verify():
    print("Starting Training Pipeline Verification...")
    
    # Use an isolated checkpoint dir to NEVER touch production checkpoints.
    work_dir = tempfile.mkdtemp(prefix="gpt2-verify-train-")
    expected_files = []
    try:
        # 1. Initialize configs
        model_config = ModelConfig()
        train_config = TrainingConfig()
        
        # Override settings for a fast verification run; isolate output paths.
        train_config.max_iterations = 15
        train_config.evaluation_interval = 10
        train_config.checkpoint_dir = os.path.join(work_dir, "checkpoints")
        train_config.log_dir = os.path.join(work_dir, "logs")
        # Use CPU to keep the smoke test deterministic and to avoid stalling GPU
        # for what is essentially an end-to-end wiring check.
        train_config.device = "cpu"
        train_config.use_amp = False
        train_config.gradient_accumulation_steps = 1
        expected_files.append(os.path.join(train_config.checkpoint_dir, "checkpoint_step_10.pt"))
        expected_files.append(os.path.join(train_config.checkpoint_dir, "checkpoint_last.pt"))
        expected_files.append(os.path.join(train_config.checkpoint_dir, "checkpoint_best.pt"))

        print(f"Device: {train_config.device}")
        print(f"Batch Size: {train_config.batch_size}")
        print(f"Gradient Accumulation Steps: {train_config.gradient_accumulation_steps}")
        print(f"AMP (autocast) enabled: {train_config.use_amp}")
        print(f"Isolated checkpoint dir: {train_config.checkpoint_dir}")
        
        # Seed environment
        seed_everything(train_config.seed)
        
        # 2. Download dataset
        download_data(train_config.dataset_raw_path)
        
        # 3. Train/Load Tokenizer
        tokenizer = SentencePieceTokenizer()
        sp_model_file = f"{train_config.sp_model_prefix}.model"
        if not os.path.exists(sp_model_file):
            print("Training SentencePiece tokenizer...")
            tokenizer.train(train_config.dataset_raw_path, train_config.sp_model_prefix, vocab_size=1000)
        else:
            tokenizer.load(sp_model_file)
            
        model_config.vocab_size = tokenizer.vocab_size()
        
        # 4. Tokenize and prepare splits
        print("Loading datasets...")
        if not os.path.exists(train_config.dataset_train_path) or not os.path.exists(train_config.dataset_val_path):
            with open(train_config.dataset_raw_path, "r", encoding="utf-8") as f:
                text = f.read()
            ids = tokenizer.encode(text)
            n = int(0.9 * len(ids))
            train_ids = torch.tensor(ids[:n], dtype=torch.long)
            val_ids = torch.tensor(ids[n:], dtype=torch.long)
            torch.save(train_ids, train_config.dataset_train_path)
            torch.save(val_ids, train_config.dataset_val_path)
        else:
            train_ids = torch.load(train_config.dataset_train_path, weights_only=False)
            val_ids = torch.load(train_config.dataset_val_path, weights_only=False)
            
        # 5. Initialize model
        print("Initializing model...")
        model = GPT2(model_config)
        
        logger = create_logger(train_config.log_dir)
        print_model_summary(model, logger)
        
        # 6. Initialize trainer and run training
        print("Starting 15 training iterations...")
        trainer = Trainer(model, train_config, train_ids, val_ids, logger=logger)
        trainer.train()
        print("Training completed.")

        # 7. Assert that all expected checkpoints were created in the isolated dir
        missing = [p for p in expected_files if not os.path.exists(p)]
        if missing:
            print(f"FAIL: Missing expected checkpoint files: {missing}")
            return 1
        # Assert production checkpoints were NOT polluted
        prod_step_1500 = "checkpoints/checkpoint_step_1500.pt"
        if os.path.exists(prod_step_1500):
            ck = torch.load(prod_step_1500, map_location="cpu", weights_only=False)
            if ck.get("step") != 1500:
                print(f"FAIL: Production checkpoint_step_1500.pt was overwritten (step={ck.get('step')})")
                return 1
        print("Verification completed successfully!")
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    import sys
    sys.exit(verify())
