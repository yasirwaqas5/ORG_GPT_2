"""Comprehensive training infrastructure verification (isolated; does not mutate production checkpoints)."""

import copy
import hashlib
import os
import shutil
import sys
import tempfile

import torch

from config import ModelConfig, TrainingConfig
from model import GPT2
from tokenizer import SentencePieceTokenizer
from train import load_checkpoint, training_config_from_checkpoint
from trainer import Trainer
from utils import count_parameters, save_checkpoint, seed_everything


def sha256_tensor_dict(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(state_dict[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def sha256_optimizer_state(optimizer_state) -> str:
    digest = hashlib.sha256()
    for key in sorted(optimizer_state.keys()):
        digest.update(repr(key).encode("utf-8"))
        value = optimizer_state[key]
        if isinstance(value, dict):
            for inner_key in sorted(value.keys()):
                digest.update(repr(inner_key).encode("utf-8"))
                for tensor in value[inner_key]:
                    if torch.is_tensor(tensor):
                        digest.update(tensor.detach().cpu().numpy().tobytes())
        elif torch.is_tensor(value):
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class VerificationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, name: str, detail: str = ""):
        self.passed.append((name, detail))

    def fail(self, name: str, detail: str):
        self.failed.append((name, detail))

    def warn(self, name: str, detail: str):
        self.warnings.append((name, detail))


def verify_legacy_checkpoint_load(result: VerificationResult) -> Trainer:
    checkpoint_path = "checkpoints/checkpoint_step_1500.pt"
    checkpoint = load_checkpoint(checkpoint_path, "cpu")

    if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
        result.fail("Legacy checkpoint structure", "Missing model_state or optimizer_state")
        return None

    model = GPT2(checkpoint["config"])
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception as exc:
        result.fail("Legacy checkpoint model load", str(exc))
        return None

    result.ok("Legacy checkpoint model load", f"params={count_parameters(model):,}")

    train_config, has_saved = training_config_from_checkpoint(checkpoint, TrainingConfig())
    if has_saved:
        result.ok("Legacy checkpoint training_config", "Present")
    else:
        result.warn(
            "Legacy checkpoint training_config",
            "Missing; resume uses current TrainingConfig defaults",
        )

    if not os.path.exists("data/processed/train.pt"):
        result.fail("Dataset availability", "data/processed/train.pt missing")
        return None

    train_ids = torch.load("data/processed/train.pt", weights_only=False)
    val_ids = torch.load("data/processed/val.pt", weights_only=False)
    trainer = Trainer(model, train_config, train_ids, val_ids)

    original_model_hash = sha256_tensor_dict(checkpoint["model_state"])
    original_opt_hash = sha256_optimizer_state(checkpoint["optimizer_state"])

    trainer.restore_checkpoint(checkpoint)

    if trainer.step != 1501:
        result.fail("Resume step", f"Expected 1501, got {trainer.step}")
    else:
        result.ok("Resume step", "Next step is 1501")

    expected_best = float(checkpoint["best_val_loss"])
    if abs(trainer.best_val_loss - expected_best) > 1e-6:
        result.fail("Best val loss restore", f"Expected {expected_best}, got {trainer.best_val_loss}")
    else:
        result.ok("Best val loss restore", f"{trainer.best_val_loss:.6f}")

    restored_model_hash = sha256_tensor_dict(trainer.model.state_dict())
    if restored_model_hash != original_model_hash:
        result.fail("Model weight restore", "Restored weights differ from checkpoint")
    else:
        result.ok("Model weight restore", "Exact match")

    restored_opt_hash = sha256_optimizer_state(trainer.optimizer.state_dict())
    if restored_opt_hash != original_opt_hash:
        result.fail("Optimizer state restore", "Restored optimizer state differs from checkpoint")
    else:
        result.ok("Optimizer state restore", "Exact match")

    lr_at_1500 = trainer.get_learning_rate(1500)
    lr_at_1501 = trainer.get_learning_rate(1501)
    if lr_at_1501 >= lr_at_1500:
        result.fail("Learning rate continuity", f"LR did not decay: {lr_at_1500:.2e} -> {lr_at_1501:.2e}")
    else:
        result.ok("Learning rate continuity", f"{lr_at_1500:.2e} -> {lr_at_1501:.2e}")

    return trainer


def verify_checkpoint_system(result: VerificationResult) -> None:
    work_dir = tempfile.mkdtemp(prefix="gpt2-verify-")
    try:
        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_config = ModelConfig(vocab_size=128, block_size=32, embedding_dimension=64, number_of_heads=2, number_of_layers=2)
        train_config = TrainingConfig(
            batch_size=4,
            max_iterations=25,
            warmup_steps=5,
            evaluation_interval=10,
            gradient_accumulation_steps=2,
            checkpoint_dir=checkpoint_dir,
            log_dir=os.path.join(work_dir, "logs"),
            seed=1234,
            device="cpu",
            use_amp=False,
        )

        torch.manual_seed(train_config.seed)
        train_ids = torch.randint(0, model_config.vocab_size, (5000,))
        val_ids = torch.randint(0, model_config.vocab_size, (1000,))

        model = GPT2(model_config)
        trainer = Trainer(model, train_config, train_ids, val_ids)
        trainer.train()

        expected_steps = {10, 20}
        for step in expected_steps:
            step_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(step_path):
                result.fail("Fresh checkpoint numbering", f"Missing {step_path}")
            else:
                ckpt = torch.load(step_path, map_location="cpu", weights_only=False)
                if ckpt["step"] != step:
                    result.fail("Fresh checkpoint metadata", f"{step_path} has step {ckpt['step']}")
                elif ckpt.get("validation_loss") is None:
                    result.fail("Fresh checkpoint metadata", f"{step_path} missing validation_loss")
                elif not isinstance(ckpt.get("training_config"), dict):
                    result.fail("Fresh checkpoint metadata", f"{step_path} missing training_config")
                else:
                    result.ok(f"Fresh checkpoint step {step}", "Metadata complete")

        last_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        if not os.path.exists(last_path):
            result.fail("checkpoint_last.pt", "Not created during fresh training")
        else:
            last_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            result.ok("checkpoint_last.pt", f"step={last_ckpt['step']}")

        if not os.path.exists(best_path):
            result.fail("checkpoint_best.pt", "Not created during fresh training")
        else:
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            result.ok("checkpoint_best.pt", f"step={best_ckpt['step']} val={best_ckpt['validation_loss']:.4f}")

        resume_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        best_before_resume = torch.load(best_path, map_location="cpu", weights_only=False)
        resumed_model = GPT2(resume_ckpt["config"])
        resumed_config, _ = training_config_from_checkpoint(resume_ckpt, train_config)
        resumed_config.max_iterations = 35
        resumed_trainer = Trainer(resumed_model, resumed_config, train_ids, val_ids)
        resumed_trainer.restore_checkpoint(resume_ckpt)
        start_step = resumed_trainer.step
        resumed_trainer.train()

        if not os.path.exists(os.path.join(checkpoint_dir, "checkpoint_step_30.pt")):
            result.fail("Resume checkpoint numbering", "Expected checkpoint_step_30.pt")
        else:
            result.ok("Resume checkpoint numbering", "checkpoint_step_30.pt created")

        historical = os.path.join(checkpoint_dir, "checkpoint_step_20.pt")
        historical_ckpt = torch.load(historical, map_location="cpu", weights_only=False)
        if historical_ckpt["step"] != 20:
            result.fail("Historical checkpoint preservation", "checkpoint_step_20.pt was overwritten")
        else:
            result.ok("Historical checkpoint preservation", "checkpoint_step_20.pt untouched")

        step_30 = torch.load(os.path.join(checkpoint_dir, "checkpoint_step_30.pt"), map_location="cpu", weights_only=False)
        best_after = torch.load(best_path, map_location="cpu", weights_only=False)
        if step_30["validation_loss"] < best_before_resume["validation_loss"]:
            if best_after["step"] != 30:
                result.fail("checkpoint_best update rule", "Best checkpoint did not advance when validation improved")
            else:
                result.ok("checkpoint_best update rule", "Updated on improvement")
        else:
            if best_after["step"] != best_before_resume["step"]:
                result.fail("checkpoint_best update rule", "Best checkpoint changed without validation improvement")
            elif abs(best_after["validation_loss"] - best_before_resume["validation_loss"]) > 1e-6:
                result.fail("checkpoint_best update rule", "Best checkpoint validation loss changed without improvement")
            else:
                result.ok("checkpoint_best update rule", "Unchanged when validation did not improve")

        result.ok("Resume start step", f"Resumed from {start_step}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def verify_overwrite_protection(result: VerificationResult) -> None:
    work_dir = tempfile.mkdtemp(prefix="gpt2-overwrite-")
    try:
        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        protected_path = os.path.join(checkpoint_dir, "checkpoint_step_100.pt")
        save_checkpoint({"step": 100}, protected_path)

        model_config = ModelConfig(vocab_size=64, block_size=16, embedding_dimension=32, number_of_heads=2, number_of_layers=1)
        train_config = TrainingConfig(
            batch_size=2,
            max_iterations=101,
            evaluation_interval=100,
            checkpoint_dir=checkpoint_dir,
            log_dir=os.path.join(work_dir, "logs"),
            device="cpu",
            use_amp=False,
            overwrite_step_checkpoints=False,
        )
        train_ids = torch.randint(0, model_config.vocab_size, (500,))
        val_ids = torch.randint(0, model_config.vocab_size, (200,))
        trainer = Trainer(GPT2(model_config), train_config, train_ids, val_ids)
        trainer.step = 100
        try:
            trainer._save_evaluation_checkpoints(100, 1.0, 0.9, 1e-4)
            result.fail("Overwrite protection", "Expected FileExistsError when step checkpoint already exists")
        except FileExistsError:
            result.ok("Overwrite protection", "Refuses to overwrite existing checkpoint_step file")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    result = VerificationResult()
    print("=" * 72)
    print("GPT-2 Training Infrastructure Verification")
    print("=" * 72)

    verify_legacy_checkpoint_load(result)
    verify_checkpoint_system(result)
    verify_overwrite_protection(result)

    print("\nPASSED")
    for name, detail in result.passed:
        suffix = f" — {detail}" if detail else ""
        print(f"  [OK] {name}{suffix}")

    if result.warnings:
        print("\nWARNINGS")
        for name, detail in result.warnings:
            print(f"  [WARN] {name} — {detail}")

    if result.failed:
        print("\nFAILED")
        for name, detail in result.failed:
            print(f"  [FAIL] {name} — {detail}")
        print(f"\nSummary: {len(result.passed)} passed, {len(result.warnings)} warnings, {len(result.failed)} failed")
        return 1

    print(f"\nSummary: {len(result.passed)} passed, {len(result.warnings)} warnings, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
