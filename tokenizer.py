import os
import json
import sentencepiece as spm
from typing import List, Union

class CharacterTokenizer:
    def __init__(self) -> None:
        self.chars: List[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def fit(self, text: str) -> None:
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def train(self, input_file: str, model_prefix: str = "", vocab_size: int = 0) -> None:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        self.fit(text)
        if model_prefix:
            self.save(f"{model_prefix}.json")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chars = data["chars"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.itos[i] for i in ids if i in self.itos])

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

    def vocab_size(self) -> int:
        return len(self.chars)


class SentencePieceTokenizer:
    def __init__(self) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.model_path: str = ""

    def fit(self, text: str) -> None:
        temp_file = "data/raw/temp_sp_train.txt"
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        model_prefix = "data/processed/sp_temp"
        self.train(temp_file, model_prefix, vocab_size=1000)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def train(self, input_file: str, model_prefix: str, vocab_size: int) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>"
        )
        self.load(f"{model_prefix}.model")

    def load(self, path: str) -> None:
        self.sp.load(path)
        self.model_path = path

    def save(self, path: str) -> None:
        if not self.model_path:
            raise ValueError("No model loaded to save.")
        if os.path.abspath(self.model_path) != os.path.abspath(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            import shutil
            shutil.copy(self.model_path, path)

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: Union[List[int], List[float]]) -> str:
        return self.sp.decode(ids)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return self.sp.encode(texts, out_type=int)

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        return self.sp.decode(batch_ids)

    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
