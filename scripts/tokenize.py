import numpy as np
from pathlib import Path
from datasets import load_dataset
from src.gpt.tokenizer import BPETokenizer


out_path = Path(__file__).resolve().parents[1] / "data/FineWebEdu"

vocab_size = 8000


tokenizer = BPETokenizer(
    special_tokens={"<|endoftext|>": vocab_size}
).load(out_path / "merges.txt")

fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)

tokens = []
TARGET = 50_000_000

for example in fw:
    ids = tokenizer.encode(example["text"] + "<|endoftext|>")
    tokens.extend(ids)

    if len(tokens) >= TARGET:
        break

tokens = np.array(tokens, dtype=np.uint16)
tokens.tofile(out_path / "train.bin")