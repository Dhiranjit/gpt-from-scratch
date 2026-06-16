from pathlib import Path
from datasets import load_dataset
from src.gpt.tokenizer import BPETokenizer

out_path = Path(__file__).resolve().parents[1] / "data/FineWebEdu"

vocab_size  = 8000
TRAIN_CHARS = 50_000_000          # ~50MB of text is plenty to learn good merges

fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)

# Pull a small sample of text to learn the merges from.
buffer, total = [], 0
for example in fw:
    buffer.append(example["text"])
    total += len(example["text"])
    if total >= TRAIN_CHARS:
        break

text = "<|endoftext|>".join(buffer)

tokenizer = BPETokenizer(special_tokens={"<|endoftext|>": vocab_size})
tokenizer.train(text, vocab_size, verbose=True)

out_path.mkdir(parents=True, exist_ok=True)
tokenizer.save(out_path / "merges.txt")

print(f"Trained on {total:,} chars. merges.txt saved to {out_path}")
