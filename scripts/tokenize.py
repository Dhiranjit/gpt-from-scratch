import argparse
import numpy as np

from pathlib import Path
from src.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and encode a text dataset.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    return parser.parse_args()


def main():
    args        = parse_args()
    vocab_size  = args.vocab_size

    data_path  = args.data_path
    out_dir    = data_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(special_tokens={"<|endoftext|>": vocab_size})

    tokenizer.train(text, vocab_size, verbose=True)
    tokenizer.save(out_dir / "merges.txt")

    print("Training done!!! Encoding data...")

    ids =  tokenizer.encode(text, verbose=True)
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(out_dir / "data.bin")

    print(f"Encoding done!!! data.bin saved to {out_dir}")


if __name__ == "__main__":
    main()