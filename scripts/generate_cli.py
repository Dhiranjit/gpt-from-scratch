import sys
import torch
import argparse

from src.gpt.model import GPT2, GPTConfig
from src.tokenizer.bpe import BPETokenizer


MAX_TOKENS = 1000   # hard safety cap; generation normally stops at <|endoftext|>


def stream(model, tokenizer, idx, eot_id, temperature):
    """Stream tokens until <|endoftext|> or the hard cap, flushing whole UTF-8 chars."""
    buffer = b""
    for idx_next in model.generate(idx, MAX_TOKENS, temperature):
        if idx_next.item() == eot_id:
            break
        buffer += tokenizer.vocab[idx_next.item()]
        try:
            text = buffer.decode("utf-8")   # succeeds once bytes form whole chars
        except UnicodeDecodeError:
            continue                         # mid-character, wait for more bytes
        sys.stdout.write(text)
        sys.stdout.flush()
        buffer = b""
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate from a trained GPT")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model_path  = f"models/{args.dataset}.pt"
    merges_path = f"data/{args.dataset}/merges.txt"

    tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"]).load(merges_path)
    eot_id    = tokenizer.stoi["<|endoftext|>"]

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model = GPT2(config=checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded {args.dataset} on {device}. Ctrl-C / empty line to quit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break

        ids = tokenizer.encode(prompt, verbose=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        sys.stdout.write(prompt)
        with torch.no_grad():
            stream(model, tokenizer, idx, eot_id, args.temperature)


if __name__ == "__main__":
    main()
