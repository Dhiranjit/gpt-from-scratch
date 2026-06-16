"""Interactive text-completion CLI for the trained FineWebEdu GPT2 checkpoint.

Type a prompt, the model autocompletes it, streaming tokens as they are
sampled. Ctrl-C / Ctrl-D to quit.

    python scripts/generate_cli.py
    python scripts/generate_cli.py --temperature 0.8
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# `scripts/tokenize.py` shadows the stdlib `tokenize` module (imported by torch
# via linecache). Drop the script's own dir from sys.path, put the repo root on
# instead so `from src...` resolves. Must run before importing torch.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.gpt.model import GPT2
from src.gpt.tokenizer import BPETokenizer


DEFAULT_MODEL  = "models/FineWebEdu.pt"
DEFAULT_MERGES = "data/FineWebEdu/merges.txt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Loading
# ===========================================================================


def load(model_path, merges_path, device):
    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt["config"]

    # Training used special_tokens={"<|endoftext|>": vocab_size} and made
    # (vocab_size - 256) merges, so the special id is the last vocab slot.
    eot_id    = config["vocab_size"] - 1
    tokenizer = BPETokenizer(
        special_tokens={"<|endoftext|>": eot_id}
    ).load(merges_path)

    model = GPT2(**config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, tokenizer, config, eot_id


# ===========================================================================
# Generation
# ===========================================================================


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature):
    """Autoregressively sample token ids, yielding one at a time."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]          # crop to the context window
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # (B, vocab) for the last step

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx     = torch.cat((idx, next_id), dim=1)
        yield next_id.item()


def byte_streamer(tokenizer):
    """Return a push(token_id) -> str that buffers bytes until UTF-8 valid.

    A single token can be a partial multi-byte character; we hold those bytes
    back until they form a decodable string so we never print a broken glyph.
    """
    buf = b""

    def push(token_id):
        nonlocal buf
        buf += tokenizer.vocab[token_id]
        try:
            text = buf.decode("utf-8")
        except UnicodeDecodeError:
            return ""        # incomplete character, wait for the next token
        buf = b""
        return text

    return push


# ===========================================================================
# REPL
# ===========================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="path to a .pt checkpoint")
    p.add_argument("--merges", default=DEFAULT_MERGES, help="path to merges.txt")
    p.add_argument("--max-new-tokens", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    model, tokenizer, config, eot_id = load(args.model, args.merges, DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.model} ({n_params/1e6:.1f}M params) on {DEVICE}")
    print(f"merges: {args.merges} | block_size {config['block_size']} | "
          f"temp {args.temperature}")
    print("Enter a prompt to autocomplete. Ctrl-C / Ctrl-D to quit.")

    while True:
        try:
            prompt = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue

        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        push = byte_streamer(tokenizer)
        sys.stdout.write(prompt)
        sys.stdout.flush()

        try:
            for tid in generate(model, idx, args.max_new_tokens,
                                config["block_size"], args.temperature):
                if tid == eot_id:
                    break
                sys.stdout.write(push(tid))
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass  # let the user cut a long generation short
        print()


if __name__ == "__main__":
    main()
