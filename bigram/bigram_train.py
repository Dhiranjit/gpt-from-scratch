import os
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass

from bigram import BigramLanguageModel


# =============================================================================
# Config
# =============================================================================

MAX_STEPS = 20000
BATCH_SIZE = 32
BLOCK_SIZE = 8
EVAL_ITERS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
args = parser.parse_args()
MAX_STEPS = args.max_steps


# =============================================================================
# Data
# =============================================================================
    
@dataclass
class Dataset:
    train: torch.Tensor
    val: torch.Tensor
    itos: dict
    vocab_size: int


def load_dataset(path: Path = Path(__file__).parent.parent / "data/tiny_shakespeare.txt"):
    with open(path, 'r', encoding='UTF-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))

    return Dataset(data[:n], data[n:], itos, vocab_size)


def get_batch(ds, split, block_size, batch_size, device):
    data = ds.train if split == "train" else ds.val
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i: i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return xb.to(device), yb.to(device)


# =============================================================================
# Eval
# =============================================================================

@torch.no_grad()
def estimate_loss(model, ds, eval_iters, block_size, batch_size, device):
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(ds, split, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.detach()
        results[split] = losses.mean().item()
    model.train()
    return results["train"], results["val"]


# =============================================================================
# Training
# =============================================================================

os.makedirs("models", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset()
model = BigramLanguageModel(ds.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(MAX_STEPS):
    xb, yb = get_batch(ds, "train", BLOCK_SIZE, BATCH_SIZE, device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0 or i == MAX_STEPS - 1:
        train_loss, val_loss = estimate_loss(model, ds, EVAL_ITERS, BLOCK_SIZE, BATCH_SIZE, device)
        print(f"Steps: {i:6d}/{MAX_STEPS:6d} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

# =============================================================================
# Save
# =============================================================================

torch.save({
    "model": model.state_dict(),
    "itos": ds.itos,
    "vocab_size": ds.vocab_size,
    "block_size": BLOCK_SIZE,
}, "models/bigram.pt")