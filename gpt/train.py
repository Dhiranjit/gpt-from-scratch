import os
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from gpt.gpt import ModelConfig, GPT


# =============================================================================
# Train Config
# =============================================================================

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    max_steps: int = 3000
    batch_size: int = 64
    eval_iters: int = 30
    lr: float = 3e-4


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


def get_batch(ds, split, tcfg: TrainConfig):
    data = ds.train if split == "train" else ds.val
    ix = torch.randint(0, len(data) - tcfg.model.block_size, (tcfg.batch_size,))
    xb = torch.stack([data[i: i+tcfg.model.block_size] for i in ix])
    yb = torch.stack([data[i+1:i+tcfg.model.block_size+1] for i in ix])
    return xb.to(tcfg.model.device), yb.to(tcfg.model.device)


# =============================================================================
# Eval
# =============================================================================

@torch.no_grad()
def estimate_loss(model, ds, tcfg: TrainConfig):
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = torch.zeros(tcfg.eval_iters, device=tcfg.model.device)
        for k in range(tcfg.eval_iters):
            xb, yb = get_batch(ds, split, tcfg)
            _, loss = model(xb, yb)
            losses[k] = loss.detach()
        results[split] = losses.mean().item()
    model.train()
    return results["train"], results["val"]


# =============================================================================
# Training
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    args = parser.parse_args()

    tcfg = TrainConfig(max_steps=args.max_steps)

    os.makedirs("models", exist_ok=True)

    ds = load_dataset()
    tcfg.model.vocab_size = ds.vocab_size
    model = GPT(tcfg.model).to(tcfg.model.device)

    param_count = sum(p.numel() for p in model.parameters())
    print("=" * 45)
    print(f"  GPT Model")
    print("=" * 45)
    print(f"  vocab_size : {tcfg.model.vocab_size}")
    print(f"  n_embed    : {tcfg.model.n_embed}")
    print(f"  n_head     : {tcfg.model.n_head}")
    print(f"  n_layers   : {tcfg.model.n_layers}")
    print(f"  block_size : {tcfg.model.block_size}")
    print(f"  dropout    : {tcfg.model.dropout}")
    print(f"  device     : {tcfg.model.device}")
    print("-" * 45)
    print(f"  parameters : {param_count:,}")
    print("=" * 45)
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    pbar = tqdm(range(tcfg.max_steps), desc="Training", unit="step")

    for i in pbar:
        xb, yb = get_batch(ds, "train", tcfg)
        
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if i % 300 == 0 or i == tcfg.max_steps - 1:
            train_loss, val_loss = estimate_loss(model, ds, tcfg)
            tqdm.write(f"step {i:5d} | train: {train_loss:.4f} | val: {val_loss:.4f}")


        # Checkpointing
        if i > 0 and i % 1000 == 0:
            torch.save({
                "model": model.state_dict(),
                "itos": ds.itos,
                "model_config": asdict(tcfg.model),
            }, f"models/gpt_{i}.pt")

    torch.save({
                "model": model.state_dict(),
                "itos": ds.itos,
                "model_config": asdict(tcfg.model),
            }, f"models/gpt_final.pt")