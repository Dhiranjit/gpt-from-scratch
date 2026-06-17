import os
import torch
import math
import argparse
import numpy as np
from tqdm import tqdm

from src.gpt.model import GPT2, GPTConfig
from tokenizer.bpe import BPETokenizer




parser = argparse.ArgumentParser(description="Train GPT")
parser.add_argument("--dataset", required=True)
args = parser.parse_args()


device      = "cuda" if torch.cuda.is_available() else "cpu"
data_path   = f"data/{args.dataset}/data.bin"
merges_path = f"data/{args.dataset}/merges.txt"
save_path   = f"models/{args.dataset}.pt"


# Load the tokenizer
tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"]).load(merges_path)

data = np.memmap(data_path,dtype=np.uint16, mode="r")
n = int(0.9 * len(data))

# Split the data into train and val
train_data, val_data = data[:n], data[n:]


### CONFIG
eval_iters = 30


config = GPTConfig(
    block_size=256,
    vocab_size=len(tokenizer.vocab),
    n_embed=384,
    n_head=8,
    n_layer=8
)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i   : i+block_size].astype(np.int64))   for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    splits = {"train": train_data, "val": val_data}
    out = {}
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, config.block_size, batch_size, device)
            with ctx:
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out



model = GPT2(config=config).to(device)

params_count = sum(p.numel() for p in model.parameters())
print(f"{params_count/1e6:.2f}M Parameters")

if device == "cuda":
    model = torch.compile(model)

raw_model = getattr(model, "_orig_mod", model) # Unwrap compiled model for saving / generate



### TRAINING CONFIG
max_steps     = 12000
eval_interval = 500
batch_size    = 32


max_lr       = 1e-3
min_lr       = 1e-4
warmup_steps = max_steps // 20 # 5% of max_steps

use_fused = device == "cuda"
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=use_fused)
ctx = torch.autocast(device, dtype=torch.bfloat16)


def get_lr(step):
    if step < warmup_steps: # Linear warmup
        return max_lr * (step+1) / warmup_steps
    ratio = (step - warmup_steps) / (max_steps - warmup_steps) # [0 - 1.0]
    # 1/2 * (1 + cos(pi*ratio)) -> [1, 0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)



os.makedirs(os.path.dirname(save_path), exist_ok=True)

pbar = tqdm(range(max_steps), desc="GPT2 training", unit="steps")

for step in pbar:

    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss(model)
        pbar.set_postfix({
            "Train": f"{losses['train']:.4f}",
            "val"  : f"{losses['val']:.4f}",
            "lr"   : f"{lr:.2e}"
        })
        tqdm.write(
            f"step {step}: train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | lr {lr:.2e}"
        )

    xb, yb = get_batch(train_data, config.block_size, batch_size, device)

    with ctx:
        logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


# Save final model
torch.save({
    "model"  : raw_model.state_dict(),
    "config" : config,
}, save_path)