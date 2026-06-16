import os
import torch
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.gpt.model import GPT2
from src.gpt.tokenizer import BPETokenizer


device      = "cuda" if torch.cuda.is_available() else "cpu"
data_path   = "data/FineWebEdu/data.bin"
merges_path = "data/FineWebEdu/merges.txt"
save_path   = "models/FineWebEdu.pt"


tokenizer = BPETokenizer(
     special_tokens={"<|endoftext|>": 8000}
).load(merges_path)


data = np.memmap(data_path,dtype=np.uint16, mode="r")
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


### CONFIG
vocab_size = len(tokenizer.vocab)
batch_size = 32
n_embed    = 320
n_head     = 8
n_layer    = 6
block_size = 256
dropout    = 0.2
eval_iters = 30


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
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out 



model = GPT2(
    vocab_size,
    n_embed,
    n_head,
    n_layer,
    block_size,
    dropout
).to(device)

params_count = sum([p.numel() for p in model.parameters()])
print(f"{params_count/1e6:.2f}M Parameters")



max_steps = 40000
eval_interval = 200


max_lr = 1e-3
min_lr = 1e-4
warmup_steps = max_steps // 20 # 5% of max_steps

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=True)
ctx = torch.autocast(device, dtype=torch.bfloat16)


def get_lr(step):
    if step < warmup_steps: # Linear warmup
        return max_lr * (step+1) / warmup_steps
    ratio = (step - warmup_steps) / (max_steps - warmup_steps) # [0 - 1.0]
    # 1/2 * (1 + cos(pi*ratio)) -> [1, 0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


config = {
    "vocab_size": vocab_size,
    "n_embed"   : n_embed,
    "n_head"    : n_head,
    "n_layer"   : n_layer,
    "block_size": block_size,
    "dropout"   : dropout,
}

os.makedirs(os.path.dirname(save_path), exist_ok=True)
best_val = float("inf")

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

        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step"     : step,
                "best_val" : best_val,
                "config"   : config,
            }, save_path)

    xb, yb = get_batch(train_data, block_size, batch_size, device)

    with ctx:
        logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()