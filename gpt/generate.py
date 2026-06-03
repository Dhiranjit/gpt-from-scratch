import sys
import torch
import argparse
from gpt.gpt import ModelConfig, GPT


# =============================================================================
# Config
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/gpt_final.pt")
parser.add_argument("--tokens", type=int, default=500)
parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Generate
# =============================================================================

checkpoint = torch.load(args.model,weights_only=False,  map_location=device)

itos = checkpoint["itos"]
stoi = {v: k for k, v in itos.items()}

prompt = "ACT 1\n=====\n\n"
idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

cfg = ModelConfig(**checkpoint["model_config"])
cfg.device = device

model = GPT(cfg).to(device)
model.load_state_dict(checkpoint["model"])

model.eval()
idx = torch.zeros(1, 1, dtype=torch.long, device=device)

if args.stream:
    for idx_next in model.generate_stream(idx, args.tokens):
        sys.stdout.write(itos[idx_next.item()])
        sys.stdout.flush()
    print()
else:
    output = model.generate(idx, args.tokens)
    print(''.join(itos[i] for i in output[0].tolist()))
