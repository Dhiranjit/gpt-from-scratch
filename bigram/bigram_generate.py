import torch
import argparse

from bigram import BigramLanguageModel


# =============================================================================
# Config
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=500)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Generate
# =============================================================================

checkpoint = torch.load("models/bigram.pt", map_location=device)

itos = checkpoint["itos"]
decode = lambda l: ''.join([itos[i] for i in l])

model = BigramLanguageModel(checkpoint["vocab_size"]).to(device)
model.load_state_dict(checkpoint["model"])

model.eval()
idx = torch.zeros(1, 1, dtype=torch.long, device=device)
output = model.generate(idx, args.tokens)
print(decode(output[0].tolist()))