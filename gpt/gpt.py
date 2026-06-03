import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =============================================================================
# Config
# =============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = None
    n_embed: int = 192
    n_head: int = 6
    n_layers: int = 6
    block_size: int = 256
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Self-Attention
# =============================================================================

class Head(nn.Module):
    def __init__(self, head_size, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, head_size, bias=False)    # (C,head_size)
        self.query = nn.Linear(cfg.n_embed, head_size, bias=False)  # (C,head_size)
        self.value = nn.Linear(cfg.n_embed, head_size, bias=False)  # (C,head_size)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size))) # (T, T)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        # compute attention scores ("affinities")
        attention_scores = q @ k.transpose(-1, -2) * (k.shape[-1] ** -0.5) # (B,T,T)
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        attention_scores = F.softmax(attention_scores, dim=-1) # (B,T,T)
        attention_scores = self.dropout(attention_scores)
        v = self.value(x) # (B,T,head_size)
        out = attention_scores @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple self-attention blocks in parallel"""
    def __init__(self, num_heads, head_size, cfg: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C)
        out = self.dropout(self.proj(out))     # (B,T,C) @ (C,C) -> (B,T,C)
        return out

class FeedForward(nn.Module):
    """A simple MLP followed by a non linearity"""

    def __init__(self, n_embed, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(cfg.dropout),
        )


    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, cfg: ModelConfig):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, cfg)
        self.ffwd = FeedForward(n_embed, cfg)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Pre-Norm
        x = x + self.ffwd(self.ln2(x)) 
        return x

# =============================================================================
# Model
# =============================================================================

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        V, C, T = cfg.vocab_size, cfg.n_embed, cfg.block_size
        self.token_embedding    = nn.Embedding(V, C)
        self.position_embedding = nn.Embedding(T, C)

        self.blocks = nn.Sequential(*[Block(cfg.n_embed, n_head=cfg.n_head, cfg=cfg) for _ in range(cfg.n_layers)])

        self.lm_head = nn.Linear(C, V)
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.block_size = T

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)   # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None: 
            loss = None
        else:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            context = idx[:, -self.block_size:]
            out = []
            for _ in range(max_new_tokens):
                logits, _ = self(context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                out.append(idx_next)
                context = torch.cat((context, idx_next), dim=1)[:, -self.block_size:]
            return torch.cat(out, dim=1)
 

    def generate_stream(self, idx, max_new_tokens):
        with torch.no_grad():
            context = idx[:, -self.block_size:]
            for _ in range(max_new_tokens):
                logits, _ = self(context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)[:, -self.block_size:]
                yield idx_next
 