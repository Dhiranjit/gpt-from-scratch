# GPT from Scratch

A personal learning journey building neural networks from the ground up — no shortcuts, mostly from scratch.

Follows the progression from a simple character-level bigram model all the way to a GPT transformer, with MLP and WaveNet in between.

---

## The Journey

### 1. Bigram Model
A character-level bigram language model trained on names. Predicts the next character purely from the current one using a learned lookup table. The simplest possible baseline.

### 2. MLP (Makemore)
A multi-layer perceptron that takes a fixed context window of characters and predicts the next one. Introduces embeddings, hidden layers, and backpropagation through a proper neural net.

### 3. WaveNet-style Model
A deeper architecture with dilated causal convolutions, inspired by WaveNet. Increases the effective receptive field without the cost of a full transformer.

### 4. GPT
A decoder-only transformer (GPT-style) trained on the tiny Shakespeare dataset. Self-attention, positional embeddings, and multi-head attention — all implemented from scratch.

---

## Project Structure

```
deep-learning-from-scratch/
├── data/
│   ├── names.txt                   # Name dataset (makemore training)
│   ├── tiny_shakespeare.txt        # Tiny Shakespeare dataset
│   └── shakespeare_combined.txt    # Extended Shakespeare dataset
│
├── notebooks/
│   ├── 01_makemore_dev.ipynb       # Bigram model
│   ├── 02_makemore_dev.ipynb       # MLP
│   ├── 03_makemore_dev.ipynb       # MLP with BatchNorm + init improvements
│   ├── 03.1_makemore_dev.ipynb     # MLP variant / experiments
│   ├── 04_makemore_dev.ipynb       # WaveNet
│   ├── 05_makemore_dev.ipynb       # WaveNet refined
│   └── 06_gpt_dev.ipynb            # GPT on tiny Shakespeare
│
├── src/
│   ├── bigram/
│   │   ├── bigram.py               # Bigram model
│   │   ├── bigram_train.py         # Training script
│   │   └── bigram_generate.py      # Generation script
│   └── gpt/
│       ├── gpt.py                  # GPT model (transformer from scratch)
│       ├── train.py                # Training script
│       └── generate.py             # Generation script
│
├── models/
│   └── gpt_final.pt                # Saved GPT checkpoint
│
└── pyproject.toml
```

---

## Setup

```bash
pip install -e .
```

Requires Python >= 3.11. Dependencies: `numpy`, `matplotlib`, `torch`.

---

## References

- [Andrej Karpathy — makemore series](https://github.com/karpathy/makemore)
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT)
