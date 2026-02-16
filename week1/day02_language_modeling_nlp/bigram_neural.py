"""
Day 2 - Part 2: Bigram Language Model (Neural Network Approach)

Same bigram model as Part 1, but implemented as a neural network.
This proves the counting approach and the neural approach converge
to the same result, and sets up the framework for the MLP.

Run: python3 bigram_neural.py
Requires: names.txt in the same directory
"""

import torch
import torch.nn.functional as F


# ============================================================
# 1. Load Dataset & Build Vocabulary
# ============================================================

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)


# ============================================================
# 2. Create Training Data (Bigram Pairs)
# ============================================================

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.shape[0]
print(f"Training examples: {num_examples}")

# One-hot encode inputs
xenc = F.one_hot(xs, num_classes=vocab_size).float()
print(f"One-hot shape: {xenc.shape}")


# ============================================================
# 3. Initialize Neural Network (Single Layer)
# ============================================================

g = torch.Generator().manual_seed(42)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)


# ============================================================
# 4. Training Loop
# ============================================================

print(f"\nTraining neural bigram model...")
print(f"{'='*50}")

for epoch in range(200):
    # Forward pass
    logits = xenc @ W                                  # raw scores
    counts = logits.exp()                              # positive "counts"
    probs = counts / counts.sum(dim=1, keepdim=True)   # probabilities (softmax)

    # Loss: NLL + regularization
    loss = -probs[torch.arange(num_examples), ys].log().mean()
    loss = loss + 0.01 * (W ** 2).mean()  # L2 regularization

    # Backward pass
    W.grad = None
    loss.backward()

    # Update
    W.data -= 50.0 * W.grad

    if epoch % 20 == 0 or epoch == 199:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")
print("(Should be close to the counting model's NLL of ~2.45)")


# ============================================================
# 5. Generate Names
# ============================================================

print(f"\n{'='*50}")
print("Generated Names (Neural Bigram):")
print(f"{'='*50}")

g = torch.Generator().manual_seed(42)

for _ in range(20):
    name = []
    ix = 0

    while True:
        xenc_single = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc_single @ W
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()

        if ix == 0:
            break
        name.append(itos[ix])

    print(f"  {''.join(name)}")
