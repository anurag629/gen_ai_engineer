"""
Day 2 - Part 3: MLP Language Model (Bengio et al. 2003)

A character-level language model using an MLP with embeddings.
Uses a context window of N previous characters to predict the next one.
This is significantly better than the bigram model.

Run: python3 mlp_lm.py
Requires: names.txt in the same directory
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


# ============================================================
# 1. Load Dataset & Build Vocabulary
# ============================================================

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)  # 27

print(f"Dataset: {len(words)} names")
print(f"Vocabulary: {vocab_size} characters")


# ============================================================
# 2. Build Dataset with Context Window
# ============================================================

block_size = 3  # number of context characters

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# Split into train/val/test
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

print(f"\nDataset splits:")
print(f"  Train:      {X_train.shape[0]:>7d} examples")
print(f"  Validation: {X_val.shape[0]:>7d} examples")
print(f"  Test:       {X_test.shape[0]:>7d} examples")

# Show some examples
print(f"\nSample training data (context -> target):")
for i in range(5):
    ctx = ''.join(itos[ix.item()] for ix in X_train[i])
    tgt = itos[Y_train[i].item()]
    print(f"  '{ctx}' -> '{tgt}'")


# ============================================================
# 3. Define Model Parameters
# ============================================================

g = torch.Generator().manual_seed(42)

# Hyperparameters
n_embd = 10       # embedding dimension
n_hidden = 200    # hidden layer size

# Learnable parameters
C  = torch.randn((vocab_size, n_embd),            generator=g)          # embedding table
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2    # hidden layer weights
b1 = torch.randn(n_hidden,                        generator=g) * 0.01   # hidden layer bias
W2 = torch.randn((n_hidden, vocab_size),           generator=g) * 0.01   # output layer weights
b2 = torch.randn(vocab_size,                       generator=g) * 0.01   # output layer bias

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

n_params = sum(p.nelement() for p in parameters)
print(f"\nModel: Embedding({vocab_size}, {n_embd}) -> Linear({block_size*n_embd}, {n_hidden}) -> tanh -> Linear({n_hidden}, {vocab_size})")
print(f"Total parameters: {n_params:,}")


# ============================================================
# 4. Training Loop
# ============================================================

print(f"\nTraining...")
print(f"{'='*60}")

batch_size = 32
n_steps = 50000
losses_i = []

for step in range(n_steps):

    # Mini-batch
    ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    # Forward pass
    emb = C[X_batch]                                          # [B, block_size, n_embd]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)  # [B, n_hidden]
    logits = h @ W2 + b2                                       # [B, vocab_size]
    loss = F.cross_entropy(logits, Y_batch)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Learning rate with decay
    lr = 0.1 if step < 25000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    # Track loss
    losses_i.append(loss.item())
    if step % 5000 == 0:
        print(f"  Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr}")

print(f"  Step {n_steps:5d} | Loss: {loss.item():.4f}")


# ============================================================
# 5. Evaluate
# ============================================================

def evaluate(X, Y, label):
    with torch.no_grad():
        emb = C[X]
        h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
    print(f"  {label:12s} loss: {loss.item():.4f}")
    return loss.item()

print(f"\n{'='*60}")
print("Evaluation:")
train_loss = evaluate(X_train, Y_train, "Train")
val_loss = evaluate(X_val, Y_val, "Validation")
test_loss = evaluate(X_test, Y_test, "Test")
print(f"\n  Improvement over bigram (~2.45): {2.45 - val_loss:.2f}")


# ============================================================
# 6. Plot Training Loss
# ============================================================

plt.figure(figsize=(12, 5))
# Smooth the loss curve
window = 200
smoothed = []
for i in range(len(losses_i)):
    start = max(0, i - window)
    smoothed.append(sum(losses_i[start:i+1]) / (i - start + 1))

plt.plot(smoothed, 'b-', linewidth=1)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
plt.title('MLP Language Model - Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mlp_training_loss.png', dpi=150)
print("\nSaved mlp_training_loss.png")


# ============================================================
# 7. Generate Names
# ============================================================

print(f"\n{'='*60}")
print("Generated Names (MLP):")
print(f"{'='*60}")

g_sample = torch.Generator().manual_seed(42)

for _ in range(30):
    name = []
    context = [0] * block_size

    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        context = context[1:] + [ix]

        if ix == 0:
            break
        name.append(itos[ix])

    generated = ''.join(name)
    if generated:  # skip empty names
        print(f"  {generated}")


# ============================================================
# 8. Visualize Embeddings (2D projection)
# ============================================================

# Use PCA to project embeddings to 2D if n_embd > 2
if n_embd > 2:
    # Simple PCA: center, then take top 2 eigenvectors
    C_centered = C.data - C.data.mean(dim=0)
    U, S, V = torch.svd(C_centered)
    C_2d = C_centered @ V[:, :2]
else:
    C_2d = C.data

plt.figure(figsize=(8, 8))
for i in range(vocab_size):
    x, y = C_2d[i]
    plt.scatter(x.item(), y.item(), s=200, zorder=5)
    plt.text(x.item() + 0.02, y.item() + 0.02, itos[i], fontsize=14, fontweight='bold')

plt.title('Character Embeddings (PCA to 2D)', fontsize=14)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('embeddings_2d.png', dpi=150)
print("Saved embeddings_2d.png")
