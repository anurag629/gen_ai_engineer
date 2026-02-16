"""
Day 3: Mini GPT - A Decoder-Only Transformer Language Model

A character-level language model built with the Transformer architecture.
This is a miniature version of GPT (Generative Pre-trained Transformer),
trained on the same names dataset from Day 2 for direct comparison.

Unlike Day 2's MLP which predicts only the NEXT character from a context,
Mini GPT trains on ALL positions simultaneously: given input tokens
[t0, t1, ..., t_{n-1}], it predicts [t1, t2, ..., t_n] at every position.
This is exactly how real GPT models are trained.

Run: python3 mini_gpt.py
Requires: names.txt in the same directory (symlinked from Day 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import math
import os


# ============================================================
# 1. Load Dataset & Build Vocabulary
# ============================================================

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'names.txt')
words = open(data_path, 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)  # 27

print(f"Dataset: {len(words)} names")
print(f"Vocabulary: {vocab_size} characters")


# ============================================================
# 2. Hyperparameters
# ============================================================

block_size = 32     # maximum context length
d_model = 64        # embedding dimension
n_heads = 4         # number of attention heads
n_layers = 4        # number of Transformer blocks
dropout = 0.1       # dropout rate
batch_size = 64
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
warmup_iters = 500

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================
# 3. Build Dataset (GPT-style: predict next token at every position)
# ============================================================
# Concatenate all names with '.' separators into one long stream,
# then create chunks of (block_size + 1) characters.
# Input = chunk[:-1], Target = chunk[1:]
# This gives block_size training signals per example (not just 1).

def encode(text):
    return [stoi[ch] for ch in text]


def build_data_stream(word_list):
    """Concatenate words into one long token stream: .name1.name2.name3."""
    text = '.' + '.'.join(word_list) + '.'
    return torch.tensor(encode(text), dtype=torch.long)


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_data = build_data_stream(words[:n1])
val_data = build_data_stream(words[n1:n2])
test_data = build_data_stream(words[n2:])

print(f"\nDataset (GPT-style contiguous stream):")
print(f"  Train:      {len(train_data):>7d} tokens")
print(f"  Validation: {len(val_data):>7d} tokens")
print(f"  Test:       {len(test_data):>7d} tokens")
print(f"  Block size: {block_size} (each example trains on {block_size} predictions)")


def get_batch(data, batch_size):
    """Sample a batch of (input, target) pairs from the data stream."""
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 4. Model Components
# ============================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, d_model, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention with causal mask
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and 4x expansion."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual."""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ff(self.ln2(x))     # residual + FFN
        return x


class MiniGPT(nn.Module):
    """A minimal GPT-style decoder-only Transformer."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)                                  # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))    # (T, d_model)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ============================================================
# 5. Create Model
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    block_size=block_size,
    dropout=dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nMini GPT: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, block_size={block_size}")
print(f"Parameters: {n_params:,}")


# ============================================================
# 6. Training Utilities
# ============================================================

def get_lr(step):
    """Linear warmup followed by cosine decay."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)


@torch.no_grad()
def estimate_loss(model, data, eval_iters):
    """Estimate average loss on a dataset."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ============================================================
# 7. Training Loop
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

print(f"\nTraining Mini GPT...")
print(f"{'='*60}")
print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10}")
print(f"{'-'*50}")

for step in range(max_iters):

    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Sample a batch
    x_batch, y_batch = get_batch(train_data, batch_size)
    logits, loss = model(x_batch, y_batch)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Evaluate periodically
    if step % eval_interval == 0 or step == max_iters - 1:
        train_loss = estimate_loss(model, train_data, eval_iters)
        val_loss = estimate_loss(model, val_data, eval_iters)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"{step:6d} | {train_loss:10.4f} | {val_loss:10.4f} | {lr:10.6f}")

print(f"\nTraining complete!")


# ============================================================
# 8. Final Evaluation
# ============================================================

test_loss = estimate_loss(model, test_data, eval_iters)
print(f"\n{'='*60}")
print(f"Final Results:")
print(f"  Train loss:      {train_losses[-1]:.4f}")
print(f"  Validation loss: {val_losses[-1]:.4f}")
print(f"  Test loss:       {test_loss:.4f}")


# ============================================================
# 9. Generate Names
# ============================================================

def generate_names(model, num_names=30, max_len=20, temperature=1.0):
    """Generate names from the trained model."""
    model.eval()
    names = []
    for _ in range(num_names):
        context = torch.zeros(1, 1, dtype=torch.long, device=device)  # start with '.'
        name = []
        for _ in range(max_len):
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            name.append(itos[ix])
            context = torch.cat([
                context,
                torch.tensor([[ix]], device=device)
            ], dim=1)
            # Crop to block_size if needed
            if context.shape[1] > block_size:
                context = context[:, -block_size:]
        names.append(''.join(name))
    model.train()
    return names


print(f"\n{'='*60}")
print("Generated Names (Mini GPT, temperature=1.0):")
print(f"{'='*60}")
for name in generate_names(model, num_names=30):
    if name:
        print(f"  {name}")

print(f"\nGenerated Names (temperature=0.7, more conservative):")
for name in generate_names(model, num_names=10, temperature=0.7):
    if name:
        print(f"  {name}")

print(f"\nGenerated Names (temperature=1.3, more creative):")
for name in generate_names(model, num_names=10, temperature=1.3):
    if name:
        print(f"  {name}")


# ============================================================
# 10. Plot Training Loss
# ============================================================

save_dir = os.path.dirname(os.path.abspath(__file__))

plt.figure(figsize=(10, 5))
steps_list = list(range(0, max_iters, eval_interval)) + [max_iters - 1]
plt.plot(steps_list, train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(steps_list, val_losses, 'r-', label='Val Loss', linewidth=2)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Mini GPT Training Progress', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mini_gpt_training.png'), dpi=150)
print("\nSaved mini_gpt_training.png")


# ============================================================
# 11. Compare with MLP Baseline
# ============================================================

print(f"\n{'='*60}")
print("Training MLP baseline for comparison...")
print(f"{'='*60}")

# MLP uses the Day-2-style dataset: (context window) -> next char
mlp_block_size = 8

def build_mlp_dataset(word_list, bs):
    """Build (context, target) pairs with a fixed window (Day 2 style)."""
    X, Y = [], []
    for w in word_list:
        context = [0] * bs
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context[:])
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X).to(device), torch.tensor(Y).to(device)


X_train_mlp, Y_train_mlp = build_mlp_dataset(words[:n1], mlp_block_size)
X_val_mlp, Y_val_mlp = build_mlp_dataset(words[n1:n2], mlp_block_size)


class MLPLanguageModel(nn.Module):
    """MLP language model from Day 2 (for comparison)."""

    def __init__(self, vocab_size, mlp_block_size, n_embd, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(mlp_block_size * n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, idx, targets=None):
        emb = self.emb(idx).view(idx.shape[0], -1)
        h = torch.tanh(self.fc1(emb))
        logits = self.fc2(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


mlp_model = MLPLanguageModel(
    vocab_size=vocab_size,
    mlp_block_size=mlp_block_size,
    n_embd=24,
    n_hidden=256,
).to(device)

mlp_params = sum(p.numel() for p in mlp_model.parameters())
print(f"MLP: block_size={mlp_block_size}, n_embd=24, n_hidden=256")
print(f"MLP parameters: {mlp_params:,}")

mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)
mlp_val_losses = []

for step in range(max_iters):
    ix = torch.randint(0, X_train_mlp.shape[0], (batch_size,))
    _, loss = mlp_model(X_train_mlp[ix], Y_train_mlp[ix])

    mlp_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    mlp_optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        mlp_model.eval()
        with torch.no_grad():
            mlp_val_list = []
            for _ in range(eval_iters):
                ix = torch.randint(0, X_val_mlp.shape[0], (batch_size,))
                _, vloss = mlp_model(X_val_mlp[ix], Y_val_mlp[ix])
                mlp_val_list.append(vloss.item())
            mlp_val = sum(mlp_val_list) / len(mlp_val_list)
        mlp_model.train()
        mlp_val_losses.append(mlp_val)
        print(f"  Step {step:6d} | MLP Val Loss: {mlp_val:.4f}")


# ============================================================
# 12. Plot Comparison
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(steps_list, val_losses, 'b-', label=f'Mini GPT ({n_params:,} params)', linewidth=2)
plt.plot(steps_list, mlp_val_losses, 'r-', label=f'MLP ({mlp_params:,} params)', linewidth=2)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Transformer vs MLP: Validation Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'transformer_vs_mlp.png'), dpi=150)
print("\nSaved transformer_vs_mlp.png")

print(f"\n{'='*60}")
print(f"FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Model':<25} {'Params':>10} {'Val Loss':>10}")
print(f"{'-'*47}")
print(f"{'Bigram (Day 2)':<25} {'729':>10} {'~2.45':>10}")
print(f"{'MLP (Day 2)':<25} {mlp_params:>10,} {mlp_val_losses[-1]:>10.4f}")
print(f"{'Mini GPT (Day 3)':<25} {n_params:>10,} {val_losses[-1]:>10.4f}")
print(f"\nThe Transformer achieves lower loss because it can:")
print(f"  1. Attend to ALL previous characters (not just last {mlp_block_size})")
print(f"  2. Learn WHICH characters matter via self-attention")
print(f"  3. Capture different relationship types via multiple heads")
print(f"  4. Stack deep layers via residual connections + layer norm")
