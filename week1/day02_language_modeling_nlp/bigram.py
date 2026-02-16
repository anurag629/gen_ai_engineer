"""
Day 2 - Part 1: Bigram Language Model (Counting Approach)

This builds a character-level bigram model by counting
how often each character follows another in a dataset of names.

Run: python3 bigram.py
Requires: names.txt in the same directory
Download: https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
"""

import torch
import matplotlib.pyplot as plt


# ============================================================
# 1. Load Dataset
# ============================================================

words = open('names.txt', 'r').read().splitlines()
print(f"Loaded {len(words)} names")
print(f"Examples: {words[:5]}")

# Build vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0  # special start/end token
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
print(f"Vocabulary ({vocab_size} chars): {chars}")


# ============================================================
# 2. Count Bigrams
# ============================================================

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

print(f"\nBigram count matrix shape: {N.shape}")
print(f"'a' follows '.' (start) {N[stoi['.'], stoi['a']].item()} times")
print(f"'.' follows 'a' (end)   {N[stoi['a'], stoi['.']].item()} times")


# ============================================================
# 3. Visualize Bigram Counts
# ============================================================

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(vocab_size):
    for j in range(vocab_size):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=6)
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=6)

plt.axis('off')
plt.title('Bigram Counts', fontsize=20)
plt.tight_layout()
plt.savefig('bigram_counts.png', dpi=100)
print("\nSaved bigram_counts.png")


# ============================================================
# 4. Convert to Probabilities
# ============================================================

P = (N + 1).float()  # Laplace smoothing
P = P / P.sum(dim=1, keepdim=True)

print(f"\nMost likely characters after 'a':")
probs_after_a = [(itos[j], P[stoi['a'], j].item()) for j in range(vocab_size)]
probs_after_a.sort(key=lambda x: -x[1])
for ch, prob in probs_after_a[:5]:
    print(f"  'a' -> '{ch}': {prob:.3f}")


# ============================================================
# 5. Generate Names
# ============================================================

print(f"\n{'='*40}")
print("Generated Names (Bigram):")
print(f"{'='*40}")

g = torch.Generator().manual_seed(42)

for _ in range(20):
    name = []
    ix = 0  # start with '.'

    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        name.append(itos[ix])

    print(f"  {''.join(name)}")


# ============================================================
# 6. Evaluate: Negative Log Likelihood
# ============================================================

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1

nll = -log_likelihood / n
print(f"\n{'='*40}")
print(f"Average NLL: {nll.item():.4f}")
print(f"(Random baseline would be: {-torch.log(torch.tensor(1.0/vocab_size)).item():.4f})")
