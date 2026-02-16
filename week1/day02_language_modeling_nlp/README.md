# Day 2: Language Modeling & NLP Basics

## The Book - From Counting Characters to Neural Language Models

> **What you need:** Python, PyTorch (pip install torch), matplotlib
> **What you'll build today:** A character-level name generator using bigrams, neural nets, and an MLP
> **Time:** ~10 hours

---

## Table of Contents

1. [Why Language Modeling Matters](#1-why-language-modeling-matters)
2. [The Big Picture](#2-the-big-picture)
3. [Chapter 1: What is a Language Model?](#chapter-1-what-is-a-language-model)
4. [Chapter 2: The Dataset - Names](#chapter-2-the-dataset---names)
5. [Chapter 3: Bigram Language Model (Counting)](#chapter-3-bigram-language-model-counting)
6. [Chapter 4: Probability and Sampling](#chapter-4-probability-and-sampling)
7. [Chapter 5: Evaluating with Loss (Negative Log Likelihood)](#chapter-5-evaluating-with-loss-negative-log-likelihood)
8. [Chapter 6: Bigram as a Neural Network](#chapter-6-bigram-as-a-neural-network)
9. [Chapter 7: One-Hot Encoding & Embeddings](#chapter-7-one-hot-encoding--embeddings)
10. [Chapter 8: The MLP Language Model (Bengio et al.)](#chapter-8-the-mlp-language-model-bengio-et-al)
11. [Chapter 9: Training the MLP](#chapter-9-training-the-mlp)
12. [Chapter 10: Generating Names & Experiments](#chapter-10-generating-names--experiments)
13. [Exercises & Projects](#exercises--projects)
14. [References & Next Steps](#references--next-steps)
15. [Interview Prep: Key Terms & Concepts](#interview-prep-key-terms--concepts-for-day-2)

---

## 1. Why Language Modeling Matters

Every large language model (GPT, Claude, Gemini, LLaMA) is a language model at its core. They predict **the next token** given the previous tokens. That's it.

Today you'll build the exact same thing, just at the character level instead of the token level, and on names instead of the entire internet. The principles are identical:

```
GPT-4:       "The capital of France is" -> predicts "Paris"
Your model:  "ann" -> predicts "a" (to make "anna")
```

## 2. The Big Picture

We'll build 3 progressively better models today:

```
Model 1: Bigram (counting)    - "given the last character, what's next?" (no neural net)
Model 2: Bigram (neural net)  - same thing, but learned by a neural network
Model 3: MLP                  - "given the last N characters, what's next?" (deep learning)
```

Each model is strictly better than the previous one.

---

## Chapter 1: What is a Language Model?

A language model assigns a **probability** to a sequence of tokens. It answers: "How likely is this sequence?"

```
P("anna")   = high   (common name)
P("xqzw")   = low    (not a name)
P("sarah")  = high   (common name)
```

Equivalently, it predicts the **next token** given the previous ones:

```
P(next_char | previous_chars)

P('n' | 'a')      = ? (bigram: only looks at 1 previous char)
P('n' | 'a','n')   = ? (trigram: looks at 2 previous chars)
P('a' | 'a','n','n') = ? (4-gram: looks at 3 previous chars)
```

### Autoregressive Generation

To generate a name, we predict one character at a time:

```
1. Start with special <START> token
2. P(? | <START>) -> sample 'a'
3. P(? | 'a')     -> sample 'n'
4. P(? | 'n')     -> sample 'n'
5. P(? | 'n')     -> sample 'a'
6. P(? | 'a')     -> sample <END>
7. Result: "anna"
```

This is called **autoregressive** generation - each prediction feeds into the next.

---

## Chapter 2: The Dataset - Names

We'll use a dataset of names. Let's set it up:

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Download the names dataset
# You can get it from: https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
# Or create your own list of names

# Read the names
words = open('names.txt', 'r').read().splitlines()

print(f"Number of names: {len(words)}")    # ~32,000
print(f"First 10: {words[:10]}")
print(f"Shortest: {min(len(w) for w in words)}")
print(f"Longest: {max(len(w) for w in words)}")

# What characters appear?
chars = sorted(list(set(''.join(words))))
print(f"Characters: {''.join(chars)}")       # a-z
print(f"Vocabulary size: {len(chars)}")      # 26
```

### Building the Vocabulary

We need to map characters to numbers and back:

```python
# Character to index mapping
# We add a special '.' character for start/end of name
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0  # special start/end token

# Index to character mapping
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(stoi)  # 27 (26 letters + 1 special token)
print(f"Vocabulary: {stoi}")
print(f"Vocab size: {vocab_size}")
```

**Why a special token?** We need to know when a name starts and ends. The `.` token signals both. So "emma" becomes `.emma.` in our model's view.

---

## Chapter 3: Bigram Language Model (Counting)

A **bigram** model looks at pairs of consecutive characters. For every pair (char1, char2), it counts how often char2 follows char1.

### Step 1: Count All Bigrams

```python
# Create a 27x27 count matrix
# N[i][j] = how many times character j follows character i
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']  # add start/end tokens
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print(f"'a' follows '.' (start) {N[stoi['.'], stoi['a']].item()} times")
print(f"'n' follows 'a' {N[stoi['a'], stoi['n']].item()} times")
```

### Step 2: Visualize the Counts

```python
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(vocab_size):
    for j in range(vocab_size):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=6)
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=6)

plt.axis('off')
plt.title('Bigram Counts', fontsize=16)
plt.tight_layout()
plt.savefig('bigram_counts.png', dpi=150)
plt.show()
```

This heatmap reveals patterns:
- Names often start with 'a', 'j', 'm', 's' (high counts in row '.')
- 'q' is almost always followed by 'u'
- Some pairs never appear (count = 0)

---

## Chapter 4: Probability and Sampling

### Converting Counts to Probabilities

To use these counts for generation, we convert each row to a probability distribution:

```python
# Convert counts to probabilities
# Add smoothing (+1) to avoid zero probabilities
P = (N + 1).float()  # +1 smoothing (Laplace smoothing)
P = P / P.sum(dim=1, keepdim=True)  # normalize each row to sum to 1

# Check: each row should sum to 1.0
print(f"Row 0 sums to: {P[0].sum().item():.4f}")  # 1.0000

# What follows 'a'?
print(f"\nProbabilities after 'a':")
for j in range(vocab_size):
    if P[stoi['a'], j] > 0.03:  # only show likely ones
        print(f"  'a' -> '{itos[j]}': {P[stoi['a'], j].item():.3f}")
```

### Why Smoothing?

Without smoothing, if a bigram never appeared in training data, its probability is 0. That means if we ever encounter it, our model says "impossible!" and the loss becomes infinity. Adding 1 to all counts (Laplace smoothing) prevents this.

### Sampling from the Model

```python
# Generate names using the bigram model
g = torch.Generator().manual_seed(42)

for _ in range(10):
    name = []
    ix = 0  # start with '.'

    while True:
        # Get probability distribution for next character
        p = P[ix]

        # Sample from the distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        # Check for end token
        if ix == 0:
            break

        name.append(itos[ix])

    print(''.join(name))
```

Output will be something like:
```
junide
janasah
p
cony
a
nn
kohin
tol
q
ry
```

Not great! The names look somewhat name-like but many are bad. That's because bigrams only look at ONE previous character - not enough context.

---

## Chapter 5: Evaluating with Loss (Negative Log Likelihood)

How do we measure how good our model is? We use **negative log likelihood (NLL)**.

### Intuition

```
If our model says P(next='m' | prev='e') = 0.15
Then:
  log(0.15) = -1.897    (log of a probability is always negative)
  -log(0.15) = 1.897    (negative log likelihood - a positive number)

Lower NLL = better model
- If model is very confident and correct: P = 0.99 -> NLL = 0.01 (great!)
- If model is uncertain: P = 0.05 -> NLL = 3.0 (bad)
- If model says P = 0.0: NLL = infinity (catastrophic!)
```

### Computing NLL for Our Model

```python
# Calculate the negative log likelihood of the entire dataset
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood / n  # average NLL per character
print(f"Negative Log Likelihood: {nll.item():.4f}")
# Bigram model gets about 2.45
```

**Benchmark:** A model that predicts uniformly random would get NLL = -log(1/27) = 3.30. Our bigram model gets ~2.45, so it learned SOMETHING. A perfect model would get NLL = 0.

---

## Chapter 6: Bigram as a Neural Network

Now let's re-implement the same bigram model, but using a neural network. Why? Because this framework will scale to much more powerful models.

### The Idea

Instead of looking up counts in a table, we'll:
1. Encode the input character as a **one-hot vector**
2. Multiply by a **weight matrix** (learnable parameters)
3. Apply **softmax** to get probabilities
4. Use **cross-entropy loss** to train

```python
# Create the training dataset
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.shape[0]
print(f"Number of training examples: {num_examples}")
# About 228,000 bigram examples
```

### One-Hot Encoding

```python
# One-hot encode the input
# If input character is 'a' (index 1), one-hot = [0, 1, 0, 0, ..., 0]
xenc = F.one_hot(xs, num_classes=vocab_size).float()
print(f"Input shape: {xenc.shape}")   # [228146, 27]
print(f"First example: '{itos[xs[0].item()]}' -> '{itos[ys[0].item()]}'")
print(f"One-hot: {xenc[0]}")
```

### The Neural Network (Single Layer)

```python
# Initialize random weights
g = torch.Generator().manual_seed(42)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# Forward pass
logits = xenc @ W              # (N, 27) @ (27, 27) = (N, 27)  raw scores
counts = logits.exp()          # exponentiate to get positive "counts"
probs = counts / counts.sum(dim=1, keepdim=True)  # normalize to probabilities
# This is just softmax! (PyTorch has F.softmax but we do it manually to understand)

print(f"Logits shape: {logits.shape}")   # [228146, 27]
print(f"Probs shape: {probs.shape}")     # [228146, 27]
print(f"Probs[0] sums to: {probs[0].sum().item():.4f}")  # 1.0
```

**What just happened?**
```
one-hot × W = logits   (raw scores, can be any number)
exp(logits)  = counts   (positive numbers)
normalize    = probs    (sum to 1, valid probability distribution)

This is the SOFTMAX function!
```

### Loss and Training

```python
# Compute loss: negative log likelihood
# For each example, we want the probability the model assigned to the correct next character
loss = -probs[torch.arange(num_examples), ys].log().mean()
print(f"Initial loss: {loss.item():.4f}")  # ~3.7 (random weights = bad)

# Training loop
for epoch in range(100):
    # Forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    # Loss
    loss = -probs[torch.arange(num_examples), ys].log().mean()

    # Add regularization to keep weights small
    loss = loss + 0.01 * (W ** 2).mean()

    # Backward pass
    W.grad = None  # zero gradients
    loss.backward()

    # Update
    W.data -= 50.0 * W.grad  # larger learning rate works here

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")
# Should converge to about 2.45 - same as the counting method!
```

**Key insight:** The neural network bigram model converges to the SAME result as the counting approach. The weight matrix W ends up encoding the same statistics as our count matrix N (up to a softmax transformation). But the neural network framework lets us easily extend to more complex models.

---

## Chapter 7: One-Hot Encoding & Embeddings

### The Problem with One-Hot

One-hot vectors are wasteful:

```
'a' = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'b' = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Problems:
1. Huge vectors (GPT has 50,000+ tokens - imagine that as one-hot!)
2. All characters are "equally different" - 'a' is as far from 'b' as from 'z'
3. No notion of similarity
```

### Embeddings: The Solution

Instead of a sparse one-hot vector, represent each character as a **small, dense, learnable vector** called an **embedding**.

```python
# Embedding table: each of 27 characters gets a 2D embedding (for visualization)
# In practice, embedding dimensions are 32, 64, 128, etc.

C = torch.randn((vocab_size, 2), generator=g)  # 27 characters x 2 dimensions

# To get the embedding for character 'a' (index 1):
# Option 1: one-hot multiply (wasteful)
# emb = one_hot_a @ C

# Option 2: just index into the table (efficient!)
emb = C[1]  # same result, much faster!
print(f"Embedding of 'a': {emb}")

# For a batch of characters:
embs = C[torch.tensor([1, 5, 13])]  # embeddings of a, e, m
print(f"Batch embeddings shape: {embs.shape}")  # [3, 2]
```

**Key insight:** `one_hot @ C` is equivalent to `C[index]`. The one-hot vector simply selects a row from the embedding table. So we skip the one-hot entirely and just do a lookup! This is what `torch.nn.Embedding` does.

### Why Embeddings Are Powerful

After training, similar characters end up with similar embeddings:

```
Before training (random):
  'a' = [0.31, -0.72]   (random)
  'e' = [-1.5,  0.43]   (random, far from 'a')

After training:
  'a' = [0.45, 0.82]    (vowels cluster together!)
  'e' = [0.51, 0.79]    (close to 'a')
  'z' = [-1.2, -0.5]    (far from vowels)
```

The network LEARNS that 'a' and 'e' behave similarly (both are vowels that appear in similar contexts), so it places them close together in embedding space.

---

## Chapter 8: The MLP Language Model (Bengio et al.)

Now the big upgrade. Instead of looking at just 1 previous character (bigram), we'll look at the previous **N** characters using an MLP.

This is based on the landmark 2003 paper by Yoshua Bengio: *"A Neural Probabilistic Language Model"*.

### Architecture

```
Input: previous 3 characters (context window = 3)

Step 1: Look up embeddings for each character
  'a','n','n' -> [emb_a, emb_n, emb_n] -> concatenate -> one long vector

Step 2: Feed through hidden layer
  hidden = tanh(concat_emb @ W1 + b1)

Step 3: Output layer
  logits = hidden @ W2 + b2

Step 4: Softmax to get probabilities
  probs = softmax(logits)

Diagram:
  char1 -> [C] ──┐
  char2 -> [C] ──┼── concat ──[W1, b1]──[tanh]──[W2, b2]──[softmax]──> probs
  char3 -> [C] ──┘
```

### Building the Dataset

```python
# Build dataset with context window
block_size = 3  # how many characters we look at to predict the next one

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size  # start with all '.'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide the window

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# Split data into train/val/test
import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

X_train, Y_train = build_dataset(words[:n1])    # 80% train
X_val, Y_val = build_dataset(words[n1:n2])      # 10% validation
X_test, Y_test = build_dataset(words[n2:])       # 10% test

print(f"Train: {X_train.shape}")  # e.g., [182625, 3]
print(f"Val:   {X_val.shape}")    # e.g., [22655, 3]
print(f"Test:  {X_test.shape}")   # e.g., [22866, 3]
```

Let's see what the data looks like:

```python
# Show some examples
for i in range(5):
    context = ''.join(itos[ix.item()] for ix in X_train[i])
    target = itos[Y_train[i].item()]
    print(f"  '{context}' -> '{target}'")

# Output:
#   '...' -> 'y'   (name starts with 'y')
#   '..y' -> 'u'
#   '.yu' -> 'h'
#   'yuh' -> 'e'
#   'uhe' -> 'n'
```

### The Model

```python
g = torch.Generator().manual_seed(42)

# Hyperparameters
n_embd = 10      # embedding dimension
n_hidden = 200   # hidden layer size

# Parameters
C = torch.randn((vocab_size, n_embd), generator=g)             # embedding table
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.01  # hidden layer
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01   # output layer
b2 = torch.randn(vocab_size, generator=g) * 0.01

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

n_params = sum(p.nelement() for p in parameters)
print(f"Total parameters: {n_params}")
# ~11,000 parameters - tiny but effective!
```

**Why `* 0.01`?** We initialize weights small to start near zero. If weights are large, the initial outputs are extreme, softmax becomes very peaked, and gradients are tiny (the network doesn't learn). Small initialization = gentle start.

---

## Chapter 9: Training the MLP

### The Training Loop

```python
# Training
losses_train = []
batch_size = 32

for epoch in range(50000):

    # Mini-batch: randomly sample a batch of examples
    ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    # Forward pass
    emb = C[X_batch]                              # [batch, block_size, n_embd]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)  # [batch, n_hidden]
    logits = h @ W2 + b2                           # [batch, vocab_size]
    loss = F.cross_entropy(logits, Y_batch)        # scalar

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update (use decaying learning rate)
    lr = 0.1 if epoch < 25000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    # Track loss
    if epoch % 5000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f}")
    losses_train.append(loss.item())
```

### Key Operations Explained

```python
# Let's trace through each step:

# 1. Embedding lookup
emb = C[X_batch]
# X_batch is [32, 3] (32 examples, each with 3 character indices)
# C is [27, 10] (27 characters, each a 10-dim embedding)
# emb is [32, 3, 10] (32 examples, 3 characters, each embedded as 10-dim)

# 2. Concatenate embeddings (flatten the last two dimensions)
emb_concat = emb.view(-1, block_size * n_embd)
# emb_concat is [32, 30] (32 examples, 3*10=30 features)
# This concatenates the 3 embeddings into one vector per example

# 3. Hidden layer
h = torch.tanh(emb_concat @ W1 + b1)
# emb_concat @ W1: [32, 30] @ [30, 200] = [32, 200]
# + b1: broadcast add bias
# tanh: squash to (-1, 1) -- non-linearity!
# h is [32, 200]

# 4. Output layer
logits = h @ W2 + b2
# h @ W2: [32, 200] @ [200, 27] = [32, 27]
# logits is [32, 27] - raw scores for each of 27 possible next characters

# 5. Loss
loss = F.cross_entropy(logits, Y_batch)
# F.cross_entropy internally does:
#   1. softmax(logits) -> probabilities
#   2. -log(prob of correct character) -> NLL
#   3. average over batch
```

### Evaluate on Validation Set

```python
# Evaluate on the full validation set
with torch.no_grad():
    emb = C[X_val]
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
    logits = h @ W2 + b2
    val_loss = F.cross_entropy(logits, Y_val)

print(f"Training loss:   {loss.item():.4f}")
print(f"Validation loss: {val_loss.item():.4f}")
# Bigram NLL was ~2.45. MLP should get ~2.1-2.2 (much better!)
```

### Plot Training Loss

```python
plt.figure(figsize=(12, 5))
# Plot smoothed loss (running average)
window = 200
smoothed = [sum(losses_train[max(0,i-window):i+1])/min(i+1,window)
            for i in range(len(losses_train))]
plt.plot(smoothed)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('MLP Language Model Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('mlp_training_loss.png', dpi=150)
plt.show()
```

---

## Chapter 10: Generating Names & Experiments

### Generating Names from the MLP

```python
g = torch.Generator().manual_seed(42)

for _ in range(20):
    name = []
    context = [0] * block_size  # start with '...'

    while True:
        # Forward pass
        emb = C[torch.tensor([context])]                    # [1, block_size, n_embd]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)          # [1, n_hidden]
        logits = h @ W2 + b2                                 # [1, vocab_size]
        probs = F.softmax(logits, dim=1)                     # [1, vocab_size]

        # Sample
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]

        if ix == 0:
            break
        name.append(itos[ix])

    print(''.join(name))
```

MLP output will look much better than bigrams:
```
mora
kayah
seel
nol
taty
jede
...
```

### Experiment 1: Visualize Embeddings

```python
# Plot the 2D embeddings to see what the model learned
# (only works if n_embd = 2, otherwise use PCA/t-SNE)

# For visualization, let's train a model with n_embd=2
# ... (retrain with n_embd=2) ...

plt.figure(figsize=(8, 8))
for i in range(vocab_size):
    x, y = C[i].data
    plt.scatter(x, y, s=100)
    plt.text(x.item() + 0.05, y.item() + 0.05, itos[i], fontsize=14)
plt.title('Character Embeddings (2D)')
plt.grid(True, alpha=0.3)
plt.savefig('embeddings_2d.png', dpi=150)
plt.show()

# You should see vowels clustering together,
# and the special '.' token far from the rest
```

### Experiment 2: Effect of Context Size

```python
# Try different block sizes and compare validation loss
for bs in [2, 3, 4, 5]:
    # Retrain with different block_size and report val loss
    print(f"block_size={bs} -> val_loss = ???")

# Expected: larger context = lower loss (up to a point)
# block_size=2: ~2.25
# block_size=3: ~2.15
# block_size=4: ~2.10
# block_size=5: ~2.08  (diminishing returns)
```

### Experiment 3: Effect of Embedding Dimension

```python
# Try different embedding sizes
for emb_dim in [2, 5, 10, 20, 50]:
    # Retrain with different n_embd and report val loss
    print(f"n_embd={emb_dim} -> val_loss = ???")

# Too small: not enough capacity to represent character similarities
# Too large: overfitting (more parameters, but not better generalization)
```

### Experiment 4: Effect of Hidden Layer Size

```python
# Try different hidden layer sizes
for hidden in [50, 100, 200, 300, 500]:
    # Retrain with different n_hidden and report val loss
    print(f"n_hidden={hidden} -> val_loss = ???")

# Bigger hidden layer = more capacity, but also more parameters to train
```

---

## Exercises & Projects

### Exercise 1: Build the Bigram from Scratch (1 hr)

Run the complete bigram counting model. Create `bigram.py`:
- Load names.txt
- Count all bigrams
- Visualize the count matrix
- Generate 20 names
- Compute NLL

### Exercise 2: Neural Bigram (1 hr)

Implement the neural network version of the bigram:
- One-hot encode inputs
- Single weight matrix
- Train with gradient descent
- Verify that the loss matches the counting approach

### Exercise 3: Full MLP (2 hrs)

Implement the MLP language model:
- Embedding table
- Hidden layer with tanh
- Output layer
- Train with mini-batches
- Plot training loss
- Evaluate on validation set
- Generate names

### Exercise 4: Hyperparameter Search (1 hr)

Systematically try different combinations:

```python
# Grid search
results = []
for n_embd in [5, 10, 20]:
    for n_hidden in [100, 200, 300]:
        for lr in [0.1, 0.05, 0.01]:
            # train and evaluate
            results.append({
                'n_embd': n_embd,
                'n_hidden': n_hidden,
                'lr': lr,
                'val_loss': val_loss
            })

# Find the best combination
best = min(results, key=lambda x: x['val_loss'])
print(f"Best config: {best}")
```

### Exercise 5: Train on a Different Dataset (1 hr)

Download a different names dataset (Indian names, city names, pokemon names) and train the model. Compare the generated outputs.

```python
# Ideas:
# - Baby names from SSA: https://www.ssa.gov/oact/babynames/names.zip
# - Pokemon names: manually collect ~800 pokemon names
# - City names: scrape a list of world cities
# - Dinosaur names: collect ~700 dinosaur names
```

### Exercise 6: Add a Second Hidden Layer (1 hr)

Extend the MLP to have 2 hidden layers instead of 1:

```python
# Current:   emb -> W1 -> tanh -> W2 -> logits
# New:       emb -> W1 -> tanh -> W3 -> tanh -> W2 -> logits

# Does the deeper model perform better? By how much?
# Watch out for vanishing gradients!
```

---

## References & Next Steps

### Watch Today

- Karpathy - "The spelled-out intro to language modeling: building makemore"
  - https://www.youtube.com/watch?v=PaCmpygFfXo
- Karpathy - "Building makemore Part 2: MLP"
  - https://www.youtube.com/watch?v=TCH_1BHY58I

### Read Today

- Bengio et al. - "A Neural Probabilistic Language Model" (2003)
  - https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
  - (The paper that started neural language models)

### Code Reference

- Karpathy's makemore: https://github.com/karpathy/makemore

### Tomorrow: Day 3

You'll build a **GPT from scratch** - using the Transformer architecture. You'll learn about self-attention, the mechanism that lets GPT look at ALL previous characters (not just the last 3), and understand why Transformers revolutionized AI.

---

## Interview Prep: Key Terms & Concepts for Day 2

> Revise this section before any ML/AI/NLP interview. These terms come up constantly.

---

### Language Modeling Fundamentals

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **Language Model** | A model that assigns probabilities to sequences of tokens. Predicts the next token given previous context. | "A language model learns the probability distribution of text. Given a sequence of tokens, it predicts what comes next. Every LLM (GPT, Claude) is fundamentally a language model trained on massive text data. The training objective is simply next-token prediction." |
| **Autoregressive Model** | A model that generates one token at a time, feeding each output back as input for the next prediction. | "Autoregressive means the model generates sequentially - each prediction depends on all previous predictions. GPT is autoregressive: it generates tokens left-to-right. This is opposed to models like BERT which can look at both directions simultaneously." |
| **Token** | The basic unit of text that a model processes. Could be a character, subword, or whole word. | "Tokens are the atomic units of text for a model. GPT uses BPE (byte pair encoding) to create subword tokens - common words are single tokens, rare words are split. 'unhappiness' might become ['un', 'happiness']. Vocabulary sizes are typically 30K-100K." |
| **Vocabulary** | The complete set of unique tokens the model knows. Each token maps to an integer index. | "The vocabulary defines what the model can 'see'. Anything not in the vocabulary gets split into known pieces or becomes an unknown token. GPT-2 has ~50K tokens, LLaMA has ~32K." |
| **Context Window (Block Size)** | The number of previous tokens the model considers when making a prediction. | "The context window determines how much history the model can use. A bigram has context=1, our MLP used context=3. GPT-4 has a context window of 128K tokens. Longer context = better understanding but more computation (quadratic cost with attention)." |

### N-gram Models

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **N-gram** | A contiguous sequence of N tokens. Used in statistical language models. | "An n-gram is a sequence of n consecutive tokens. Unigram=1 token, bigram=2, trigram=3. N-gram language models predict the next token based on the previous (n-1) tokens. They were the dominant approach before neural models." |
| **Bigram** | A pair of consecutive tokens. A bigram model predicts the next token given only the immediately previous token. | "A bigram model computes P(next | previous). It's the simplest possible language model beyond random. It only looks at one previous token, so it can't capture long-range dependencies. But it's fast, interpretable, and a good baseline." |
| **Trigram** | A sequence of 3 consecutive tokens. A trigram model uses 2 previous tokens for prediction. | "A trigram model computes P(next | prev2, prev1). More context means better predictions, but the number of possible trigrams grows as V^3 (vocabulary cubed), causing sparsity issues." |
| **Markov Assumption** | The assumption that the next token depends only on the last N tokens, not the entire history. | "N-gram models make the Markov assumption: P(next | all_history) ≈ P(next | last_N_tokens). This makes computation tractable but limits the model. Transformers break this assumption by attending to the entire context." |

### Embeddings

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **Embedding** | A dense, low-dimensional, learnable vector representation of a discrete token. | "An embedding maps a discrete token (like a word or character) to a continuous vector. Unlike one-hot (sparse, high-dim), embeddings are dense and low-dim (typically 64-1024 dims). The key insight is that similar tokens end up with similar embeddings - the model LEARNS these representations during training." |
| **Embedding Table (Lookup Table)** | A matrix of shape [vocab_size, embedding_dim]. Each row is one token's embedding. | "The embedding table is just a matrix where row i is the embedding for token i. Looking up an embedding is equivalent to multiplying a one-hot vector by this matrix, but implemented as a direct index lookup for efficiency. In PyTorch: `nn.Embedding(vocab_size, dim)`." |
| **One-Hot Encoding** | Representing a token as a vector of all zeros except a 1 at the token's index. | "One-hot creates a sparse vector of size vocab_size with a single 1. It's simple but wasteful: every token is equidistant from every other token. No notion of similarity. For a vocabulary of 50K tokens, each vector has 50K dimensions with only one non-zero entry." |
| **Embedding Dimension** | The size of each embedding vector. A hyperparameter. | "Embedding dimension is a trade-off: too small limits what the model can learn, too large wastes parameters and risks overfitting. Common choices: character-level models use 10-64, word-level models use 128-1024. GPT-3 uses 12,288-dimensional embeddings." |
| **Embedding Space** | The continuous vector space where embeddings live. Similar tokens cluster together. | "Embedding space is where the magic happens. After training, semantically similar words end up close together. The famous example: king - man + woman ≈ queen. This structure emerges purely from training on text - no one programs these relationships." |

### Loss Functions & Evaluation

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **Cross-Entropy Loss** | Standard loss for classification. Measures the distance between predicted probability distribution and the true distribution. | "Cross-entropy loss is the standard for classification and language modeling. For a correct class c with predicted probability p(c): loss = -log(p(c)). If the model is confident and correct (p=0.99), loss is small (0.01). If wrong (p=0.01), loss is huge (4.6). It heavily penalizes confident wrong predictions." |
| **Negative Log Likelihood (NLL)** | The negative of the log of the probability assigned to the correct token. Lower is better. | "NLL measures how surprised the model is by the correct answer. NLL = -log(P(correct_token)). It's equivalent to cross-entropy for one-hot targets. For language models, we report average NLL per token. A perfect model gets NLL=0, random gets NLL=log(vocab_size)." |
| **Perplexity** | exp(NLL). Intuitive measure of how many tokens the model is 'confused' between. Lower is better. | "Perplexity = exp(average_NLL). If perplexity = 10, the model is as confused as if it were choosing uniformly among 10 options at each step. Random model on 27 chars: perplexity=27. Good model: perplexity might be 5-10. GPT-4 on text: perplexity < 10." |
| **Softmax** | Function that converts a vector of raw scores (logits) into a probability distribution. | "Softmax takes any real-valued vector and converts it to probabilities: softmax(x_i) = exp(x_i) / sum(exp(x_j)). Properties: all outputs are positive, they sum to 1, larger inputs get larger probabilities. It's the standard final step before cross-entropy loss." |
| **Logits** | Raw, unnormalized scores output by the model before softmax. Can be any real number. | "Logits are the model's raw output scores before normalization. The name comes from 'log-odds'. They can be any real number (-inf to +inf). Softmax converts logits to probabilities. We often compute loss directly from logits (more numerically stable) using F.cross_entropy in PyTorch." |
| **Temperature** | A hyperparameter that controls the randomness of sampling from a probability distribution. | "Temperature scales logits before softmax: softmax(logits / T). T=1.0 is normal. T<1 makes the distribution sharper (more confident, less random). T>1 makes it flatter (more random, more creative). T->0 becomes argmax (greedy). T->inf becomes uniform random." |

### Training Techniques

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **Mini-Batch** | A small random subset of the training data used for each gradient update. | "Instead of computing gradients on the entire dataset (slow) or one example (noisy), we use mini-batches of 32-512 examples. This balances speed and gradient quality. It also adds beneficial noise that helps escape local minima." |
| **Train/Validation/Test Split** | Dividing data into three sets: train (learn), validation (tune hyperparameters), test (final evaluation). | "Train set: the model learns from this. Validation set: we use this to tune hyperparameters and detect overfitting. Test set: touched ONLY once for final evaluation. Typical split: 80/10/10. The validation set prevents us from fooling ourselves during development." |
| **Overfitting** | When the model memorizes training data instead of learning general patterns. Training loss << validation loss. | "Overfitting means the model is essentially memorizing. Signs: training loss keeps decreasing but validation loss starts increasing. Solutions: more data, regularization, dropout, simpler model, early stopping." |
| **Smoothing (Laplace)** | Adding a small count to all events to prevent zero probabilities in n-gram models. | "Laplace smoothing adds 1 to every count in an n-gram model, so P(never_seen_bigram) > 0 instead of P=0. Without smoothing, any unseen event causes log(0)=-infinity during evaluation. It's a simple form of regularization specific to count-based models." |
| **Learning Rate Decay** | Reducing the learning rate during training for finer convergence. | "Start with a high learning rate for fast initial progress, then reduce it for fine-grained optimization near the minimum. Common strategies: step decay (halve every N epochs), cosine decay (smooth curve), warmup + decay (Transformer standard)." |
| **Weight Initialization** | How weights are set before training begins. Affects convergence speed and quality. | "Bad initialization can kill training. Too large weights = exploding gradients. Too small = vanishing gradients. Common strategies: Xavier/Glorot (scales by 1/sqrt(fan_in)), Kaiming/He (accounts for ReLU). For embeddings, small random works well." |

### Architecture Concepts

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **MLP (as Language Model)** | Using a Multi-Layer Perceptron to predict the next token from a fixed window of previous tokens. | "The Bengio MLP language model (2003) was revolutionary: embed previous N tokens, concatenate embeddings, feed through hidden layers, output next-token probabilities. It was the first successful neural language model and introduced the idea of learned word embeddings." |
| **Hidden Layer** | An intermediate layer between input and output that learns internal representations. | "Hidden layers extract features. In a language model, the hidden layer learns to combine the embeddings of context characters into a useful representation for prediction. Deeper/wider hidden layers = more capacity but more parameters." |
| **Concatenation (of Embeddings)** | Joining multiple embedding vectors into one long vector as input to the MLP. | "When our context is 3 characters with 10-dim embeddings each, we concatenate them into a 30-dim vector. This preserves the position: the model knows which embedding came from which position. This is the simplest way to combine multiple inputs." |

### Sampling & Generation

| Term | Definition | Interview-Ready Answer |
|------|-----------|----------------------|
| **Greedy Decoding** | Always picking the most probable next token. | "Greedy decoding picks argmax at each step. It's deterministic and fast but often produces repetitive, boring text. 'The the the the...' It gets stuck in loops because it never explores alternatives." |
| **Sampling** | Randomly picking the next token according to the model's probability distribution. | "Sampling draws from the full distribution, so rare but interesting tokens have a chance. It produces more diverse and creative text. Temperature controls the trade-off between diversity and quality." |
| **Top-k Sampling** | Only sampling from the k most probable tokens, ignoring the rest. | "Top-k restricts sampling to the k most likely tokens and renormalizes. k=1 is greedy, k=50 allows diversity while filtering garbage. Problem: for some distributions, 5 tokens might be equally likely; for others, 1 token dominates. Fixed k doesn't adapt." |
| **Top-p (Nucleus) Sampling** | Sampling from the smallest set of tokens whose cumulative probability exceeds p. | "Top-p (nucleus sampling) dynamically selects the number of candidates. It includes tokens until their cumulative probability exceeds p (e.g., 0.9). This adapts: when the model is confident, few tokens pass; when uncertain, many pass. Often preferred over top-k." |

### Common Interview Questions

**Q: What is the difference between a character-level and word-level language model?**
> Character-level: vocabulary is individual characters (~26-256). Smaller vocabulary, no unknown words, but sequences are much longer. Word-level: vocabulary is words (~30K-100K). Shorter sequences but can't handle unseen words. Modern models use subword tokenization (BPE) which combines the best of both.

**Q: Why did we move from n-gram models to neural language models?**
> N-gram models suffer from the curse of dimensionality: the number of possible n-grams grows exponentially with n. For large contexts, most n-grams are never seen in training. Neural models solve this with embeddings: similar words share similar representations, so the model generalizes to unseen combinations.

**Q: What's the relationship between language modeling and generation?**
> Language modeling (training) and generation (inference) use the same model differently. Training: feed real text, maximize P(correct_next_token). Generation: sample from P(next_token | context), append, repeat. The better the language model, the better the generated text.

**Q: Explain the softmax bottleneck.**
> The softmax output layer has a rank constraint: with hidden size h and vocab size V, the logits are h->V, limiting the rank of the output distribution to h. This means the model can't represent certain probability distributions when h < V. Solutions: mixture of softmax, larger hidden layers.

**Q: How does torch.nn.Embedding work under the hood?**
> It's a matrix of shape [vocab_size, embedding_dim]. Forward pass: given integer indices, it simply indexes into the matrix (like a dictionary lookup). Backward pass: only the rows that were accessed get gradient updates. It's mathematically equivalent to one-hot @ embedding_matrix, but much more efficient.

**Q: What is teacher forcing?**
> During training, we feed the REAL previous tokens as context (not the model's own predictions). This is called teacher forcing. Without it, early mistakes cascade. Downside: at generation time, the model sees its own (potentially wrong) predictions, which it never encountered during training. This train/test mismatch is called exposure bias.

---

### Flashcard Summary (Quick Revision)

```
Language Model     = Predicts P(next token | previous tokens)
Autoregressive     = Generate one token at a time, each feeds into the next
N-gram             = Model based on counting sequences of N tokens
Bigram             = P(next | prev_1), simplest language model
Embedding          = Dense, learnable vector for each token (replaces one-hot)
Embedding Table    = Matrix[vocab_size, emb_dim], row i = embedding of token i
One-Hot            = Sparse vector, all zeros except a 1 at the token's index
Softmax            = Converts raw logits to probabilities (positive, sum to 1)
Logits             = Raw model scores before softmax
Cross-Entropy      = Loss = -log(P(correct token)), lower is better
NLL                = Negative Log Likelihood, same as cross-entropy for one-hot labels
Perplexity         = exp(NLL), intuitive measure of model confusion
Temperature        = Scales logits before softmax: <1=confident, >1=random
Mini-Batch         = Small random subset of data for each training step
Smoothing          = Adding counts to prevent zero probabilities in n-grams
Context Window     = How many previous tokens the model considers
Teacher Forcing    = Using real tokens (not model predictions) as input during training
Greedy Decoding    = Always pick the highest probability token (deterministic)
Top-k Sampling     = Sample from the k most likely tokens only
Top-p Sampling     = Sample from smallest set of tokens with cumulative prob >= p
```
