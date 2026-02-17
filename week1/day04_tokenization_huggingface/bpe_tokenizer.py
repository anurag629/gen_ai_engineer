"""
Day 4: Byte Pair Encoding (BPE) Tokenizer from Scratch

Implements the BPE algorithm that powers GPT-2, GPT-3, GPT-4, and many other
modern language models. We build it step by step:

1. Start with individual characters (or bytes) as the initial vocabulary
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until we reach the desired vocabulary size

This is essentially Karpathy's minbpe approach, distilled for learning.

Run: python3 bpe_tokenizer.py
"""

import re
from collections import Counter


# ============================================================
# 1. Training Corpus (inline to avoid extra files)
# ============================================================

CORPUS = """
The quick brown fox jumps over the lazy dog. The dog barked at the fox,
but the fox was too quick. Meanwhile, the cat sat on the mat and watched
the entire scene unfold. The cat was not impressed.

Natural language processing is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers
and human language. The goal is to enable computers to understand, interpret,
and generate human language in a way that is both meaningful and useful.

The transformer architecture has revolutionized natural language processing.
Transformers use self-attention mechanisms to process sequences in parallel,
rather than sequentially like recurrent neural networks. This parallelism
makes transformers much faster to train on modern hardware.

GPT models are decoder-only transformers trained on massive text corpora.
They predict the next token given a sequence of previous tokens. The training
objective is simple: minimize the cross-entropy loss between predicted and
actual next tokens. Despite this simple objective, GPT models develop
remarkable capabilities including reasoning, translation, and code generation.

Tokenization is the process of converting raw text into a sequence of tokens
that a language model can process. Good tokenization balances vocabulary size
against sequence length. Too small a vocabulary leads to very long sequences;
too large a vocabulary wastes model capacity on rare tokens.

The byte pair encoding algorithm starts with individual characters and
iteratively merges the most frequent adjacent pairs. This creates a subword
vocabulary that can represent any text without out-of-vocabulary tokens.
Common words become single tokens while rare words are broken into subword
pieces. This is an elegant solution to the open vocabulary problem.
"""


# ============================================================
# 2. Helper Functions
# ============================================================

def get_stats(ids):
    """Count the frequency of each adjacent pair in the token list.

    Args:
        ids: list of integer token IDs

    Returns:
        Counter mapping (pair) -> count
    """
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids, pair, new_id):
    """Replace all occurrences of `pair` in `ids` with `new_id`.

    Args:
        ids: list of integer token IDs
        pair: tuple of two consecutive token IDs to merge
        new_id: the new token ID replacing the pair

    Returns:
        new list with the pair replaced
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # If we see the pair and we're not at the last position, merge
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ============================================================
# 3. BasicTokenizer Class
# ============================================================

class BasicTokenizer:
    """A minimal BPE tokenizer built from scratch.

    The vocabulary starts with 256 byte values (0-255), and we add
    `num_merges` new tokens by iteratively merging the most frequent pair.
    """

    def __init__(self):
        self.merges = {}       # (pair) -> new_token_id
        self.vocab = {}        # token_id -> bytes

    def train(self, text, num_merges, verbose=False):
        """Train BPE on the given text.

        Args:
            text: training corpus string
            num_merges: number of merge operations (= new tokens to create)
            verbose: if True, print each merge step
        """
        # Convert text to bytes (UTF-8), then to list of ints
        tokens = list(text.encode("utf-8"))

        # Build initial vocab: 256 single-byte tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        if verbose:
            print(f"Training BPE with {num_merges} merges")
            print(f"Initial corpus length: {len(tokens)} bytes")
            print(f"Initial vocab size: {len(self.vocab)}")
            print("-" * 55)

        for i in range(num_merges):
            stats = get_stats(tokens)
            if not stats:
                break

            # Find the most frequent pair
            top_pair = max(stats, key=stats.get)
            top_count = stats[top_pair]

            # Create new token ID
            new_id = 256 + i

            # Perform the merge
            tokens = merge(tokens, top_pair, new_id)

            # Record the merge rule
            self.merges[top_pair] = new_id

            # Build the new token's bytes
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                token_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  Merge {i+1:3d}: {top_pair} -> {new_id}  "
                      f"('{token_str}' x{top_count}, "
                      f"tokens: {len(tokens)})")

        if verbose:
            print("-" * 55)
            print(f"Final vocab size: {len(self.vocab)}")
            print(f"Final corpus length: {len(tokens)} tokens")
            print(f"Compression ratio: {len(text.encode('utf-8'))}/{len(tokens)} "
                  f"= {len(text.encode('utf-8'))/len(tokens):.2f}x")

    def encode(self, text):
        """Encode a string into a list of token IDs.

        We apply the learned merge rules in the same order they were learned.
        """
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            # Find the pair with the lowest merge index (earliest learned)
            stats = get_stats(tokens)
            # Among all pairs present, find the one that was merged earliest
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # No more merges possible

            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)

        return tokens

    def decode(self, ids):
        """Decode a list of token IDs back to a string."""
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def token_to_str(self, token_id):
        """Get the string representation of a single token."""
        return self.vocab[token_id].decode("utf-8", errors="replace")


# ============================================================
# 4. Train and Demonstrate
# ============================================================

def demo_basic_tokenizer():
    """Train BPE and show results."""
    print("=" * 60)
    print("PART 1: Training a BPE Tokenizer from Scratch")
    print("=" * 60)

    tok = BasicTokenizer()
    tok.train(CORPUS, num_merges=100, verbose=True)

    # Show some vocabulary entries
    print("\n--- Sample Vocabulary (last 20 merges) ---")
    for token_id in range(356 - 20, 356):
        token_bytes = tok.vocab[token_id]
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"  Token {token_id}: '{token_str}' ({len(token_bytes)} bytes)")

    return tok


# ============================================================
# 5. Encoding / Decoding Tests
# ============================================================

def demo_encode_decode(tok):
    """Test encoding and decoding with various strings."""
    print("\n" + "=" * 60)
    print("PART 2: Encoding & Decoding Tests")
    print("=" * 60)

    test_strings = [
        "The quick brown fox",
        "transformer architecture",
        "Hello, World!",
        "tokenization is important",
        "xyzzy",  # rare / unseen text
        "GPT models are decoder-only transformers",
    ]

    for s in test_strings:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        raw_bytes = len(s.encode("utf-8"))

        print(f"\n  Input:    '{s}'")
        print(f"  Tokens:   {ids}")
        print(f"  Decoded:  '{decoded}'")
        print(f"  # tokens: {len(ids)}  (raw bytes: {raw_bytes}, "
              f"ratio: {raw_bytes/len(ids):.2f}x)")
        assert decoded == s, f"Round-trip FAILED for '{s}'"

    print("\n  All round-trip tests PASSED!")


# ============================================================
# 6. Compression Ratio Analysis
# ============================================================

def demo_compression_analysis():
    """Train with different numbers of merges and compare compression."""
    print("\n" + "=" * 60)
    print("PART 3: Compression Ratio vs Number of Merges")
    print("=" * 60)

    test_text = CORPUS
    raw_bytes = len(test_text.encode("utf-8"))

    merge_counts = [0, 10, 25, 50, 100, 150, 200, 256]
    print(f"\n  Raw text: {raw_bytes} bytes\n")
    print(f"  {'Merges':>8}  {'Vocab':>6}  {'Tokens':>7}  {'Ratio':>7}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}")

    for n in merge_counts:
        tok = BasicTokenizer()
        tok.train(test_text, num_merges=n)
        ids = tok.encode(test_text)
        ratio = raw_bytes / len(ids)
        vocab_size = 256 + n
        print(f"  {n:>8}  {vocab_size:>6}  {len(ids):>7}  {ratio:>7.2f}x")

    print("\n  More merges = smaller token sequences = better compression")
    print("  But diminishing returns after ~200 merges on this small corpus")


# ============================================================
# 7. Compare with tiktoken (GPT-2's real tokenizer)
# ============================================================

def demo_tiktoken_comparison():
    """Compare our BPE with tiktoken's GPT-2 tokenizer."""
    print("\n" + "=" * 60)
    print("PART 4: Comparison with tiktoken (GPT-2 tokenizer)")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("\n  tiktoken not installed. Skipping comparison.")
        print("  Install with: pip install tiktoken")
        return

    enc = tiktoken.get_encoding("gpt2")

    test_strings = [
        "Hello, World!",
        "The transformer architecture has revolutionized NLP.",
        "tokenization is the unsung hero of language models",
        "GPT-4o is a multimodal model",
        "Anthropic builds Claude",
    ]

    # Train our tokenizer for comparison
    tok = BasicTokenizer()
    tok.train(CORPUS, num_merges=200)

    print(f"\n  {'Text':<50}  {'Ours':>5}  {'GPT-2':>5}  {'Ratio':>6}")
    print(f"  {'-'*50}  {'-'*5}  {'-'*5}  {'-'*6}")

    for s in test_strings:
        our_ids = tok.encode(s)
        gpt2_ids = enc.encode(s)
        display = s[:48] + ".." if len(s) > 50 else s
        ratio = len(our_ids) / len(gpt2_ids) if gpt2_ids else 0
        print(f"  {display:<50}  {len(our_ids):>5}  {len(gpt2_ids):>5}  {ratio:>5.1f}x")

    print("\n  GPT-2 uses ~50K merges trained on ~40GB of text")
    print("  Our tokenizer uses 200 merges trained on ~1KB of text")
    print("  So GPT-2 achieves much better compression!")

    # Show GPT-2's tokenization
    print("\n  --- GPT-2 tokenization examples ---")
    for s in test_strings[:3]:
        ids = enc.encode(s)
        tokens = [enc.decode([i]) for i in ids]
        print(f"\n  Text:   '{s}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs:    {ids}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    tok = demo_basic_tokenizer()
    demo_encode_decode(tok)
    demo_compression_analysis()
    demo_tiktoken_comparison()

    print("\n" + "=" * 60)
    print("BPE Tokenizer demo complete!")
    print("=" * 60)
