"""
Educational visualizations for Tokenization & Hugging Face concepts.
Generates PNG diagrams illustrating key components of tokenization and text generation.

Uses only matplotlib and numpy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Consistent styling (matches Day 1-3)
COLORS = {
    "blue": "#3B82F6",
    "blue_light": "#93C5FD",
    "green": "#10B981",
    "green_light": "#6EE7B7",
    "red": "#EF4444",
    "red_light": "#FCA5A5",
    "purple": "#8B5CF6",
    "purple_light": "#C4B5FD",
    "orange": "#F59E0B",
    "orange_light": "#FCD34D",
    "gray": "#6B7280",
    "gray_light": "#D1D5DB",
    "dark": "#1F2937",
    "white": "#FFFFFF",
    "bg": "#F9FAFB",
}

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, filename):
    """Save figure to PNG and print confirmation."""
    filepath = os.path.join(SAVE_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# 1. Tokenization Comparison: Character vs Word vs BPE
# ---------------------------------------------------------------------------
def viz_tokenization_comparison():
    """Show the same sentence tokenized three different ways."""
    text = "unhappiness is not unrelated"

    char_tokens = list(text)
    word_tokens = text.split()
    bpe_tokens = ["un", "happiness", " is", " not", " un", "related"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=COLORS["bg"])

    methods = [
        ("Character-Level", char_tokens, COLORS["red_light"], COLORS["red"]),
        ("Word-Level", word_tokens, COLORS["blue_light"], COLORS["blue"]),
        ("BPE (Subword)", bpe_tokens, COLORS["green_light"], COLORS["green"]),
    ]

    for idx, (title, tokens, bg_color, border_color) in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(-0.5, max(len(char_tokens), 10) + 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis("off")

        # Title on the left
        ax.text(-0.3, 0.5, f"{title}\n({len(tokens)} tokens)",
                ha="right", va="center", fontsize=11, fontweight="bold",
                color=border_color)

        # Draw token boxes
        x = 0.0
        for token in tokens:
            display = repr(token)[1:-1] if token == " " else token
            w = max(len(display) * 0.35, 0.5)
            box = FancyBboxPatch(
                (x, 0.1), w, 0.8,
                boxstyle="round,pad=0.08",
                facecolor=bg_color, edgecolor=border_color, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(x + w / 2, 0.5, display,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=COLORS["dark"], family="monospace")
            x += w + 0.15

    fig.suptitle(f'Tokenization Comparison: "{text}"',
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.12, 0, 1, 0.94])
    save_figure(fig, "viz_tokenization_comparison.png")


# ---------------------------------------------------------------------------
# 2. BPE Algorithm Step-by-Step
# ---------------------------------------------------------------------------
def viz_bpe_algorithm():
    """Show BPE merge steps on 'low lower lowest'."""
    steps = [
        ("Initial", ["l o w _", "l o w e r _", "l o w e s t _"], None),
        ("Merge 1: l+o → lo", ["lo w _", "lo w e r _", "lo w e s t _"], "lo"),
        ("Merge 2: lo+w → low", ["low _", "low e r _", "low e s t _"], "low"),
        ("Merge 3: e+r → er", ["low _", "low er _", "low e s t _"], "er"),
        ("Merge 4: e+s → es", ["low _", "low er _", "low es t _"], "es"),
        ("Merge 5: es+t → est", ["low _", "low er _", "low est _"], "est"),
    ]

    fig, axes = plt.subplots(len(steps), 1, figsize=(12, 10), facecolor=COLORS["bg"])

    merge_colors = [COLORS["gray_light"], COLORS["blue_light"], COLORS["green_light"],
                    COLORS["orange_light"], COLORS["purple_light"], COLORS["red_light"]]

    for idx, (title, words, highlight) in enumerate(steps):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(-0.5, 20)
        ax.set_ylim(-0.3, 1.3)
        ax.axis("off")

        # Step label
        ax.text(-0.3, 0.5, title, ha="right", va="center",
                fontsize=10, fontweight="bold", color=COLORS["dark"])

        # Draw each word's tokens
        x = 0.0
        for wi, word in enumerate(words):
            tokens = word.split()
            for token in tokens:
                is_highlight = (highlight and token == highlight)
                bg = merge_colors[idx] if is_highlight else COLORS["white"]
                border = COLORS["dark"] if is_highlight else COLORS["gray"]
                lw = 2.0 if is_highlight else 1.0

                w = max(len(token) * 0.4, 0.5)
                box = FancyBboxPatch(
                    (x, 0.1), w, 0.7,
                    boxstyle="round,pad=0.06",
                    facecolor=bg, edgecolor=border, linewidth=lw,
                )
                ax.add_patch(box)
                ax.text(x + w / 2, 0.45, token,
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=COLORS["dark"],
                        family="monospace")
                x += w + 0.1

            # Space between words
            x += 0.4

    fig.suptitle("BPE Algorithm: Iterative Merging on 'low lower lowest'",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.18, 0, 1, 0.96])
    save_figure(fig, "viz_bpe_algorithm.png")


# ---------------------------------------------------------------------------
# 3. UTF-8 Encoding Visualization
# ---------------------------------------------------------------------------
def viz_utf8_encoding():
    """Show how characters map to bytes in UTF-8."""
    examples = [
        ("A", "U+0041", [0x41], "1 byte (ASCII)"),
        ("é", "U+00E9", [0xC3, 0xA9], "2 bytes"),
        ("€", "U+20AC", [0xE2, 0x82, 0xAC], "3 bytes"),
        ("\u2764", "U+2764", [0xE2, 0x9D, 0xA4], "3 bytes (heart)"),
        ("h", "U+0068", [0x68], "1 byte (ASCII)"),
        ("i", "U+0069", [0x69], "1 byte (ASCII)"),
        ("!", "U+0021", [0x21], "1 byte (ASCII)"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, len(examples) + 0.5)
    ax.axis("off")

    # Column headers
    headers = ["Character", "Code Point", "UTF-8 Bytes", "Size"]
    header_x = [0.5, 2.5, 5.5, 11.0]
    for hx, header in zip(header_x, headers):
        ax.text(hx, len(examples) + 0.1, header,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=COLORS["dark"])

    byte_colors = [COLORS["blue_light"], COLORS["green_light"],
                   COLORS["orange_light"], COLORS["purple_light"]]

    for idx, (char, codepoint, utf8_bytes, size) in enumerate(examples):
        y = len(examples) - 1 - idx

        # Character
        ax.text(0.5, y, char, ha="center", va="center",
                fontsize=20, fontweight="bold", color=COLORS["dark"])

        # Code point
        ax.text(2.5, y, codepoint, ha="center", va="center",
                fontsize=11, color=COLORS["purple"], family="monospace",
                fontweight="bold")

        # UTF-8 bytes as colored boxes
        bx = 4.0
        for bi, byte_val in enumerate(utf8_bytes):
            box = FancyBboxPatch(
                (bx, y - 0.25), 1.2, 0.5,
                boxstyle="round,pad=0.06",
                facecolor=byte_colors[bi], edgecolor=COLORS["dark"],
                linewidth=1.2,
            )
            ax.add_patch(box)
            ax.text(bx + 0.6, y, f"0x{byte_val:02X}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=COLORS["dark"],
                    family="monospace")
            bx += 1.4

        # Size
        ax.text(11.0, y, size, ha="center", va="center",
                fontsize=10, color=COLORS["gray"], fontstyle="italic")

    # Horizontal line under header
    ax.axhline(y=len(examples) - 0.3, xmin=0.02, xmax=0.95,
               color=COLORS["gray"], linewidth=1.0, alpha=0.5)

    ax.set_title("UTF-8 Encoding: Characters → Bytes",
                 fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    save_figure(fig, "viz_utf8_encoding.png")


# ---------------------------------------------------------------------------
# 4. Vocab Size vs Sequence Length Tradeoff
# ---------------------------------------------------------------------------
def viz_vocab_size_tradeoff():
    """Show the tradeoff between vocabulary size and sequence length."""
    # Simulated data
    vocab_sizes = [256, 500, 1000, 2000, 4000, 8000, 16000, 32000, 50257, 100000]
    # As vocab grows, sequences get shorter but vocab table gets bigger
    seq_lengths = [450, 300, 210, 160, 125, 100, 82, 70, 65, 58]
    # Training efficiency (sweet spot around 32K-50K)
    efficiency = [30, 45, 60, 72, 82, 90, 95, 98, 100, 97]

    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    ax1.set_facecolor(COLORS["bg"])

    # Sequence length (left axis)
    line1 = ax1.plot(vocab_sizes, seq_lengths, color=COLORS["blue"],
                     linewidth=2.5, marker="o", markersize=8,
                     label="Avg Sequence Length", zorder=5)
    ax1.set_xlabel("Vocabulary Size", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Avg Sequence Length (tokens)", fontsize=12,
                   color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Efficiency (right axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(vocab_sizes, efficiency, color=COLORS["green"],
                     linewidth=2.5, marker="s", markersize=8,
                     label="Training Efficiency", zorder=5)
    ax2.set_ylabel("Training Efficiency (%)", fontsize=12,
                   color=COLORS["green"])
    ax2.tick_params(axis="y", labelcolor=COLORS["green"])

    # Mark sweet spot
    ax1.axvspan(30000, 55000, alpha=0.15, color=COLORS["orange"],
                label="Sweet Spot (~32K-50K)")

    # Annotations
    ax1.annotate("GPT-2: 50,257", xy=(50257, 65), xytext=(50257, 150),
                 fontsize=10, fontweight="bold", color=COLORS["orange"],
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5))

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    labels.append("Sweet Spot (~32K-50K)")
    handles = lines + [mpatches.Patch(color=COLORS["orange"], alpha=0.3)]
    ax1.legend(handles, labels, loc="center right", fontsize=10, framealpha=0.9)

    ax1.set_title("Vocabulary Size vs Sequence Length Tradeoff",
                  fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "viz_vocab_size_tradeoff.png")


# ---------------------------------------------------------------------------
# 5. Temperature Effect on Probability Distributions
# ---------------------------------------------------------------------------
def viz_temperature_effect():
    """Show how temperature affects the softmax distribution."""
    # Raw logits for next token prediction
    tokens = ["the", "a", "my", "one", "that", "this", "an", "his"]
    logits = np.array([3.5, 2.8, 2.1, 1.5, 1.0, 0.8, 0.3, -0.2])

    temperatures = [0.3, 0.7, 1.0, 1.5]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=COLORS["bg"])
    bar_colors = [COLORS["red"], COLORS["orange"], COLORS["blue"], COLORS["purple"]]

    for idx, (temp, color) in enumerate(zip(temperatures, bar_colors)):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])

        # Apply temperature
        scaled = logits / temp
        probs = np.exp(scaled - np.max(scaled))
        probs = probs / probs.sum()

        bars = ax.barh(range(len(tokens)), probs, color=color, alpha=0.8,
                       edgecolor=COLORS["dark"], linewidth=0.8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Probability", fontsize=10)
        ax.invert_yaxis()

        # Add probability labels
        for i, (bar, p) in enumerate(zip(bars, probs)):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{p:.3f}", va="center", fontsize=9, color=COLORS["dark"])

        # Describe the effect
        if temp < 0.5:
            desc = "Very focused\n(nearly greedy)"
        elif temp < 0.9:
            desc = "Moderately\nfocused"
        elif temp <= 1.0:
            desc = "Original\ndistribution"
        else:
            desc = "More uniform\n(creative)"

        ax.set_title(f"T = {temp}\n{desc}", fontsize=12, fontweight="bold",
                     color=color)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Temperature Effect on Token Probabilities",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "viz_temperature_effect.png")


# ---------------------------------------------------------------------------
# 6. Sampling Strategies: Greedy vs Top-k vs Top-p
# ---------------------------------------------------------------------------
def viz_sampling_strategies():
    """Compare greedy, top-k, and top-p (nucleus) sampling."""
    tokens = ["the", "a", "my", "one", "that", "this", "an", "his",
              "some", "each"]
    probs = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03,
                      0.02, 0.02])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=COLORS["bg"])

    # Greedy
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    colors_greedy = [COLORS["green"] if i == 0 else COLORS["gray_light"]
                     for i in range(len(tokens))]
    ax.barh(range(len(tokens)), probs, color=colors_greedy,
            edgecolor=COLORS["dark"], linewidth=0.8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title("Greedy Decoding\n(Always pick highest)", fontsize=12,
                 fontweight="bold", color=COLORS["green"])
    ax.set_xlabel("Probability", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    # Arrow pointing to selected
    ax.annotate("Selected!", xy=(0.30, 0), xytext=(0.35, 1.5),
                fontsize=10, fontweight="bold", color=COLORS["green"],
                arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2))

    # Top-k (k=5)
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    k = 5
    colors_topk = [COLORS["blue"] if i < k else COLORS["gray_light"]
                   for i in range(len(tokens))]
    ax.barh(range(len(tokens)), probs, color=colors_topk,
            edgecolor=COLORS["dark"], linewidth=0.8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title(f"Top-k Sampling (k={k})\n(Sample from top {k})",
                 fontsize=12, fontweight="bold", color=COLORS["blue"])
    ax.set_xlabel("Probability", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    # Draw cutoff line
    ax.axhline(y=k - 0.5, color=COLORS["red"], linewidth=2, linestyle="--")
    ax.text(0.25, k - 0.3, "Cutoff", fontsize=10, color=COLORS["red"],
            fontweight="bold")

    # Top-p (p=0.9)
    ax = axes[2]
    ax.set_facecolor(COLORS["bg"])
    cumsum = np.cumsum(probs)
    p_threshold = 0.9
    nucleus_size = np.searchsorted(cumsum, p_threshold) + 1
    colors_topp = [COLORS["purple"] if i < nucleus_size else COLORS["gray_light"]
                   for i in range(len(tokens))]
    ax.barh(range(len(tokens)), probs, color=colors_topp,
            edgecolor=COLORS["dark"], linewidth=0.8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title(f"Top-p / Nucleus (p={p_threshold})\n(Cumulative prob ≥ {p_threshold})",
                 fontsize=12, fontweight="bold", color=COLORS["purple"])
    ax.set_xlabel("Probability", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # Show cumulative probabilities
    for i in range(len(tokens)):
        label = f"Σ={cumsum[i]:.2f}"
        color = COLORS["purple"] if i < nucleus_size else COLORS["gray"]
        ax.text(probs[i] + 0.01, i, label, va="center", fontsize=8,
                color=color, fontstyle="italic")

    ax.axhline(y=nucleus_size - 0.5, color=COLORS["red"],
               linewidth=2, linestyle="--")
    ax.text(0.25, nucleus_size - 0.3, f"p={p_threshold} cutoff",
            fontsize=10, color=COLORS["red"], fontweight="bold")

    fig.suptitle("Sampling Strategies for Text Generation",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "viz_sampling_strategies.png")


# ---------------------------------------------------------------------------
# 7. GPT-2 Architecture Comparison
# ---------------------------------------------------------------------------
def viz_gpt2_architecture():
    """Compare GPT-2 model variants side by side."""
    models = [
        ("DistilGPT-2", 82, 6, 12, 768, "~330 MB"),
        ("GPT-2 Small", 124, 12, 12, 768, "~500 MB"),
        ("GPT-2 Medium", 355, 24, 16, 1024, "~1.4 GB"),
        ("GPT-2 Large", 774, 36, 20, 1280, "~3.1 GB"),
        ("GPT-2 XL", 1558, 48, 25, 1600, "~6.2 GB"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=COLORS["bg"])

    names = [m[0] for m in models]
    x = np.arange(len(names))
    bar_colors = [COLORS["green"], COLORS["blue"], COLORS["orange"],
                  COLORS["purple"], COLORS["red"]]

    # Parameters
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    params = [m[1] for m in models]
    bars = ax.bar(x, params, color=bar_colors, edgecolor=COLORS["dark"],
                  linewidth=1.0, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Parameters (M)", fontsize=11)
    ax.set_title("Parameters", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val}M", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Layers
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    layers = [m[2] for m in models]
    bars = ax.bar(x, layers, color=bar_colors, edgecolor=COLORS["dark"],
                  linewidth=1.0, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Number of Layers", fontsize=11)
    ax.set_title("Transformer Layers", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, layers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Embedding dimension
    ax = axes[2]
    ax.set_facecolor(COLORS["bg"])
    d_model = [m[4] for m in models]
    bars = ax.bar(x, d_model, color=bar_colors, edgecolor=COLORS["dark"],
                  linewidth=1.0, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Embedding Dimension", fontsize=11)
    ax.set_title("d_model", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, d_model):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("GPT-2 Model Variants Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "viz_gpt2_architecture.png")


# ---------------------------------------------------------------------------
# 8. Same Text Through Different Tokenizers
# ---------------------------------------------------------------------------
def viz_tokenizer_comparison():
    """Show how different tokenizers split the same input."""
    text = "Hello world! Let's tokenize this"

    tokenizers = [
        ("Character", list(text)),
        ("Whitespace", text.split()),
        ("BPE (GPT-2 style)",
         ["Hello", " world", "!", " Let", "'s", " token", "ize", " this"]),
        ("WordPiece (BERT style)",
         ["Hello", "world", "!", "Let", "'", "s", "token", "##ize", "this"]),
    ]

    fig, axes = plt.subplots(len(tokenizers), 1, figsize=(14, 8),
                             facecolor=COLORS["bg"])

    row_colors = [
        (COLORS["red_light"], COLORS["red"]),
        (COLORS["gray_light"], COLORS["gray"]),
        (COLORS["blue_light"], COLORS["blue"]),
        (COLORS["green_light"], COLORS["green"]),
    ]

    for idx, (name, tokens) in enumerate(tokenizers):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(-0.5, 22)
        ax.set_ylim(-0.3, 1.3)
        ax.axis("off")

        bg_color, border_color = row_colors[idx]

        # Label
        ax.text(-0.3, 0.5, f"{name}\n({len(tokens)} tokens)",
                ha="right", va="center", fontsize=10, fontweight="bold",
                color=border_color)

        # Draw tokens
        x = 0.0
        for token in tokens:
            display = token.replace(" ", "·")
            w = max(len(display) * 0.35, 0.6)
            box = FancyBboxPatch(
                (x, 0.1), w, 0.7,
                boxstyle="round,pad=0.06",
                facecolor=bg_color, edgecolor=border_color, linewidth=1.3,
            )
            ax.add_patch(box)
            ax.text(x + w / 2, 0.45, display,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=COLORS["dark"],
                    family="monospace")
            x += w + 0.12

    fig.suptitle(f"Same Text Through Different Tokenizers",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.14, 0, 1, 0.95])
    save_figure(fig, "viz_tokenizer_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Tokenization & HuggingFace concept visualizations...")
    print("=" * 60)

    viz_tokenization_comparison()
    viz_bpe_algorithm()
    viz_utf8_encoding()
    viz_vocab_size_tradeoff()
    viz_temperature_effect()
    viz_sampling_strategies()
    viz_gpt2_architecture()
    viz_tokenizer_comparison()

    print("=" * 60)
    print("All 8 visualizations generated successfully!")
