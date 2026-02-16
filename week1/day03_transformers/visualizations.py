"""
Educational visualizations for Transformer architecture concepts.
Generates PNG diagrams illustrating key components of the Transformer model.

Uses only matplotlib and numpy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

# Consistent styling
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
    print(f"Saved: {filepath}")


# ---------------------------------------------------------------------------
# 1. Self-Attention Heatmap
# ---------------------------------------------------------------------------
def viz_self_attention():
    tokens = ["the", "cat", "sat", "down"]
    # Hand-crafted attention weights to tell a story:
    # "cat" attends strongly to "the", "sat" attends to "cat", etc.
    attention = np.array(
        [
            [0.70, 0.10, 0.10, 0.10],  # "the" mostly attends to itself
            [0.45, 0.35, 0.10, 0.10],  # "cat" attends strongly to "the"
            [0.10, 0.50, 0.30, 0.10],  # "sat" attends to "cat"
            [0.05, 0.15, 0.40, 0.40],  # "down" attends to "sat" and itself
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    im = ax.imshow(attention, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=13, fontweight="bold")
    ax.set_yticklabels(tokens, fontsize=13, fontweight="bold")
    ax.set_xlabel("Key (attending to)", fontsize=12)
    ax.set_ylabel("Query (token)", fontsize=12)

    # Annotate each cell with the weight value
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            color = "white" if attention[i, j] > 0.45 else COLORS["dark"]
            ax.text(
                j, i, f"{attention[i, j]:.2f}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=color,
            )

    ax.set_title("Self-Attention Weights", fontsize=16, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=11)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    fig.tight_layout()
    save_figure(fig, "viz_self_attention.png")


# ---------------------------------------------------------------------------
# 2. Positional Encoding
# ---------------------------------------------------------------------------
def viz_positional_encoding():
    max_pos = 50
    d_model = 64
    pe = np.zeros((max_pos, d_model))

    position = np.arange(max_pos)[:, np.newaxis]  # (50, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS["bg"])

    # -- Subplot 1: full heatmap --
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    im = ax.imshow(pe, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Position", fontsize=12)
    ax.set_title("Sinusoidal Positional Encodings", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -- Subplot 2: individual waves --
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg"])
    dims_to_show = [0, 1, 4, 5, 20, 21]
    wave_colors = [COLORS["blue"], COLORS["blue_light"],
                   COLORS["green"], COLORS["green_light"],
                   COLORS["red"], COLORS["red_light"]]
    labels = [
        "sin(dim 0)", "cos(dim 1)",
        "sin(dim 4)", "cos(dim 5)",
        "sin(dim 20)", "cos(dim 21)",
    ]
    for idx, dim in enumerate(dims_to_show):
        ax2.plot(pe[:, dim], label=labels[idx], color=wave_colors[idx], linewidth=1.8)

    ax2.set_xlabel("Position", fontsize=12)
    ax2.set_ylabel("Encoding Value", fontsize=12)
    ax2.set_title("Individual Positional Waves", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "viz_positional_encoding.png")


# ---------------------------------------------------------------------------
# 3. Transformer Decoder Block (flowchart)
# ---------------------------------------------------------------------------
def viz_transformer_block():
    fig, ax = plt.subplots(figsize=(6, 12), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box_width = 3.2
    box_height = 0.7
    center_x = 3.0
    left = center_x - box_width / 2

    # Define blocks from bottom to top
    blocks = [
        ("Input\nEmbedding", 0.8, COLORS["gray_light"], COLORS["dark"]),
        ("Layer Norm", 2.2, COLORS["orange_light"], COLORS["dark"]),
        ("Multi-Head\nSelf-Attention", 3.6, COLORS["blue_light"], COLORS["dark"]),
        ("Add & Residual", 5.2, COLORS["green_light"], COLORS["dark"]),
        ("Layer Norm", 6.6, COLORS["orange_light"], COLORS["dark"]),
        ("Feed Forward\nNetwork (FFN)", 8.0, COLORS["purple_light"], COLORS["dark"]),
        ("Add & Residual", 9.6, COLORS["green_light"], COLORS["dark"]),
        ("Output", 11.0, COLORS["gray_light"], COLORS["dark"]),
    ]

    box_objects = []
    for label, y, bg, fg in blocks:
        h = 0.9 if "\n" in label else box_height
        box = FancyBboxPatch(
            (left, y - h / 2), box_width, h,
            boxstyle="round,pad=0.12",
            facecolor=bg, edgecolor=COLORS["dark"], linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            center_x, y, label,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=fg, linespacing=1.3,
        )
        box_objects.append((center_x, y, h))

    # Draw arrows between consecutive blocks
    for i in range(len(box_objects) - 1):
        _, y1, h1 = box_objects[i]
        _, y2, h2 = box_objects[i + 1]
        ax.annotate(
            "", xy=(center_x, y2 - h2 / 2 - 0.05),
            xytext=(center_x, y1 + h1 / 2 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["dark"],
                            lw=1.8, mutation_scale=15),
        )

    # Draw residual connections (curved arrows on the right)
    # Residual 1: from after Input (block 0 top) to Add (block 3)
    res_x = left + box_width + 0.5
    # Residual around attention
    y_start_1 = blocks[1][1]  # Layer Norm
    y_end_1 = blocks[3][1]    # Add & Residual
    ax.annotate(
        "", xy=(res_x - 0.3, y_end_1),
        xytext=(res_x - 0.3, y_start_1),
        arrowprops=dict(
            arrowstyle="-|>", color=COLORS["green"],
            lw=2.0, mutation_scale=14,
            connectionstyle="arc3,rad=-0.4",
        ),
    )
    ax.text(res_x + 0.15, (y_start_1 + y_end_1) / 2, "residual",
            fontsize=9, color=COLORS["green"], rotation=-90,
            ha="center", va="center", fontstyle="italic")

    # Residual around FFN
    y_start_2 = blocks[4][1]  # Layer Norm
    y_end_2 = blocks[6][1]    # Add & Residual
    ax.annotate(
        "", xy=(res_x - 0.3, y_end_2),
        xytext=(res_x - 0.3, y_start_2),
        arrowprops=dict(
            arrowstyle="-|>", color=COLORS["green"],
            lw=2.0, mutation_scale=14,
            connectionstyle="arc3,rad=-0.4",
        ),
    )
    ax.text(res_x + 0.15, (y_start_2 + y_end_2) / 2, "residual",
            fontsize=9, color=COLORS["green"], rotation=-90,
            ha="center", va="center", fontstyle="italic")

    ax.set_title("Transformer Decoder Block", fontsize=16, fontweight="bold",
                 pad=15, y=0.97)
    fig.tight_layout()
    save_figure(fig, "viz_transformer_block.png")


# ---------------------------------------------------------------------------
# 4. Multi-Head Attention Diagram
# ---------------------------------------------------------------------------
def viz_multi_head_attention():
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, color, label, fontsize=10, text_color=None):
        tc = text_color or COLORS["dark"]
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=COLORS["dark"], linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=tc, linespacing=1.2)
        return (x, y, w, h)

    def arrow(x1, y1, x2, y2, color=COLORS["dark"]):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            mutation_scale=12),
        )

    # Input box
    draw_box(0, 2.5, 1.5, 1.5, COLORS["gray_light"], "Input\nX", fontsize=12)

    # Arrows from input to heads
    head_colors_q = [COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["blue"]]
    head_colors_k = [COLORS["green"], COLORS["green"], COLORS["green"], COLORS["green"]]
    head_colors_v = [COLORS["red"], COLORS["red"], COLORS["red"], COLORS["red"]]
    head_labels = ["Head 1", "Head 2", "Head 3", "Head 4"]

    head_x_start = 3.0
    head_spacing = 2.8
    qkv_w = 0.6
    qkv_h = 0.6
    head_box_w = 2.2
    head_box_h = 4.5

    for i in range(4):
        hx = head_x_start + i * head_spacing
        hy = 1.0

        # Head bounding box (subtle)
        head_rect = FancyBboxPatch(
            (hx - 0.15, hy - 0.15), head_box_w + 0.3, head_box_h + 0.3,
            boxstyle="round,pad=0.1",
            facecolor=COLORS["white"], edgecolor=COLORS["gray"],
            linewidth=1.0, linestyle="--", alpha=0.6,
        )
        ax.add_patch(head_rect)

        # Head label
        ax.text(hx + head_box_w / 2, hy + head_box_h + 0.4, head_labels[i],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=COLORS["dark"])

        # Q, K, V boxes inside head
        qkv_y = hy + 0.3
        draw_box(hx + 0.05, qkv_y, qkv_w, qkv_h, COLORS["blue_light"], "Q", fontsize=10)
        draw_box(hx + 0.05 + qkv_w + 0.15, qkv_y, qkv_w, qkv_h, COLORS["green_light"], "K", fontsize=10)
        draw_box(hx + 0.05 + 2 * (qkv_w + 0.15), qkv_y, qkv_w, qkv_h, COLORS["red_light"], "V", fontsize=10)

        # Attention box
        att_y = qkv_y + qkv_h + 0.5
        draw_box(hx + 0.1, att_y, head_box_w - 0.2, 0.9,
                 COLORS["purple_light"], "Scaled Dot-Product\nAttention", fontsize=9)

        # Output of attention
        out_y = att_y + 0.9 + 0.4
        draw_box(hx + 0.4, out_y, head_box_w - 0.8, 0.6,
                 COLORS["orange_light"], "Output", fontsize=9)

        # Arrows: Q,K,V -> Attention
        for j in range(3):
            bx = hx + 0.05 + j * (qkv_w + 0.15) + qkv_w / 2
            arrow(bx, qkv_y + qkv_h, bx, att_y)

        # Arrow: Attention -> Output
        arrow(hx + head_box_w / 2, att_y + 0.9, hx + head_box_w / 2, out_y)

        # Arrow from Input to this head
        arrow(1.5, 3.25, hx, hy + head_box_h / 2)

    # Concat box
    concat_x = head_x_start + 4 * head_spacing + 0.3
    draw_box(concat_x, 2.0, 1.4, 1.2, COLORS["orange_light"], "Concat", fontsize=11)

    # Arrows from each head output to concat
    for i in range(4):
        hx = head_x_start + i * head_spacing
        out_y = 1.0 + 0.3 + qkv_h + 0.5 + 0.9 + 0.4
        arrow(hx + head_box_w - 0.4, out_y + 0.3, concat_x, 2.6)

    # Linear projection
    lin_x = concat_x + 1.4 + 0.4
    draw_box(lin_x, 2.0, 1.4, 1.2, COLORS["blue_light"], "Linear\nW_O", fontsize=11)
    arrow(concat_x + 1.4, 2.6, lin_x, 2.6)

    # Final output
    final_x = lin_x + 1.4 + 0.4
    draw_box(final_x, 2.2, 1.2, 0.8, COLORS["gray_light"], "Output", fontsize=11)
    arrow(lin_x + 1.4, 2.6, final_x, 2.6)

    ax.set_title("Multi-Head Attention Mechanism (h=4 heads)",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, "viz_multi_head_attention.png")


# ---------------------------------------------------------------------------
# 5. Attention Patterns
# ---------------------------------------------------------------------------
def viz_attention_patterns():
    n = 6
    patterns = {}

    # (a) attend to previous token (shifted diagonal)
    prev_token = np.zeros((n, n))
    for i in range(n):
        prev_token[i, max(0, i - 1)] = 1.0
    patterns["(a) Attend to\nPrevious Token"] = prev_token

    # (b) attend to first token
    first_token = np.full((n, n), 0.02)
    first_token[:, 0] = 1.0
    # renormalize rows
    first_token = first_token / first_token.sum(axis=1, keepdims=True)
    patterns["(b) Attend to\nFirst Token"] = first_token

    # (c) attend to self (diagonal)
    self_attn = np.eye(n)
    patterns["(c) Attend to\nSelf"] = self_attn

    # (d) uniform attention
    uniform = np.ones((n, n)) / n
    patterns["(d) Uniform\nAttention"] = uniform

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor=COLORS["bg"])

    token_labels = [f"t{i}" for i in range(n)]

    for idx, (title, pattern) in enumerate(patterns.items()):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])
        im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(token_labels, fontsize=9)
        ax.set_yticklabels(token_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Key", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Query", fontsize=10)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = pattern[i, j]
                color = "white" if val > 0.5 else COLORS["dark"]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.suptitle("Different Attention Patterns Learned by Different Heads",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    save_figure(fig, "viz_attention_patterns.png")


# ---------------------------------------------------------------------------
# 6. Transformer vs MLP Training Loss
# ---------------------------------------------------------------------------
def viz_transformer_vs_mlp():
    np.random.seed(42)
    steps = np.arange(0, 5001, 50)

    # MLP: slower convergence, higher final loss
    mlp_loss = 2.1 + 1.2 * np.exp(-steps / 1800) + np.random.normal(0, 0.02, len(steps))
    mlp_loss = np.clip(mlp_loss, 2.05, 3.35)

    # Transformer: faster convergence, lower final loss
    tf_loss = 1.8 + 1.5 * np.exp(-steps / 1000) + np.random.normal(0, 0.015, len(steps))
    tf_loss = np.clip(tf_loss, 1.75, 3.35)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    ax.plot(steps, mlp_loss, color=COLORS["red"], linewidth=2.2,
            label="MLP Language Model", alpha=0.9)
    ax.plot(steps, tf_loss, color=COLORS["blue"], linewidth=2.2,
            label="Transformer", alpha=0.9)

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training Loss: Transformer vs MLP Language Model",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.5, 3.5)
    ax.set_xlim(0, 5000)

    # Add annotations
    ax.annotate("Transformer converges\nfaster and lower",
                xy=(3500, tf_loss[70]), xytext=(3000, 2.5),
                fontsize=10, color=COLORS["blue"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.5))

    fig.tight_layout()
    save_figure(fig, "viz_transformer_vs_mlp.png")


# ---------------------------------------------------------------------------
# 7. Q, K, V Computation
# ---------------------------------------------------------------------------
def viz_qkv_computation():
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1, 6.5)
    ax.axis("off")

    def draw_matrix(x, y, rows, cols, color, label, dim_label="", values=None):
        cell_w = 0.45
        cell_h = 0.45
        for r in range(rows):
            for c in range(cols):
                rect = plt.Rectangle(
                    (x + c * cell_w, y - r * cell_h), cell_w, cell_h,
                    facecolor=color, edgecolor=COLORS["dark"],
                    linewidth=0.8, alpha=0.8,
                )
                ax.add_patch(rect)
                if values is not None:
                    ax.text(x + c * cell_w + cell_w / 2,
                            y - r * cell_h + cell_h / 2,
                            f"{values[r][c]:.1f}",
                            ha="center", va="center", fontsize=7,
                            color=COLORS["dark"])
        # Label above
        ax.text(x + cols * cell_w / 2, y + 0.7, label,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=COLORS["dark"])
        # Dimension label below
        if dim_label:
            ax.text(x + cols * cell_w / 2, y - rows * cell_h - 0.35,
                    dim_label, ha="center", va="center", fontsize=10,
                    color=COLORS["gray"], fontstyle="italic")
        return x + cols * cell_w

    def draw_arrow_label(x1, x2, y, label, color=COLORS["dark"]):
        mid = (x1 + x2) / 2
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                            mutation_scale=14),
        )
        ax.text(mid, y + 0.45, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

    np.random.seed(7)
    X_vals = np.round(np.random.randn(4, 4) * 0.5, 1)

    # Input X
    x_end = draw_matrix(0.5, 4.5, 4, 4, COLORS["gray_light"], "X (Input)",
                        "(seq_len x d_model)\n4 x 4", values=X_vals)

    # Arrows to W_Q, W_K, W_V
    arrow_start_x = x_end + 0.3
    w_x = arrow_start_x + 1.2

    # W_Q
    draw_arrow_label(arrow_start_x, w_x, 5.2, "x W_Q", COLORS["blue"])
    wq_end = draw_matrix(w_x + 0.1, 5.6, 4, 4, COLORS["blue_light"],
                         "W_Q", "(d_model x d_k)\n4 x 4")

    # W_K
    draw_arrow_label(arrow_start_x, w_x, 3.2, "x W_K", COLORS["green"])
    wk_end = draw_matrix(w_x + 0.1, 3.6, 4, 4, COLORS["green_light"],
                         "W_K", "(d_model x d_k)\n4 x 4")

    # W_V
    draw_arrow_label(arrow_start_x, w_x, 1.2, "x W_V", COLORS["red"])
    wv_end = draw_matrix(w_x + 0.1, 1.6, 4, 4, COLORS["red_light"],
                         "W_V", "(d_model x d_v)\n4 x 4")

    # Result arrows
    res_x = max(wq_end, wk_end, wv_end) + 0.5
    out_x = res_x + 1.2

    # Q
    draw_arrow_label(res_x, out_x, 5.2, "=", COLORS["blue"])
    draw_matrix(out_x + 0.1, 5.6, 4, 4, COLORS["blue"], "Q",
                "(seq_len x d_k)\n4 x 4")

    # K
    draw_arrow_label(res_x, out_x, 3.2, "=", COLORS["green"])
    draw_matrix(out_x + 0.1, 3.6, 4, 4, COLORS["green"], "K",
                "(seq_len x d_k)\n4 x 4")

    # V
    draw_arrow_label(res_x, out_x, 1.2, "=", COLORS["red"])
    draw_matrix(out_x + 0.1, 1.6, 4, 4, COLORS["red"], "V",
                "(seq_len x d_v)\n4 x 4")

    ax.set_title("Query, Key, Value Computation: Q = XW_Q,  K = XW_K,  V = XW_V",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "viz_qkv_computation.png")


# ---------------------------------------------------------------------------
# 8. Scaled Dot-Product Attention (step by step)
# ---------------------------------------------------------------------------
def viz_scaled_dot_product():
    np.random.seed(42)
    n = 4
    d_k = 4

    Q = np.round(np.random.randn(n, d_k) * 0.5 + 0.3, 2)
    K = np.round(np.random.randn(n, d_k) * 0.5 + 0.1, 2)
    V = np.round(np.random.randn(n, d_k) * 0.5, 2)

    # Step 1: Q @ K^T
    scores = Q @ K.T

    # Step 2: divide by sqrt(d_k)
    scaled = scores / np.sqrt(d_k)

    # Step 3: mask (causal: upper triangle = -inf)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    masked = scaled.copy()
    masked[mask] = -1e9

    # Step 4: softmax
    def softmax(x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / ex.sum(axis=-1, keepdims=True)

    attn_weights = softmax(masked)

    # Step 5: multiply by V
    output = attn_weights @ V

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=COLORS["bg"])
    axes = axes.flatten()

    steps = [
        ("Step 1: Q x K^T\n(Raw Scores)", scores, "RdBu_r"),
        (f"Step 2: Divide by sqrt(d_k)\n(sqrt({d_k}) = {np.sqrt(d_k):.1f})", scaled, "RdBu_r"),
        ("Step 3: Apply Causal Mask\n(upper triangle = -inf)", masked, "RdBu_r"),
        ("Step 4: Softmax\n(Attention Weights)", attn_weights, "Blues"),
        ("Step 5: Weights x V\n(Output)", output, "RdBu_r"),
    ]

    token_labels = ["t0", "t1", "t2", "t3"]

    for idx, (title, matrix, cmap) in enumerate(steps):
        ax = axes[idx]
        ax.set_facecolor(COLORS["bg"])

        # For the masked matrix, clip display values
        display_mat = matrix.copy()
        if idx == 2:
            display_mat = np.where(mask, np.nan, display_mat)

        vmin = np.nanmin(display_mat)
        vmax = np.nanmax(display_mat)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        im = ax.imshow(display_mat, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

        rows, cols = matrix.shape
        row_labels = token_labels[:rows]
        col_labels = token_labels[:cols]

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=9)

        # Annotate
        for i in range(rows):
            for j in range(cols):
                val = matrix[i, j]
                if idx == 2 and mask[i, j]:
                    ax.text(j, i, "-inf", ha="center", va="center",
                            fontsize=8, color=COLORS["red"], fontweight="bold")
                else:
                    display_val = f"{val:.2f}"
                    text_color = "white" if abs(val - vmin) > 0.6 * (vmax - vmin) else COLORS["dark"]
                    if idx == 3:
                        text_color = "white" if val > 0.4 else COLORS["dark"]
                    ax.text(j, i, display_val, ha="center", va="center",
                            fontsize=8, color=text_color, fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide the 6th subplot
    axes[5].axis("off")
    axes[5].text(0.5, 0.5,
                 "Attention(Q, K, V) =\n\nsoftmax( Q K^T / sqrt(d_k) ) V",
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color=COLORS["dark"],
                 transform=axes[5].transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["blue_light"],
                           edgecolor=COLORS["blue"], alpha=0.3))

    fig.suptitle("Scaled Dot-Product Attention: Step by Step",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "viz_scaled_dot_product.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Transformer concept visualizations...")
    print("=" * 55)

    viz_self_attention()
    viz_positional_encoding()
    viz_transformer_block()
    viz_multi_head_attention()
    viz_attention_patterns()
    viz_transformer_vs_mlp()
    viz_qkv_computation()
    viz_scaled_dot_product()

    print("=" * 55)
    print("All visualizations generated successfully!")
