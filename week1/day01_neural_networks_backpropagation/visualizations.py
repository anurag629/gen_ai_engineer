"""
Educational Visualizations for Neural Network Concepts
=======================================================
Generates PNG diagrams covering:
  - Activation functions and their derivatives
  - Computational graphs with forward/backward values
  - Gradient descent on a 2D contour surface
  - Learning rate comparison (1D)
  - Single neuron anatomy
  - MLP architecture
  - Forward/backward training loop
  - 3D loss landscape

Usage:
    python visualizations.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, ArrowStyle
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
plt.style.use("default")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 150

# Consistent colour palette
C_BLUE = "#2176AE"
C_RED = "#D7263D"
C_GREEN = "#2A9D8F"
C_ORANGE = "#F4A261"
C_PURPLE = "#6C63FF"
C_DARK = "#264653"
C_LIGHT_GRAY = "#F0F0F0"


def _save(fig, filename):
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# 1. Activation Functions
# ===================================================================
def viz_activation_functions():
    """Plot ReLU, Sigmoid, Tanh with their derivatives side-by-side."""
    x = np.linspace(-5, 5, 500)

    # ---- function / derivative pairs ----
    def relu(z):
        return np.maximum(0, z)

    def relu_deriv(z):
        return (z > 0).astype(float)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_deriv(z):
        s = sigmoid(z)
        return s * (1 - s)

    def tanh(z):
        return np.tanh(z)

    def tanh_deriv(z):
        return 1 - np.tanh(z) ** 2

    funcs = [
        ("ReLU", relu, relu_deriv, "Range: [0, +inf)\nf'(x) = 0 or 1"),
        ("Sigmoid", sigmoid, sigmoid_deriv, "Range: (0, 1)\nf'(x) max = 0.25"),
        ("Tanh", tanh, tanh_deriv, "Range: (-1, 1)\nf'(x) max = 1.0"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Activation Functions and Their Derivatives", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, fn, fn_d, note) in zip(axes, funcs):
        ax.plot(x, fn(x), color=C_BLUE, linewidth=2.5, label=f"{name}(x)")
        ax.plot(x, fn_d(x), color=C_RED, linewidth=2.0, linestyle="--", label=f"{name}'(x)")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        # Annotation box
        ax.text(
            0.97, 0.03, note,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT_GRAY, edgecolor="grey", alpha=0.9),
        )

    fig.tight_layout()
    _save(fig, "viz_activation_functions.png")


# ===================================================================
# 2. Computational Graph
# ===================================================================
def viz_computational_graph():
    """Draw the computation graph for L = (a*b + c) * f with forward values and gradients."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Computational Graph:  L = (a * b + c) * f", fontsize=15, fontweight="bold")

    # --- helpers ---
    def draw_var(x, y, label, value, grad, is_input=False):
        """Draw a rounded rectangle for a variable node."""
        w, h = 1.4, 0.9
        color = C_GREEN if is_input else "#E8E8E8"
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12", facecolor=color, edgecolor=C_DARK, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.12, label, ha="center", va="center", fontsize=12, fontweight="bold", color=C_DARK)
        ax.text(x, y - 0.25, f"= {value}", ha="center", va="center", fontsize=10, color=C_BLUE)
        ax.text(x, y + 0.62, f"dL/d{label} = {grad}", ha="center", va="center", fontsize=9.5, color=C_RED, fontweight="bold")

    def draw_op(x, y, symbol):
        """Draw a circle for an operation node."""
        circ = plt.Circle((x, y), 0.45, facecolor="#FFE4B5", edgecolor=C_DARK, linewidth=1.5)
        ax.add_patch(circ)
        ax.text(x, y, symbol, ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.5),
        )

    # Coordinates
    # Inputs:  a(1,5)  b(1,3)  c(1,1)  f(7,1)
    # Ops:     *(3,4)  +(5,3)  *(9,3)
    # Intermediates: e=a*b(3,4 result -> 5,3), d=e+c(5,3 result -> 9,3)
    # Output: L(11,3)

    # Forward: a=2, b=-3, c=10, f=-2
    # e = a*b = -6,  d = e+c = 4,  L = d*f = -8
    # Backward: dL/dL=1
    # dL/dd = f = -2,  dL/df = d = 4
    # dL/de = dL/dd * 1 = -2,  dL/dc = dL/dd * 1 = -2
    # dL/da = dL/de * b = -2*-3 = 6,  dL/db = dL/de * a = -2*2 = -4

    # Draw input variables
    draw_var(0.5, 5, "a", 2, 6, is_input=True)
    draw_var(0.5, 3, "b", -3, -4, is_input=True)
    draw_var(0.5, 1, "c", 10, -2, is_input=True)
    draw_var(7, 0.5, "f", -2, 4, is_input=True)

    # Draw operations
    draw_op(3, 4, "*")
    draw_op(5.5, 2.5, "+")
    draw_op(9, 2.5, "*")

    # Draw intermediate + output
    draw_var(4.2, 4.5, "e", -6, -2)
    draw_var(7, 3, "d", 4, -2)
    draw_var(11.5, 2.5, "L", -8, 1)

    # Arrows: inputs -> ops
    arrow(1.2, 5, 2.55, 4.2)
    arrow(1.2, 3, 2.55, 3.8)
    arrow(1.2, 1, 5.05, 2.3)

    # e -> + op
    arrow(4.9, 4.3, 5.3, 2.95)
    # + op -> d
    arrow(5.95, 2.5, 6.3, 2.85)
    # f -> * op
    arrow(7.7, 0.7, 8.65, 2.1)
    # d -> * op
    arrow(7.7, 2.9, 8.55, 2.6)
    # * op -> L
    arrow(9.45, 2.5, 10.8, 2.5)

    # * op output -> e (label)
    arrow(3.45, 4.0, 3.5, 4.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_GREEN, edgecolor=C_DARK, label="Input variable"),
        mpatches.Patch(facecolor="#FFE4B5", edgecolor=C_DARK, label="Operation"),
        mpatches.Patch(facecolor="#E8E8E8", edgecolor=C_DARK, label="Intermediate / Output"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)
    # Color key
    ax.text(0.0, -0.6, "Blue values = forward pass", fontsize=11, color=C_BLUE, fontweight="bold")
    ax.text(5.5, -0.6, "Red values = gradients (backward pass)", fontsize=11, color=C_RED, fontweight="bold")

    fig.tight_layout()
    _save(fig, "viz_computational_graph.png")


# ===================================================================
# 3. Gradient Descent on 2D Contour
# ===================================================================
def viz_gradient_descent():
    """Show gradient descent optimisation path on a 2D loss contour."""
    fig, ax = plt.subplots(figsize=(9, 8))

    # Loss surface: L(w1, w2) = 3*(w1-1)^2 + (w2-2)^2 + 2*w1*w2 - 4*w1 - 6*w2
    # Simpler: use a rotated quadratic bowl
    w1 = np.linspace(-4, 6, 400)
    w2 = np.linspace(-3, 7, 400)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss: an elongated bowl with minimum near (1, 2)
    L = 2 * (W1 - 1) ** 2 + 0.5 * (W2 - 2) ** 2 + 0.8 * (W1 - 1) * (W2 - 2)

    levels = np.linspace(0, 40, 25)
    contour = ax.contourf(W1, W2, L, levels=levels, cmap="Blues", alpha=0.7)
    ax.contour(W1, W2, L, levels=levels, colors="grey", linewidths=0.5, alpha=0.5)
    fig.colorbar(contour, ax=ax, label="Loss", shrink=0.8)

    # Gradient descent path
    def grad(w):
        dw1 = 4 * (w[0] - 1) + 0.8 * (w[1] - 2)
        dw2 = 1 * (w[1] - 2) + 0.8 * (w[0] - 1)
        return np.array([dw1, dw2])

    lr = 0.15
    w = np.array([5.0, 6.0])
    path = [w.copy()]
    for _ in range(30):
        g = grad(w)
        w = w - lr * g
        path.append(w.copy())

    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=C_RED, markersize=5, linewidth=1.8, label="GD path")

    # Arrows between consecutive points
    for i in range(min(12, len(path) - 1)):
        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]
        ax.annotate(
            "", xy=(path[i + 1, 0], path[i + 1, 1]),
            xytext=(path[i, 0], path[i, 1]),
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.5),
        )

    ax.plot(1, 2, marker="*", markersize=20, color=C_ORANGE, zorder=5, label="Minimum (1, 2)")
    ax.plot(path[0, 0], path[0, 1], marker="s", markersize=10, color=C_PURPLE, zorder=5, label="Start (5, 6)")

    ax.set_xlabel("Weight $w_1$", fontsize=13)
    ax.set_ylabel("Weight $w_2$", fontsize=13)
    ax.set_title("Gradient Descent on a 2D Loss Surface", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "viz_gradient_descent.png")


# ===================================================================
# 4. Learning Rate Comparison (1D)
# ===================================================================
def viz_learning_rates():
    """Show effect of different learning rates on 1D loss curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simple 1D loss: L(w) = (w - 3)^2 + 1
    w_range = np.linspace(-3, 10, 500)
    loss_fn = lambda w: (w - 3) ** 2 + 1
    grad_fn = lambda w: 2 * (w - 3)

    ax.plot(w_range, loss_fn(w_range), color=C_DARK, linewidth=2.5, label="Loss = $(w-3)^2 + 1$")

    configs = [
        ("Too small LR = 0.02", 0.02, -2.0, C_ORANGE, 40),
        ("Good LR = 0.3", 0.3, -2.0, C_GREEN, 15),
        ("Too large LR = 1.05", 1.05, -2.0, C_RED, 15),
    ]

    for label, lr, w0, color, steps in configs:
        w = w0
        ws = [w]
        for _ in range(steps):
            g = grad_fn(w)
            w = w - lr * g
            ws.append(w)
        ws = np.array(ws)
        losses = loss_fn(ws)
        ax.plot(ws, losses, "o-", color=color, markersize=5, linewidth=1.5, label=label, alpha=0.85)
        # Start marker
        ax.plot(ws[0], losses[0], "s", color=color, markersize=9, zorder=5)

    ax.plot(3, 1, marker="*", markersize=18, color="gold", zorder=5, markeredgecolor="black", label="Minimum")
    ax.set_xlabel("Weight $w$", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Effect of Learning Rate on Gradient Descent (1D)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 60)

    fig.tight_layout()
    _save(fig, "viz_learning_rates.png")


# ===================================================================
# 5. Single Neuron Anatomy
# ===================================================================
def viz_neuron():
    """Draw the anatomy of a single artificial neuron."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Anatomy of a Single Neuron", fontsize=16, fontweight="bold")

    # --- Inputs ---
    input_names = ["$x_1$", "$x_2$", "$x_3$"]
    weight_names = ["$w_1$", "$w_2$", "$w_3$"]
    input_ys = [4, 2, 0]

    for i, (name, wname, iy) in enumerate(zip(input_names, weight_names, input_ys)):
        # Input box
        box = FancyBboxPatch(
            (-0.7, iy - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.15", facecolor="#D4EDDA", edgecolor=C_DARK, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(0, iy, name, ha="center", va="center", fontsize=14, fontweight="bold")

        # Arrow with weight label
        ax.annotate(
            "", xy=(4.0, 2.0), xytext=(0.9, iy),
            arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2),
        )
        # Weight label on the arrow
        mid_x = (0.9 + 4.0) / 2
        mid_y = (iy + 2.0) / 2
        offset_x = 0.15 if i == 1 else -0.15
        offset_y = 0.3 if i == 0 else (-0.3 if i == 2 else 0.35)
        ax.text(
            mid_x + offset_x, mid_y + offset_y, wname,
            ha="center", va="center", fontsize=12, color=C_BLUE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
        )

    # --- Summation node ---
    sigma = plt.Circle((5, 2), 0.8, facecolor="#FFE4B5", edgecolor=C_DARK, linewidth=2)
    ax.add_patch(sigma)
    ax.text(5, 2, r"$\Sigma$", ha="center", va="center", fontsize=22, fontweight="bold")

    # --- Bias arrow coming from below ---
    ax.annotate(
        "", xy=(5, 1.2), xytext=(5, -0.5),
        arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=2),
    )
    bias_box = FancyBboxPatch(
        (4.3, -1.3), 1.4, 0.7,
        boxstyle="round,pad=0.12", facecolor="#E8DAEF", edgecolor=C_DARK, linewidth=1.5,
    )
    ax.add_patch(bias_box)
    ax.text(5, -0.95, "$b$", ha="center", va="center", fontsize=14, fontweight="bold", color=C_PURPLE)

    # --- Arrow from summation to activation ---
    ax.annotate(
        "", xy=(8.0, 2.0), xytext=(5.8, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2),
    )
    ax.text(6.9, 2.4, r"$z = \sum w_i x_i + b$", ha="center", va="center", fontsize=11, color=C_DARK)

    # --- Activation function box ---
    act_box = FancyBboxPatch(
        (8, 1.2), 2.5, 1.6,
        boxstyle="round,pad=0.2", facecolor="#D6EAF8", edgecolor=C_DARK, linewidth=2,
    )
    ax.add_patch(act_box)
    ax.text(9.25, 2.0, r"$\sigma(z)$", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(9.25, 1.45, "Activation", ha="center", va="center", fontsize=10, color="grey")

    # --- Output arrow ---
    ax.annotate(
        "", xy=(13, 2.0), xytext=(10.5, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.5),
    )
    # Output box
    out_box = FancyBboxPatch(
        (13, 1.5), 1.5, 1.0,
        boxstyle="round,pad=0.15", facecolor="#FADBD8", edgecolor=C_DARK, linewidth=1.5,
    )
    ax.add_patch(out_box)
    ax.text(13.75, 2.0, "$y$", ha="center", va="center", fontsize=16, fontweight="bold")

    ax.text(11.75, 2.4, r"$y = \sigma(z)$", ha="center", va="center", fontsize=11, color=C_DARK)

    fig.tight_layout()
    _save(fig, "viz_neuron.png")


# ===================================================================
# 6. MLP Architecture
# ===================================================================
def viz_mlp_architecture():
    """Draw a full MLP: 3 inputs, 4-4 hidden, 1 output."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Multi-Layer Perceptron (MLP) Architecture", fontsize=16, fontweight="bold")

    layer_sizes = [3, 4, 4, 1]
    layer_names = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
    layer_colors = [C_GREEN, C_BLUE, C_BLUE, C_RED]
    layer_face = ["#D4EDDA", "#D6EAF8", "#D6EAF8", "#FADBD8"]
    x_positions = [1, 4, 7, 10]

    # Compute y positions for each layer (centered)
    max_neurons = max(layer_sizes)
    spacing = 1.6

    def get_ys(n):
        total_height = (n - 1) * spacing
        start = (max_neurons - 1) * spacing / 2 - total_height / 2 + 1.5
        return [start + i * spacing for i in range(n)]

    all_positions = []
    for li, (n, xp) in enumerate(zip(layer_sizes, x_positions)):
        ys = get_ys(n)
        positions = [(xp, y) for y in ys]
        all_positions.append(positions)

    # Draw connections first (behind neurons)
    for li in range(len(layer_sizes) - 1):
        for x1, y1 in all_positions[li]:
            for x2, y2 in all_positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="grey", linewidth=0.6, alpha=0.5, zorder=1)

    # Draw neurons
    radius = 0.35
    for li, (n, xp) in enumerate(zip(layer_sizes, x_positions)):
        for ni, (nx, ny) in enumerate(all_positions[li]):
            circ = plt.Circle(
                (nx, ny), radius,
                facecolor=layer_face[li], edgecolor=layer_colors[li], linewidth=2, zorder=3,
            )
            ax.add_patch(circ)

            # Labels inside neurons
            if li == 0:
                ax.text(nx, ny, f"$x_{ni+1}$", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
            elif li == len(layer_sizes) - 1:
                ax.text(nx, ny, "$y$", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
            else:
                ax.text(nx, ny, f"$h_{{{li},{ni+1}}}$", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

        # Layer label below
        bottom_y = min(p[1] for p in all_positions[li])
        ax.text(xp, bottom_y - 1.0, layer_names[li], ha="center", va="center", fontsize=11,
                fontweight="bold", color=layer_colors[li])

    # Size annotations
    for li in range(len(layer_sizes)):
        top_y = max(p[1] for p in all_positions[li])
        ax.text(x_positions[li], top_y + 0.8, f"n={layer_sizes[li]}", ha="center", va="center",
                fontsize=10, color="grey",
                bbox=dict(facecolor="white", edgecolor="grey", alpha=0.7, pad=2, boxstyle="round"))

    fig.tight_layout()
    _save(fig, "viz_mlp_architecture.png")


# ===================================================================
# 7. Forward / Backward Pass Diagram
# ===================================================================
def viz_forward_backward():
    """Flowchart style diagram of the training loop."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Neural Network Training Loop", fontsize=16, fontweight="bold")

    # --- Boxes ---
    def draw_box(x, y, w, h, text, color, text_color=C_DARK, fontsize=13):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.25", facecolor=color, edgecolor=C_DARK, linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color)

    # Forward pass boxes
    bx = [1.5, 5, 8.5, 12, 15.5]
    by_top = 4.0

    draw_box(bx[0], by_top, 2.5, 1.3, "Input\n$X$", "#D4EDDA")
    draw_box(bx[1], by_top, 2.5, 1.3, "Hidden\nLayers", "#D6EAF8")
    draw_box(bx[2], by_top, 2.5, 1.3, "Output\n$\\hat{y}$", "#D6EAF8")
    draw_box(bx[3], by_top, 2.5, 1.3, "Loss\n$\\mathcal{L}$", "#FADBD8")

    # Forward arrows (top row)
    for i in range(3):
        ax.annotate(
            "", xy=(bx[i + 1] - 1.3, by_top), xytext=(bx[i] + 1.3, by_top),
            arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=2.5),
        )

    # Label: Forward Pass
    ax.text(5, by_top + 1.3, "FORWARD PASS", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_BLUE,
            bbox=dict(facecolor="#E8F4FD", edgecolor=C_BLUE, boxstyle="round,pad=0.3"))

    # Curve down from Loss
    ax.annotate(
        "", xy=(bx[3], by_top - 0.65), xytext=(bx[3], by_top - 0.65),
    )
    # Down arrow from Loss
    ax.annotate(
        "", xy=(bx[3], 1.0 + 0.65), xytext=(bx[3], by_top - 0.65),
        arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5, connectionstyle="arc3,rad=0"),
    )

    # Backward pass boxes (bottom row)
    by_bot = 1.0
    draw_box(bx[3], by_bot, 2.5, 1.3, "Compute\nGradients", "#FCE4E4")
    draw_box(bx[2], by_bot, 2.5, 1.3, "Backprop\nthrough layers", "#FCE4E4")
    draw_box(bx[1], by_bot, 2.5, 1.3, "Update\nWeights", "#FFF3CD")

    # Backward arrows (bottom row, right to left)
    for i in [3, 2]:
        ax.annotate(
            "", xy=(bx[i - 1] + 1.3, by_bot), xytext=(bx[i] - 1.3, by_bot),
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.5),
        )

    # Label: Backward Pass
    ax.text(10, by_bot - 1.3, "BACKWARD PASS", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_RED,
            bbox=dict(facecolor="#FDE8E8", edgecolor=C_RED, boxstyle="round,pad=0.3"))

    # Loop arrow from Update Weights back to Input
    ax.annotate(
        "", xy=(bx[0], by_top - 0.65), xytext=(bx[1] - 1.3, by_bot),
        arrowprops=dict(
            arrowstyle="-|>", color=C_GREEN, lw=2.5,
            connectionstyle="arc3,rad=0.4",
        ),
    )
    ax.text(1.5, 2.0, "Repeat", ha="center", va="center", fontsize=12, fontweight="bold", color=C_GREEN,
            bbox=dict(facecolor="#E8F8E8", edgecolor=C_GREEN, boxstyle="round,pad=0.2"))

    # Annotations
    ax.text(8.5, by_top + 1.3, r"$\hat{y} = f(X; W)$", ha="center", fontsize=12, color=C_BLUE)
    ax.text(12, by_top + 1.3, r"$\mathcal{L}(\hat{y}, y)$", ha="center", fontsize=12, color=C_RED)
    ax.text(5, by_bot - 1.3, r"$W \leftarrow W - \eta \nabla_W \mathcal{L}$", ha="center",
            fontsize=13, color=C_DARK, fontstyle="italic")

    fig.tight_layout()
    _save(fig, "viz_forward_backward.png")


# ===================================================================
# 8. 3D Loss Landscape
# ===================================================================
def viz_loss_landscape_3d():
    """3D surface plot of a loss function with minimum marked."""
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    w1 = np.linspace(-4, 4, 200)
    w2 = np.linspace(-4, 4, 200)
    W1, W2 = np.meshgrid(w1, w2)

    # Bowl-shaped loss with some character
    L = 0.5 * W1 ** 2 + 0.8 * W2 ** 2 + 0.3 * W1 * W2 + 0.1 * np.sin(2 * W1) * np.cos(2 * W2)

    surf = ax.plot_surface(W1, W2, L, cmap="coolwarm", alpha=0.85, edgecolor="none", antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="Loss", pad=0.1)

    # Mark the minimum
    min_idx = np.unravel_index(np.argmin(L), L.shape)
    min_w1 = W1[min_idx]
    min_w2 = W2[min_idx]
    min_L = L[min_idx]
    ax.scatter([min_w1], [min_w2], [min_L], color=C_ORANGE, s=200, marker="*", edgecolors="black",
               linewidths=1, zorder=10, label=f"Minimum ({min_w1:.1f}, {min_w2:.1f})")

    ax.set_xlabel("\nWeight $w_1$", fontsize=12, labelpad=10)
    ax.set_ylabel("\nWeight $w_2$", fontsize=12, labelpad=10)
    ax.set_zlabel("\nLoss", fontsize=12, labelpad=10)
    ax.set_title("3D Loss Landscape", fontsize=15, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="upper right")
    ax.view_init(elev=30, azim=225)

    fig.tight_layout()
    _save(fig, "viz_loss_landscape_3d.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating neural network visualizations ...")
    print(f"Output directory: {SAVE_DIR}\n")

    viz_activation_functions()
    viz_computational_graph()
    viz_gradient_descent()
    viz_learning_rates()
    viz_neuron()
    viz_mlp_architecture()
    viz_forward_backward()
    viz_loss_landscape_3d()

    print("\nAll visualizations generated successfully!")
