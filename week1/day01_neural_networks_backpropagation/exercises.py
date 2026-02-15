"""
Day 1 Exercises - Neural Networks & Backpropagation

Work through these exercises after reading the book (README.md)
and building micrograd.py.

Run each exercise one at a time by uncommenting the function call at the bottom.
"""

from micrograd import Value, Neuron, Layer, MLP
import random
import math


# ============================================================
# EXERCISE 1: Verify Gradients Numerically
# ============================================================

def exercise_1_verify_gradients():
    """
    For any expression, verify that backprop gradients match
    numerical (wiggle) gradients.
    """
    print("Exercise 1: Verify Gradients")
    print("=" * 50)

    def check_gradient(label, value_obj, compute_fn, h=1e-5):
        """Compare analytical gradient with numerical gradient."""
        analytical = value_obj.grad

        # Numerical: wiggle and observe
        original = value_obj.data
        value_obj.data = original + h
        loss_plus = compute_fn()
        value_obj.data = original - h
        loss_minus = compute_fn()
        value_obj.data = original  # restore
        numerical = (loss_plus - loss_minus) / (2 * h)

        match = abs(analytical - numerical) < 1e-4
        status = "PASS" if match else "FAIL"
        print(f"  {label:10s} | analytical={analytical:+.6f} | numerical={numerical:+.6f} | {status}")

    # Test expression: L = tanh(a*b + c) * d
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = Value(-2.0, label='d')

    def compute():
        return ((a * b + c).tanh() * d).data

    # Forward + backward
    L = (a * b + c).tanh() * d
    L.backward()

    check_gradient('a', a, compute)
    check_gradient('b', b, compute)
    check_gradient('c', c, compute)
    check_gradient('d', d, compute)

    print("\nIf all PASS, your backprop is correct!\n")


# ============================================================
# EXERCISE 2: XOR Problem
# ============================================================

def exercise_2_xor():
    """
    Train a neural network to learn XOR.
    XOR cannot be solved by a single neuron - you need hidden layers.

    XOR truth table:
      0 XOR 0 = 0
      0 XOR 1 = 1
      1 XOR 0 = 1
      1 XOR 1 = 0
    """
    print("Exercise 2: Learn XOR")
    print("=" * 50)

    random.seed(42)

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    # Using -1 and +1 as targets (works better with tanh)
    ys = [-1.0, 1.0, 1.0, -1.0]

    # TODO: Create a model. Experiment with different architectures:
    #   MLP(2, [4, 1])      - might not work well
    #   MLP(2, [8, 1])      - better
    #   MLP(2, [4, 4, 1])   - should work
    model = MLP(2, [4, 4, 1])
    print(f"Parameters: {len(model.parameters())}")

    # TODO: Train the model
    # Hint: you may need 200-500 epochs and learning_rate around 0.1
    learning_rate = 0.1

    for epoch in range(500):
        predictions = [model(x) for x in xs]
        loss = sum((pred - target) ** 2 for pred, target in zip(predictions, ys))

        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.data:.6f}")

    print(f"\nFinal Results:")
    for x, y in zip(xs, ys):
        pred = model(x)
        expected = "True" if y > 0 else "False"
        got = "True" if pred.data > 0 else "False"
        print(f"  {x[0]:.0f} XOR {x[1]:.0f} = {expected:5s} | Predicted: {got:5s} (raw: {pred.data:+.4f})")


# ============================================================
# EXERCISE 3: Plot Training Loss
# ============================================================

def exercise_3_plot_loss():
    """
    Track and plot the training loss over epochs.
    You need: pip install matplotlib
    """
    print("Exercise 3: Plot Training Loss")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib first: pip install matplotlib")
        return

    random.seed(42)

    xs = [
        [2.0, 3.0, -1.0], [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0], [1.0, 1.0, -1.0],
        [-1.0, -1.0, 0.5], [-2.0, 1.0, 1.0],
        [-1.0, -2.0, -1.0], [-3.0, 0.5, 0.5],
    ]
    ys = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

    model = MLP(3, [4, 4, 1])
    losses = []

    for epoch in range(200):
        predictions = [model(x) for x in xs]
        loss = sum((pred - t) ** 2 for pred, t in zip(predictions, ys))

        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad

        losses.append(loss.data)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # log scale shows the improvement better
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    print("Saved plot to training_loss.png")
    plt.show()


# ============================================================
# EXERCISE 4: Compare Learning Rates
# ============================================================

def exercise_4_learning_rates():
    """
    Train the same model with different learning rates and compare.
    This builds intuition for one of the most important hyperparameters.
    """
    print("Exercise 4: Compare Learning Rates")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib first: pip install matplotlib")
        return

    xs = [
        [2.0, 3.0, -1.0], [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0], [1.0, 1.0, -1.0],
        [-1.0, -1.0, 0.5], [-2.0, 1.0, 1.0],
        [-1.0, -2.0, -1.0], [-3.0, 0.5, 0.5],
    ]
    ys = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
    all_losses = {}

    for lr in learning_rates:
        random.seed(42)  # same initialization for fair comparison
        model = MLP(3, [4, 4, 1])
        losses = []

        for epoch in range(200):
            predictions = [model(x) for x in xs]
            loss = sum((pred - t) ** 2 for pred, t in zip(predictions, ys))

            for p in model.parameters():
                p.grad = 0.0
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

            losses.append(min(loss.data, 100))  # cap for visualization

        all_losses[lr] = losses
        print(f"  lr={lr:.3f} | Final loss: {losses[-1]:.6f}")

    plt.figure(figsize=(12, 6))
    for lr, losses in all_losses.items():
        plt.plot(losses, label=f'lr={lr}', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Effect of Learning Rate on Training', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('learning_rates.png', dpi=150)
    print("\nSaved plot to learning_rates.png")
    plt.show()


# ============================================================
# EXERCISE 5: Moon Dataset (Real Classification)
# ============================================================

def exercise_5_moons():
    """
    Train on the sklearn moons dataset - a real 2D classification problem.
    You need: pip install scikit-learn matplotlib numpy
    """
    print("Exercise 5: Moon Dataset Classification")
    print("=" * 50)

    try:
        from sklearn.datasets import make_moons
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install dependencies: pip install scikit-learn matplotlib numpy")
        return

    random.seed(42)
    np.random.seed(42)

    # Generate dataset
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    y = y * 2 - 1  # convert 0,1 -> -1,+1

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=40, label='Class +1', edgecolors='black')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=40, label='Class -1', edgecolors='black')
    plt.legend()
    plt.title('Moon Dataset')
    plt.savefig('moon_dataset.png', dpi=150)
    print("Saved moon_dataset.png")

    # Create model - bigger for this harder problem
    model = MLP(2, [16, 16, 1])
    print(f"Parameters: {len(model.parameters())}")

    # Training
    for epoch in range(100):
        # Forward
        predictions = [model(x.tolist()) for x in X]

        # Data loss + regularization
        data_loss = sum((pred - yi) ** 2 for pred, yi in zip(predictions, y))
        reg_loss = 1e-4 * sum(p * p for p in model.parameters())
        total_loss = data_loss + reg_loss

        # Zero grad
        for p in model.parameters():
            p.grad = 0.0

        # Backward
        total_loss.backward()

        # Learning rate decay
        lr = 0.1 - 0.09 * (epoch / 100)

        # Update
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 10 == 0:
            correct = sum(1 for pred, yi in zip(predictions, y) if (pred.data > 0) == (yi > 0))
            acc = correct / len(y) * 100
            print(f"  Epoch {epoch:3d} | Loss: {total_loss.data:.4f} | Accuracy: {acc:.1f}%")

    # Plot decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model(p.tolist()).data for p in grid_points])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'blue'])
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=40, label='Class +1', edgecolors='black')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=40, label='Class -1', edgecolors='black')
    plt.legend()
    plt.title('Decision Boundary (trained micrograd MLP)')
    plt.savefig('moon_decision_boundary.png', dpi=150)
    print("Saved moon_decision_boundary.png")
    plt.show()


# ============================================================
# EXERCISE 6: Compare with PyTorch
# ============================================================

def exercise_6_pytorch_comparison():
    """
    Run the same computation in your micrograd and PyTorch.
    Verify that gradients match exactly.

    You need: pip install torch
    """
    print("Exercise 6: Compare with PyTorch")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        print("Install PyTorch first: pip install torch")
        return

    # --- micrograd ---
    a_mg = Value(2.0, label='a')
    b_mg = Value(-3.0, label='b')
    c_mg = Value(10.0, label='c')

    d_mg = a_mg * b_mg + c_mg
    e_mg = d_mg.tanh()
    f_mg = e_mg ** 2
    f_mg.backward()

    print("micrograd:")
    print(f"  result = {f_mg.data:.6f}")
    print(f"  a.grad = {a_mg.grad:.6f}")
    print(f"  b.grad = {b_mg.grad:.6f}")
    print(f"  c.grad = {c_mg.grad:.6f}")

    # --- PyTorch ---
    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(-3.0, requires_grad=True)
    c_pt = torch.tensor(10.0, requires_grad=True)

    d_pt = a_pt * b_pt + c_pt
    e_pt = d_pt.tanh()
    f_pt = e_pt ** 2
    f_pt.backward()

    print("\nPyTorch:")
    print(f"  result = {f_pt.item():.6f}")
    print(f"  a.grad = {a_pt.grad.item():.6f}")
    print(f"  b.grad = {b_pt.grad.item():.6f}")
    print(f"  c.grad = {c_pt.grad.item():.6f}")

    # --- Compare ---
    print("\nMatch check:")
    checks = [
        ("result", f_mg.data, f_pt.item()),
        ("a.grad", a_mg.grad, a_pt.grad.item()),
        ("b.grad", b_mg.grad, b_pt.grad.item()),
        ("c.grad", c_mg.grad, c_pt.grad.item()),
    ]
    all_pass = True
    for name, mg_val, pt_val in checks:
        match = abs(mg_val - pt_val) < 1e-6
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  {name:10s} | micrograd={mg_val:+.6f} | pytorch={pt_val:+.6f} | {status}")

    if all_pass:
        print("\nAll values match! Your autograd engine is correct.")


# ============================================================
# Uncomment ONE exercise at a time and run:
#   python exercises.py
# ============================================================

if __name__ == "__main__":
    exercise_1_verify_gradients()
    # exercise_2_xor()
    # exercise_3_plot_loss()
    # exercise_4_learning_rates()
    # exercise_5_moons()
    # exercise_6_pytorch_comparison()
