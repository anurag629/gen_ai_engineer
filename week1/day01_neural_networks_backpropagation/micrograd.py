"""
micrograd: A tiny autograd engine + neural network library.
Inspired by Andrej Karpathy's micrograd.

Run: python micrograd.py
"""

import math
import random


# ============================================================
# PART 1: The Autograd Engine
# ============================================================

class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1


# ============================================================
# PART 2: The Neural Network Library
# ============================================================

class Neuron:

    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:

    def __init__(self, n_inputs, n_outputs_per_layer):
        sizes = [n_inputs] + n_outputs_per_layer
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(n_outputs_per_layer))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================
# PART 3: Training
# ============================================================

if __name__ == "__main__":

    random.seed(42)

    # --- Dataset ---
    # Simple pattern: positive first element = class +1, negative = class -1
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 0.5],
        [-2.0, 1.0, 1.0],
        [-1.0, -2.0, -1.0],
        [-3.0, 0.5, 0.5],
    ]
    ys = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

    # --- Create the model ---
    model = MLP(3, [4, 4, 1])
    n_params = len(model.parameters())
    print(f"Model: MLP(3, [4, 4, 1])")
    print(f"Number of parameters: {n_params}")
    print(f"{'='*50}")

    # --- Training loop ---
    learning_rate = 0.05
    n_epochs = 100

    for epoch in range(n_epochs):

        # Forward pass
        predictions = [model(x) for x in xs]

        # Compute loss (MSE)
        loss = sum(
            (pred - target) ** 2
            for pred, target in zip(predictions, ys)
        )

        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Update weights (gradient descent)
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

    # --- Results ---
    print(f"\n{'='*50}")
    print("Final Predictions:")
    print(f"{'='*50}")
    for x, y in zip(xs, ys):
        pred = model(x)
        status = "correct" if (pred.data > 0) == (y > 0) else "WRONG"
        print(f"Input: {str(x):25s} | Target: {y:+.1f} | Pred: {pred.data:+.4f} | {status}")

    # --- Accuracy ---
    predictions = [model(x) for x in xs]
    correct = sum(1 for pred, y in zip(predictions, ys) if (pred.data > 0) == (y > 0))
    print(f"\nAccuracy: {correct}/{len(ys)} ({correct/len(ys)*100:.0f}%)")
