"""
Day 4: Hugging Face Transformers - Loading Pre-trained Models & Text Generation

Demonstrates:
1. Setting up device (MPS/CUDA/CPU)
2. Loading GPT-2 and DistilGPT-2 from Hugging Face Hub
3. Exploring the tokenizer (encode, decode, vocab)
4. Text generation with different sampling parameters
5. Temperature experiments
6. Top-k and top-p sampling
7. Model comparison

Run: python3 huggingface_generate.py
Requires: transformers, accelerate, torch, matplotlib
First run downloads ~600MB of model weights.
"""

import torch
import time
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. Device Setup
# ============================================================

def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ============================================================
# 2. Load Models
# ============================================================

def load_model(model_name, device):
    """Load a causal language model and tokenizer from Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    elapsed = time.time() - start
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Max position embeddings: {model.config.n_positions}")

    return model, tokenizer


# ============================================================
# 3. Tokenizer Exploration
# ============================================================

def explore_tokenizer(tokenizer):
    """Show how the tokenizer encodes and decodes text."""
    print("\n" + "=" * 60)
    print("TOKENIZER EXPLORATION")
    print("=" * 60)

    # Basic encode/decode
    examples = [
        "Hello, World!",
        "The transformer architecture",
        "GPT-2 is a language model",
        "Tokenization matters!",
        "unbelievably extraordinary",
    ]

    for text in examples:
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"\n  Text:   '{text}'")
        print(f"  IDs:    {ids}")
        print(f"  Tokens: {tokens}")
        print(f"  Count:  {len(ids)}")

    # Special tokens
    print(f"\n  --- Special Tokens ---")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    if tokenizer.pad_token:
        print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    else:
        print(f"  PAD token: None (GPT-2 doesn't use padding)")

    # Vocab samples
    print(f"\n  --- Vocabulary Samples ---")
    print(f"  Total vocab size: {tokenizer.vocab_size:,}")

    # Show some interesting tokens
    interesting = ["hello", " hello", "Hello", " Hello", "the", " the",
                   "Ġthe", " AI", "transformer", " transform"]
    for t in interesting:
        token_id = tokenizer.convert_tokens_to_ids(t)
        if token_id != tokenizer.unk_token_id:
            print(f"  '{t}' -> {token_id}")


# ============================================================
# 4. Text Generation
# ============================================================

def generate_text(model, tokenizer, prompt, device, max_new_tokens=50,
                  temperature=1.0, top_k=0, top_p=1.0, do_sample=True):
    """Generate text from a prompt with configurable sampling parameters."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated


# ============================================================
# 5. Temperature Experiments
# ============================================================

def experiment_temperature(model, tokenizer, device):
    """Show how temperature affects generation quality."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Temperature")
    print("=" * 60)

    prompt = "The future of artificial intelligence is"
    temperatures = [0.3, 0.7, 1.0, 1.5]

    results = {}
    for temp in temperatures:
        print(f"\n  Temperature = {temp}")
        print(f"  Prompt: '{prompt}'")

        text = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=60, temperature=temp,
            do_sample=(temp != 0.3),  # near-greedy for very low temp
        )
        # Get only the generated part
        generated = text[len(prompt):]
        results[temp] = generated
        print(f"  Output: {text[:150]}...")

    print("\n  --- Summary ---")
    print("  Low temp (0.3): focused, repetitive, 'safe' completions")
    print("  Med temp (0.7): good balance of quality and creativity")
    print("  Temp 1.0: original distribution, more diverse")
    print("  High temp (1.5): creative but potentially incoherent")

    return results


# ============================================================
# 6. Top-k and Top-p Experiments
# ============================================================

def experiment_sampling(model, tokenizer, device):
    """Compare different sampling strategies."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Sampling Strategies")
    print("=" * 60)

    prompt = "In a world where robots have gained consciousness,"

    configs = [
        ("Greedy", {"do_sample": False}),
        ("Top-k=10", {"do_sample": True, "top_k": 10, "temperature": 0.8}),
        ("Top-k=50", {"do_sample": True, "top_k": 50, "temperature": 0.8}),
        ("Top-p=0.9", {"do_sample": True, "top_p": 0.9, "temperature": 0.8}),
        ("Top-p=0.95", {"do_sample": True, "top_p": 0.95, "temperature": 0.8}),
        ("Top-k=50 + Top-p=0.9", {"do_sample": True, "top_k": 50,
                                    "top_p": 0.9, "temperature": 0.8}),
    ]

    results = {}
    for name, kwargs in configs:
        print(f"\n  Strategy: {name}")
        text = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=60, **kwargs,
        )
        results[name] = text
        print(f"  Output: {text[:150]}...")

    return results


# ============================================================
# 7. Model Comparison
# ============================================================

def compare_models(models_dict, tokenizer, device):
    """Compare generation quality across model variants."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    prompts = [
        "The meaning of life is",
        "To write good code, you should",
        "Machine learning models learn by",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        print(f"  {'─' * 50}")

        for name, model in models_dict.items():
            text = generate_text(
                model, tokenizer, prompt, device,
                max_new_tokens=40, temperature=0.7,
                top_p=0.9, do_sample=True,
            )
            generated = text[len(prompt):]
            print(f"  {name:>15}: {generated[:100].strip()}")


# ============================================================
# 8. Perplexity Measurement
# ============================================================

def measure_perplexity(model, tokenizer, device, text):
    """Compute perplexity of a model on given text.

    Perplexity = exp(average cross-entropy loss)
    Lower is better - means the model is less "surprised" by the text.
    """
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def experiment_perplexity(models_dict, tokenizer, device):
    """Compare perplexity across models."""
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON")
    print("=" * 60)

    test_texts = [
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Transformers use self-attention to process sequences in parallel.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    ]

    print(f"\n  {'Model':<18} {'Text':<50} {'Perplexity':>10}")
    print(f"  {'─'*18} {'─'*50} {'─'*10}")

    perplexities = {name: [] for name in models_dict}

    for text in test_texts:
        display = text[:48] + ".." if len(text) > 50 else text
        for name, model in models_dict.items():
            ppl = measure_perplexity(model, tokenizer, device, text)
            perplexities[name].append(ppl)
            print(f"  {name:<18} {display:<50} {ppl:>10.1f}")
        print()

    # Averages
    print(f"\n  --- Average Perplexity ---")
    for name, ppls in perplexities.items():
        avg = sum(ppls) / len(ppls)
        print(f"  {name}: {avg:.1f}")

    print("\n  Lower perplexity = model predicts the text better")
    print("  Larger models generally have lower perplexity")

    return perplexities


# ============================================================
# 9. Plot Results
# ============================================================

def plot_perplexity(perplexities):
    """Bar chart comparing average perplexity across models."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    names = list(perplexities.keys())
    avgs = [sum(ppls) / len(ppls) for ppls in perplexities.values()]
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

    bars = ax.bar(names, avgs, color=colors[:len(names)],
                  edgecolor="#1F2937", linewidth=1.0, alpha=0.85)

    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average Perplexity", fontsize=12)
    ax.set_title("Model Comparison: Average Perplexity (lower is better)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    filepath = os.path.join(SAVE_DIR, "model_perplexity_comparison.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Saved: {filepath}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 4: Hugging Face Transformers - Text Generation")
    print("=" * 60)

    device = get_device()

    # Load models (GPT-2 Small + DistilGPT-2)
    gpt2_model, gpt2_tokenizer = load_model("gpt2", device)
    distil_model, distil_tokenizer = load_model("distilgpt2", device)

    # Both share the same tokenizer, so we can use either
    tokenizer = gpt2_tokenizer

    # Explore the tokenizer
    explore_tokenizer(tokenizer)

    # Run experiments
    experiment_temperature(gpt2_model, tokenizer, device)
    experiment_sampling(gpt2_model, tokenizer, device)

    # Compare models
    models = {
        "GPT-2 (124M)": gpt2_model,
        "DistilGPT-2 (82M)": distil_model,
    }
    compare_models(models, tokenizer, device)

    # Perplexity
    perplexities = experiment_perplexity(models, tokenizer, device)
    plot_perplexity(perplexities)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)
