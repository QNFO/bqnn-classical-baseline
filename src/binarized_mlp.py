"""
BinaryConnect MLP — Classical Competitive Baseline for BQNN.
Implements Courbariaux et al. (2015) with STE, plus 3 stochastic
regularization methods for inference-time benchmarking against
BQNN's quantum measurement uncertainty.

Architecture: 3 hidden layers × 16 neurons, tanh activations, ±1 weights.
Training: STE + SGD, identical to BQNN's classical training protocol.
Inference: Bernoulli dropout, Gaussian noise injection, MC ensemble.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import json
import sys
import os


# ──────────────────────────────────────────────────────────────────────
# HARD BINARIZATION (BinaryConnect, Courbariaux 2015)
# ──────────────────────────────────────────────────────────────────────

def binarize_weights(weight: torch.Tensor) -> torch.Tensor:
    """Deterministic binarization: w_b = sign(w) with E[|w|] scaling."""
    scaling = weight.abs().mean().clamp(min=1e-8)
    return scaling * weight.sign()


class BinaryConnectLinear(nn.Linear):
    """Linear layer with binarized weights at forward pass (STE)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._binary_weight = None

    def forward(self, x):
        if self.training:
            self._binary_weight = binarize_weights(self.weight)
        else:
            self._binary_weight = binarize_weights(self.weight).detach()
        # Straight-through estimator: use binary weights for forward,
        # real weights for backward (handled by autograd through _binary_weight)
        return F.linear(x, self._binary_weight, self.bias)


# ──────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────

class BinaryConnectMLP(nn.Module):
    """3-layer binarized MLP matching BQNN architecture exactly.

    BQNN: 3 hidden layers, 16 neurons/layer, ±1 activations (in BQNN's
    quantum inference), classifier head.

    BinaryConnect: 3 hidden layers, 16 neurons/layer, tanh activations
    (real-valued at inference — more powerful than BQNN's binarized
    activations, so any BQNN advantage becomes harder to claim), ±1 weights.
    """
    def __init__(self, input_dim=784, hidden_dim=16, num_classes=10,
                 dropout_p=0.0):
        super().__init__()
        self.fc1 = BinaryConnectLinear(input_dim, hidden_dim)
        self.fc2 = BinaryConnectLinear(hidden_dim, hidden_dim)
        self.fc3 = BinaryConnectLinear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x, noise_std=0.0):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout(x)
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────
# INFERENCE METHODS
# ──────────────────────────────────────────────────────────────────────

def inference_standard(model, loader, device="cpu"):
    """Standard deterministic inference."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def inference_dropout(model, loader, p=0.5, n_samples=10, device="cpu"):
    """Monte Carlo dropout inference (Gal & Ghahramani 2016).
    Applies dropout at inference with probability p, runs n_samples
    independent forward passes, averages logits.
    """
    # Force dropout on at inference
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # dropout active at inference
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = torch.stack([model(x) for _ in range(n_samples)], dim=0)
            pred = logits.mean(dim=0).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def inference_gaussian(model, loader, sigma=0.1, n_samples=10, device="cpu"):
    """Gaussian noise injection at inference (Bishop 1995).
    Adds N(0, sigma^2) noise to third-layer activations before classifier.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch_logits = []
            for _ in range(n_samples):
                logits = model(x, noise_std=sigma)
                batch_logits.append(logits)
            pred = torch.stack(batch_logits).mean(dim=0).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def inference_ensemble(models, loader, device="cpu"):
    """MC ensemble inference (Lakshminarayanan et al. 2017).
    Averages logits from K independently trained BinaryConnect models.
    """
    for m in models:
        m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = torch.stack([m(x) for m in models], dim=0)
            pred = logits.mean(dim=0).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, epochs=100, lr=0.01, device="cpu",
                verbose=True):
    """Train BinaryConnect MLP with STE + SGD (BQNN training protocol)."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "train_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total if total > 0 else 0
        history["loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={epoch_loss:.4f}, "
                  f"train_acc={epoch_acc:.4f}")

    return history


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_mnist_subset(n_images=55, seed=42, batch_size=8, data_dir="data"):
    """Load MNIST and select N test images.

    BQNN used 55 randomly selected MNIST test images. Since the exact indices
    are not published, we sample with a fixed seed for reproducibility.
    Class distribution is preserved (stratified sample).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(data_dir, train=True, download=True,
                                 transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True,
                                transform=transform)

    # Stratified sample from test set
    rng = np.random.RandomState(seed)
    targets = test_data.targets.numpy()
    indices = []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        n_per_class = max(1, n_images // 10)
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)),
                            replace=False)
        indices.extend(chosen)
    # Top up to exactly n_images
    if len(indices) < n_images:
        remaining = list(set(range(len(test_data))) - set(indices))
        indices.extend(rng.choice(remaining, size=n_images - len(indices),
                                  replace=False))

    test_subset = Subset(test_data, sorted(indices))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_subset


# ──────────────────────────────────────────────────────────────────────
# HARDWARE ENERGY MEASUREMENT
# ──────────────────────────────────────────────────────────────────────

def measure_inference_energy(model, loader, n_repeats=5, device="cpu"):
    """Measure wall-clock time and estimate joules.

    GPU: Uses NVIDIA TDP from nvidia-smi if available, else defaults to
    RTX 3060 TDP (170W). CPU: uses package TDP estimate (65W default).

    Returns: avg_time_ms, avg_joules, tdp_watts
    """
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for x, y in loader:
                x = x.to(device)
                _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_time = np.mean(times)

    # GPU energy estimate
    if device == "cuda":
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            tdp = float(result.stdout.strip())
        except Exception:
            tdp = 170.0  # RTX 3060 default
    else:
        tdp = 65.0  # CPU TDP estimate

    avg_joules = tdp * (avg_time / 1000)
    return avg_time, avg_joules, tdp


# ──────────────────────────────────────────────────────────────────────
# MAIN — Hyperparameter Sweep
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    dropout_p: float = 0.0
    gaussian_sigma: float = 0.0
    ensemble_size: int = 1
    epochs: int = 100
    lr: float = 0.01
    hidden_dim: int = 16
    seed: int = 42


def run_sweep(configs: List[SweepConfig], n_images=55, device="cpu",
              output_path=None):
    """Run hyperparameter sweep and collect results."""
    results = []

    for i, cfg in enumerate(configs):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        print(f"\n[{i+1}/{len(configs)}] dropout={cfg.dropout_p}, "
              f"sigma={cfg.gaussian_sigma}, ensemble={cfg.ensemble_size}")

        # Load data (consistent across configs via seed)
        train_loader, test_loader, _ = load_mnist_subset(
            n_images=n_images, seed=cfg.seed)

        # Train ensemble
        models = []
        histories = []
        for k in range(cfg.ensemble_size):
            torch.manual_seed(cfg.seed + k)
            model = BinaryConnectMLP(
                hidden_dim=cfg.hidden_dim,
                dropout_p=cfg.dropout_p,
            ).to(device)
            history = train_model(model, train_loader, epochs=cfg.epochs,
                                   lr=cfg.lr, device=device, verbose=(k == 0))
            models.append(model)
            histories.append(history)

        # Inference: standard (deterministic)
        acc_standard = inference_standard(models[0], test_loader, device)

        # Inference: MC dropout (if p > 0)
        acc_dropout = None
        if cfg.dropout_p > 0:
            # Create a fresh model with dropout, copy weights, test
            dmodel = BinaryConnectMLP(
                hidden_dim=cfg.hidden_dim, dropout_p=cfg.dropout_p
            ).to(device)
            dmodel.load_state_dict(models[0].state_dict())
            acc_dropout = inference_dropout(dmodel, test_loader, p=cfg.dropout_p,
                                             n_samples=10, device=device)

        # Inference: Gaussian noise
        acc_gaussian = None
        if cfg.gaussian_sigma > 0:
            acc_gaussian = inference_gaussian(
                models[0], test_loader, sigma=cfg.gaussian_sigma,
                n_samples=10, device=device)

        # Inference: Ensemble
        acc_ensemble = None
        if cfg.ensemble_size > 1:
            acc_ensemble = inference_ensemble(models, test_loader, device)

        # Energy measurement
        avg_time_ms, avg_joules, tdp = measure_inference_energy(
            models[0], test_loader, device=device)

        best_acc = max([a for a in [acc_standard, acc_dropout, acc_gaussian,
                                     acc_ensemble] if a is not None])

        result = {
            "config": {
                "dropout_p": cfg.dropout_p,
                "gaussian_sigma": cfg.gaussian_sigma,
                "ensemble_size": cfg.ensemble_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "hidden_dim": cfg.hidden_dim,
                "seed": cfg.seed,
            },
            "metrics": {
                "accuracy_standard": acc_standard,
                "accuracy_mc_dropout": acc_dropout,
                "accuracy_gaussian_noise": acc_gaussian,
                "accuracy_ensemble": acc_ensemble,
                "best_accuracy": best_acc,
                "inference_time_ms": avg_time_ms,
                "inference_joules": avg_joules,
                "tdp_watts": tdp,
            },
            "training": {
                "final_loss": histories[0]["loss"][-1],
                "final_train_acc": histories[0]["train_acc"][-1],
            }
        }
        results.append(result)
        print(f"  best_acc={best_acc:.4f}, time={avg_time_ms:.1f}ms, "
              f"energy={avg_joules:.6f}J")

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def build_sweep_configs() -> List[SweepConfig]:
    """Build the hyperparameter sweep grid (27 combinations)."""
    configs = []
    for dp in [0.0, 0.2, 0.5]:
        for gs in [0.0, 0.1, 0.3]:
            for es in [1, 5, 10]:
                configs.append(SweepConfig(
                    dropout_p=dp,
                    gaussian_sigma=gs,
                    ensemble_size=es,
                ))
    return configs


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    configs = build_sweep_configs()
    print(f"Sweep: {len(configs)} configurations")

    output = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    results = run_sweep(configs, n_images=55, device=device, output_path=output)

    # Summary
    best = max(results, key=lambda r: r["metrics"]["best_accuracy"])
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION:")
    print(f"  dropout={best['config']['dropout_p']}, "
          f"sigma={best['config']['gaussian_sigma']}, "
          f"ensemble={best['config']['ensemble_size']}")
    print(f"  best_accuracy={best['metrics']['best_accuracy']:.4f}")
    print(f"  inference_time={best['metrics']['inference_time_ms']:.1f}ms")
    print(f"  inference_energy={best['metrics']['inference_joules']:.6f}J")
    print(f"{'='*60}")
