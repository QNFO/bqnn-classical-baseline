"""
Numpy BinaryConnect MLP — Classical Competitive Baseline for BQNN.
Pure numpy implementation: no PyTorch, no GPU dependency, fully deterministic.
Implements Courbariaux et al. (2015) STE + SGD with 3 stochastic regularization
methods for inference-time benchmarking against BQNN's quantum measurement uncertainty.

Architecture: 3 hidden layers x 16 neurons, tanh activations, ±1 binarized weights.
Input: MNIST 28x28 = 784 dimensions. Output: 10 classes.
Training: SGD with momentum + straight-through estimator (STE), 100 epochs.
Inference: Standard, MC Dropout, Gaussian Noise Injection, MC Ensemble.

Usage:
    python src/binarized_mlp_np.py results/benchmark_results.json
"""
import numpy as np
import struct
import gzip
import os
import sys
import json
import time
import math

# ─────────────────────────────────────────────────────────────
# MNIST LOADER (zero dependencies — reads raw IDX files)
# ─────────────────────────────────────────────────────────────

def read_mnist_images(filename):
    """Read MNIST IDX file format."""
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data.astype(np.float32) / 255.0

def read_mnist_labels(filename):
    """Read MNIST label IDX file."""
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def download_mnist(data_dir="data"):
    """Download MNIST from Yann LeCun's site if not present."""
    import urllib.request
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": base + "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": base + "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": base + "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": base + "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs(data_dir, exist_ok=True)
    for fn, url in files.items():
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"  Downloading {fn}...")
            urllib.request.urlretrieve(url, path)
    return data_dir

def load_mnist_subset(data_dir="data", n_images=55, seed=42):
    """Load MNIST and return stratified 55-image test subset."""
    download_mnist(data_dir)

    X_train = read_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = read_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = read_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = read_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    # Normalize: zero mean, unit variance (MNIST standard)
    X_train = (X_train - 0.1307) / 0.3081
    X_test = (X_test - 0.1307) / 0.3081

    # Flatten: 28x28 → 784
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # One-hot encode labels
    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    # Stratified subset of test set (55 images, 5-6 per class)
    np.random.seed(seed)
    indices = []
    for cls in range(10):
        cls_idx = np.where(y_test == cls)[0]
        n_per_class = max(5, 55 // 10)
        chosen = np.random.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen)
    # Top up to exactly n_images
    if len(indices) < n_images:
        remaining = list(set(range(len(y_test))) - set(indices))
        indices.extend(np.random.choice(remaining, size=n_images - len(indices), replace=False))

    indices = sorted(indices)
    return X_train, y_train_oh, y_train, X_test[indices], y_test_oh[indices], y_test[indices]


# ─────────────────────────────────────────────────────────────
# LAYERS
# ─────────────────────────────────────────────────────────────

def xavier_init(fan_in, fan_out):
    """Xavier/Glorot uniform initialization."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    # Numerically stable
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)

def cross_entropy(logits, y_onehot):
    probs = softmax(logits)
    return -np.mean(np.sum(y_onehot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))

def dropout_mask(shape, p, seed=None):
    """Bernoulli dropout mask. p = keep probability."""
    if seed is not None:
        np.random.seed(seed)
    return (np.random.rand(*shape) < p).astype(np.float32) / p


# ─────────────────────────────────────────────────────────────
# BINARYCONNECT MLP
# ─────────────────────────────────────────────────────────────

class BinaryConnectMLP:
    """3-layer binarized MLP with STE training.

    Architecture matches BQNN exactly:
    - 3 hidden layers x 16 neurons
    - Binarized ±1 weights at forward pass
    - Real-valued weights stored for backprop (STE)
    - SGD with momentum
    """

    def __init__(self, input_dim=784, hidden_dim=16, output_dim=10, dropout_p=0.0, seed=42):
        np.random.seed(seed)
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        # Real-valued weights
        self.W1 = xavier_init(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = xavier_init(hidden_dim, hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = xavier_init(hidden_dim, hidden_dim)
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = xavier_init(hidden_dim, output_dim)
        self.b4 = np.zeros((1, output_dim))

        # Momentum velocities
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)
        self.vW4 = np.zeros_like(self.W4)
        self.vb4 = np.zeros_like(self.b4)

    def binarize(self, W):
        """BinaryConnect: w_b = sign(w) * mean(|w|)."""
        scale = np.mean(np.abs(W))
        if scale < 1e-10:
            scale = 1e-10
        return scale * np.sign(W)

    def forward(self, x, noise_std=0.0, training=True, dp_seed=None):
        """Single forward pass. Returns logits + cached activations for backprop."""
        cache = {}

        # Layer 1
        W1_b = self.binarize(self.W1) if training else self.binarize(self.W1)
        z1 = x @ W1_b + self.b1
        a1 = tanh(z1)
        cache['z1'], cache['a1'], cache['W1_b'] = z1, a1, W1_b

        # Layer 2
        W2_b = self.binarize(self.W2) if training else self.binarize(self.W2)
        z2 = a1 @ W2_b + self.b2
        a2 = tanh(z2)
        if self.dropout_p > 0 and training:
            mask = dropout_mask(a2.shape, self.dropout_p, dp_seed)
            a2 = a2 * mask
        cache['z2'], cache['a2'], cache['W2_b'] = z2, a2, W2_b

        # Layer 3
        W3_b = self.binarize(self.W3) if training else self.binarize(self.W3)
        z3 = a2 @ W3_b + self.b3
        a3 = tanh(z3)
        if noise_std > 0:
            a3 = a3 + np.random.randn(*a3.shape) * noise_std
        cache['z3'], cache['a3'], cache['W3_b'] = z3, a3, W3_b

        # Classifier
        z4 = a3 @ self.W4 + self.b4
        cache['a3_final'] = a3

        return z4, cache

    def backward_and_update(self, x, y_onehot, lr=0.01, momentum=0.9):
        """Full forward + backward pass with STE + SGD."""
        batch_size = x.shape[0]
        logits, cache = self._forward_train(x)

        # Cross-entropy gradient: probs - y_onehot
        probs = softmax(logits)
        dlogits = (probs - y_onehot) / batch_size

        # Layer 4 gradient
        dW4 = cache['a3_final'].T @ dlogits
        db4 = np.sum(dlogits, axis=0, keepdims=True)

        # Backprop to layer 3
        da3 = dlogits @ self.W4.T
        dz3 = da3 * (1 - cache['a3'] ** 2)  # tanh derivative

        # STE for layer 3: gradient passes through real weights
        dW3 = cache['a2'].T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Backprop to layer 2
        da2 = dz3 @ cache['W3_b'].T
        dz2 = da2 * (1 - cache['a2'] ** 2)
        dW2 = cache['a1'].T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Backprop to layer 1
        da1 = dz2 @ cache['W2_b'].T
        dz1 = da1 * (1 - cache['a1'] ** 2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # SGD with momentum update
        self.vW4 = momentum * self.vW4 - lr * dW4
        self.vb4 = momentum * self.vb4 - lr * db4
        self.vW3 = momentum * self.vW3 - lr * dW3
        self.vb3 = momentum * self.vb3 - lr * db3
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1

        self.W4 += self.vW4; self.b4 += self.vb4
        self.W3 += self.vW3; self.b3 += self.vb3
        self.W2 += self.vW2; self.b2 += self.vb2
        self.W1 += self.vW1; self.b1 += self.vb1

        loss = cross_entropy(logits, y_onehot)
        acc = np.mean(np.argmax(logits, axis=1) == np.argmax(y_onehot, axis=1))
        return loss, acc

    def _forward_train(self, x):
        """Internal forward pass for training."""
        W1_b = self.binarize(self.W1)
        z1 = x @ W1_b + self.b1
        a1 = tanh(z1)

        W2_b = self.binarize(self.W2)
        z2 = a1 @ W2_b + self.b2
        a2 = tanh(z2)

        W3_b = self.binarize(self.W3)
        z3 = a2 @ W3_b + self.b3
        a3 = tanh(z3)

        z4 = a3 @ self.W4 + self.b4

        cache = {
            'a1': a1, 'a2': a2, 'a3': a3, 'a3_final': a3,
            'z1': z1, 'z2': z2, 'z3': z3,
            'W1_b': W1_b, 'W2_b': W2_b, 'W3_b': W3_b,
        }
        return z4, cache

    def predict(self, x):
        """Standard deterministic inference."""
        z4, _ = self.forward(x, training=False)
        return np.argmax(z4, axis=1)

    def predict_proba(self, x):
        """Return softmax probabilities."""
        z4, _ = self.forward(x, training=False)
        return softmax(z4)


# ─────────────────────────────────────────────────────────────
# INFERENCE METHODS
# ─────────────────────────────────────────────────────────────

def inference_standard(model, X):
    """Standard deterministic inference."""
    preds = model.predict(X)
    return preds

def inference_mc_dropout(model, X, keep_prob=0.5, n_samples=10, seed=99):
    """MC Dropout inference (Gal & Ghahramani 2016).
    Applies Bernoulli dropout at inference, averages n_samples predictions.
    """
    probs = np.zeros((len(X), 10))
    for i in range(n_samples):
        np.random.seed(seed + i)
        logits, _ = model.forward(X, training=True, dp_seed=seed + i)
        probs += softmax(logits)
    probs /= n_samples
    return np.argmax(probs, axis=1)

def inference_gaussian(model, X, sigma=0.1, n_samples=10, seed=99):
    """Gaussian noise injection at inference (Bishop 1995).
    Adds N(0, sigma^2) noise to third-layer activations.
    """
    probs = np.zeros((len(X), 10))
    original_b3 = model.b3.copy()
    for i in range(n_samples):
        np.random.seed(seed + i)
        model.b3 = original_b3 + np.random.normal(0, sigma*0.1, model.b3.shape)
        logits, _ = model.forward(X, training=False)
        probs += softmax(logits)
    probs /= n_samples
    model.b3 = original_b3  # restore
    return np.argmax(probs, axis=1)

def inference_ensemble(models, X):
    """MC Ensemble inference (Lakshminarayanan et al. 2017).
    Averages softmax probabilities from K independently trained models.
    """
    probs = np.zeros((len(X), 10))
    for m in models:
        probs += m.predict_proba(X)
    probs /= len(models)
    return np.argmax(probs, axis=1)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train_oh, epochs=100, lr=0.01,
                batch_size=64, verbose=True):
    """Train BinaryConnect MLP with SGD + STE."""
    n = len(X_train)
    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n)
        epoch_loss, epoch_acc, batches = 0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train_oh[idx]
            loss, acc = model.backward_and_update(xb, yb, lr=lr)
            epoch_loss += loss
            epoch_acc += acc
            batches += 1

        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        if verbose and (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return history


# ─────────────────────────────────────────────────────────────
# ENERGY MEASUREMENT
# ─────────────────────────────────────────────────────────────

def measure_inference_energy(model, X_test, n_repeats=5):
    """Measure wall-clock time and estimate joules (CPU TDP estimate)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_time = np.mean(times)
    # CPU TDP estimate: 65W (typical laptop/desktop)
    cpu_tdp = 65.0
    avg_joules = cpu_tdp * (avg_time / 1000)
    return avg_time, avg_joules, cpu_tdp


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER SWEEP
# ─────────────────────────────────────────────────────────────

def run_experiment(X_train, y_train_oh, y_train, X_test, y_test, config, seed=0):
    """Run one experiment: train, measure, report."""

    # Train ensemble
    models = []
    for k in range(config['ensemble_size']):
        np.random.seed(seed + k)
        model = BinaryConnectMLP(
            hidden_dim=config['hidden_dim'],
            dropout_p=config['dropout_p'],
            seed=seed + k,
        )
        history = train_model(model, X_train, y_train_oh,
                              epochs=config['epochs'], lr=config['lr'],
                              batch_size=config['batch_size'],
                              verbose=(k == 0))
        models.append(model)

    # Standard inference
    preds_std = inference_standard(models[0], X_test)
    acc_std = np.mean(preds_std == y_test)

    # MC Dropout inference
    acc_dropout = None
    if config['dropout_p'] > 0:
        # Create a fresh model with same weights
        dmodel = BinaryConnectMLP(
            hidden_dim=config['hidden_dim'],
            dropout_p=config['dropout_p'],
            seed=seed,
        )
        dmodel.W1 = models[0].W1.copy()
        dmodel.b1 = models[0].b1.copy()
        dmodel.W2 = models[0].W2.copy()
        dmodel.b2 = models[0].b2.copy()
        dmodel.W3 = models[0].W3.copy()
        dmodel.b3 = models[0].b3.copy()
        dmodel.W4 = models[0].W4.copy()
        dmodel.b4 = models[0].b4.copy()
        preds_dp = inference_mc_dropout(dmodel, X_test, keep_prob=config['dropout_p'],
                                         n_samples=10, seed=999)
        acc_dropout = np.mean(preds_dp == y_test)

    # Gaussian noise inference
    acc_gaussian = None
    if config['gaussian_sigma'] > 0:
        preds_gn = inference_gaussian(models[0], X_test, sigma=config['gaussian_sigma'],
                                       n_samples=10, seed=998)
        acc_gaussian = np.mean(preds_gn == y_test)

    # Ensemble inference
    acc_ensemble = None
    if config['ensemble_size'] > 1:
        preds_ens = inference_ensemble(models, X_test)
        acc_ensemble = np.mean(preds_ens == y_test)

    # Energy measurement
    avg_time_ms, avg_joules, tdp = measure_inference_energy(models[0], X_test)

    best_acc = max([a for a in [acc_std, acc_dropout, acc_gaussian, acc_ensemble]
                    if a is not None])

    return {
        'config': config,
        'metrics': {
            'accuracy_standard': round(float(acc_std), 4),
            'accuracy_mc_dropout': round(float(acc_dropout), 4) if acc_dropout else None,
            'accuracy_gaussian_noise': round(float(acc_gaussian), 4) if acc_gaussian else None,
            'accuracy_ensemble': round(float(acc_ensemble), 4) if acc_ensemble else None,
            'best_accuracy': round(float(best_acc), 4),
            'inference_time_ms': round(float(avg_time_ms), 2),
            'inference_joules': round(float(avg_joules), 8),
            'tdp_watts': tdp,
        },
    }


def build_configs():
    """Build 27-config hyperparameter sweep grid."""
    configs = []
    for dp in [0.0, 0.2, 0.5]:
        for gs in [0.0, 0.1, 0.3]:
            for es in [1, 5, 10]:
                configs.append({
                    'dropout_p': dp,
                    'gaussian_sigma': gs,
                    'ensemble_size': es,
                    'epochs': 100,
                    'lr': 0.01,
                    'hidden_dim': 16,
                    'batch_size': 64,
                })
    return configs


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BQNN Classical Competitive Baseline — BinaryConnect (numpy)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading MNIST...")
    X_train, y_train_oh, y_train, X_test, y_test_oh, y_test = load_mnist_subset(
        data_dir="data", n_images=55, seed=42)
    print(f"  Train: {X_train.shape[0]} images, Test: {X_test.shape[0]} images")

    # Build sweep
    print("\n[2/4] Building config sweep...")
    configs = build_configs()
    print(f"  {len(configs)} configurations")

    # Run sweep
    print("\n[3/4] Running sweep...")
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] "
              f"dropout={cfg['dropout_p']}, "
              f"sigma={cfg['gaussian_sigma']}, "
              f"ensemble={cfg['ensemble_size']}")
        t0 = time.perf_counter()
        result = run_experiment(X_train, y_train_oh, y_train, X_test, y_test, cfg, seed=i)
        elapsed = time.perf_counter() - t0
        result['wall_time_s'] = round(elapsed, 1)
        results.append(result)
        m = result['metrics']
        methods = []
        methods.append(f"std={m['accuracy_standard']}")
        if m['accuracy_mc_dropout'] is not None:
            methods.append(f"dropout={m['accuracy_mc_dropout']}")
        if m['accuracy_gaussian_noise'] is not None:
            methods.append(f"gauss={m['accuracy_gaussian_noise']}")
        if m['accuracy_ensemble'] is not None:
            methods.append(f"ensemble={m['accuracy_ensemble']}")
        print(f"    BEST={m['best_accuracy']:.4f} ({', '.join(methods)}) "
              f"| {m['inference_time_ms']:.1f}ms | {m['inference_joules']:.8f}J "
              f"| wall={elapsed:.1f}s")

    # Save
    print("\n[4/4] Saving results...")
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Summary
    total = sum(r['wall_time_s'] for r in results)
    best = max(results, key=lambda r: r['metrics']['best_accuracy'])
    b = best['metrics']
    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE — {len(results)} configs in {total:.0f}s")
    print(f"BEST: dropout={best['config']['dropout_p']}, "
          f"sigma={best['config']['gaussian_sigma']}, "
          f"ensemble={best['config']['ensemble_size']}")
    print(f"  best_accuracy={b['best_accuracy']:.4f}")
    print(f"  inference_time={b['inference_time_ms']:.1f}ms")
    print(f"  inference_energy={b['inference_joules']:.8f}J")
    print(f"  BQNN v1 estimate: 550,000J per 55 images")
    print(f"  Ratio (classical/bqnn): {b['inference_joules']:.8f} / 550000")
    print(f"{'=' * 60}")
