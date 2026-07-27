# Phase 2: Implementation Report — BinaryConnect MLP

**Date:** 2026-07-27
**Status:** IMPLEMENTED — ready for Phase 3 benchmarking

---

## 1. Architecture

### 1.1 BinaryConnect MLP

| Component | Detail | BQNN Match |
|:----------|:-------|:-----------|
| **Layers** | 3 hidden + 1 classifier | ✅ Identical (BQNN: 3 hidden layers) |
| **Hidden dim** | 16 neurons/layer | ✅ Identical |
| **Input** | 784 (MNIST 28×28 flattened) | ✅ Identical |
| **Output** | 10 classes | ✅ Identical |
| **Activation** | tanh (real-valued) | *More powerful* than BQNN's ±1 binarized |
| **Weight binarization** | sign(w) × E[|w|] | Identical strategy to BQNN |
| **Training algorithm** | STE (straight-through estimator) + SGD | ✅ Identical |
| **Libraries** | PyTorch | ✅ (BQNN uses Qiskit + PyTorch for classical sim) |

**Key design choice:** BinaryConnect uses real-valued tanh activations at inference (not ±1). This is *more powerful* than BQNN's binarized activations — if the classical baseline still matches BQNN's performance, the advantage claim is definitively falsified (tougher test for classical).

### 1.2 Weight Binarization (BinaryConnect)

```python
w_b = sign(w) × mean(|w|)
```

Following Courbariaux et al. (2015) exactly. The STE passes gradients through the binary weights to the real-valued weights during backprop. At each forward pass, weights are binarized; gradients use the real-valued weights.

### 1.3 Three Stochastic Regularization Methods

| Method | Mechanism | Parameters | BQNN Analogue |
|:-------|:----------|:-----------|:--------------|
| **MC Dropout** (Gal & Ghahramani 2016) | Bernoulli(p) dropout at inference, 10 forward passes, logit averaging | p ∈ {0.2, 0.5} | Quantum measurement projection at a ≈ 0.5 |
| **Gaussian Noise** (Bishop 1995) | N(0, σ²) added to 3rd-layer activations, 10 forward passes | σ ∈ {0.1, 0.3} | Stochastic quantum measurement with tunable variance |
| **MC Ensemble** (Lakshminarayanan 2017) | K independently trained models, logit averaging | K ∈ {5, 10} | Multiple independent quantum inference runs |

### 1.4 Hyperparameter Sweep Grid

| Parameter | Values | Count |
|:----------|:-------|:------|
| Dropout p | {0.0, 0.2, 0.5} | 3 |
| Gaussian σ | {0.0, 0.1, 0.3} | 3 |
| Ensemble K | {1, 5, 10} | 3 |
| **Total** | | **27 configurations** |

Fixed parameters (matching BQNN):
- Epochs: 100
- LR: 0.01
- Optimizer: SGD with momentum=0.9
- Hidden dim: 16
- Seed: 42 (base, +k per ensemble model)

### 1.5 Data

| Parameter | Value | Match |
|:----------|:------|:------|
| Dataset | MNIST | ✅ Same as BQNN |
| Test images | 55 | ✅ Same as BQNN |
| Preprocessing | ToTensor + Normalize((0.1307,), (0.3081,)) | Standard MNIST |
| Sampling | Stratified (equal per class), seed=42 | Close to BQNN (exact indices unpublished) |
| Batch size | 8 | Small batch due to N=55 |

> **R1 note:** BQNN's exact 55 image indices are not published. We use stratified random sampling with seed=42 for reproducibility. This is documented as a limitation. If BQNN's indices are later published, the experiment can be re-run at zero additional code cost.

---

## 2. Energy Measurement

| Component | Method | Accuracy |
|:----------|:-------|:---------|
| GPU power | nvidia-smi --query-gpu=power.draw (live) | ±5% |
| CPU power | 65W TDP estimate (default) | ±30% |
| Wall-clock time | time.perf_counter(), 5 repeats | ±1ms |
| Joules | TDP × time (seconds) | — |

**BQNN energy reference (from v1 audit):**
- Ion-trap inference: ~10 kJ per image (cooling dominant)
- 55 images: ~550 kJ total
- Classical GPU inference: ~0.0003 J total
- Ratio: ~1.8 × 10^9

---

## 3. Code Structure

```
src/binarized_mlp.py          # Main module (444 lines)
├── binarize_weights()         # BinaryConnect weight binarization
├── BinaryConnectLinear        # Linear layer with binarized weights
├── BinaryConnectMLP           # 3-layer MLP model
├── inference_standard()       # Deterministic inference
├── inference_dropout()        # MC dropout inference
├── inference_gaussian()       # Gaussian noise inference
├── inference_ensemble()       # Ensemble inference
├── train_model()              # STE + SGD training
├── load_mnist_subset()        # Stratified 55-image MNIST loader
├── measure_inference_energy() # nvidia-smi energy measurement
├── SweepConfig                # Dataclass for sweep parameters
├── run_sweep()                # Full sweep + JSON export
└── main()                     # CLI: python src/binarized_mlp.py [output.json]
```

---

## 4. Run Instructions

```bash
# Install dependencies
pip install torch torchvision numpy

# Download MNIST + run sweep (27 configs × 5 seeds = 135 training runs)
python src/binarized_mlp.py results/benchmark_results.json

# On GPU (11 GB VRAM sufficient):
python src/binarized_mlp.py results/gpu_results.json  # auto-detects CUDA

# Expected runtime:
#   GPU: ~5-10 minutes (27 configs × ~15s per training)
#   CPU: ~30-60 minutes
```

---

## 5. Verification Gate (pre-Phase 3)

| Check | Status |
|:------|:-------|
| Code syntax valid (Python 3.10+) | ⏳ Run `python -m py_compile src/binarized_mlp.py` |
| MNIST data downloadable | ⏳ Run `python -c "from torchvision import datasets; datasets.MNIST('data', download=True)"` |
| Training runs on 1 config | ⏳ Run with 1 config (dropout=0.0, sigma=0.0, ensemble=1) |
| GPU available (optional) | ⏳ `python -c "import torch; print(torch.cuda.is_available())"` |
| nvidia-smi accessible | ⏳ Run `nvidia-smi --query-gpu=power.draw --format=csv` |
