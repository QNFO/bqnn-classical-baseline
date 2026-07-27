# BQNN Classical Competitive Baseline — v2

**Series:** The Qubit Delusion — Quantum Advantage Audit  
**Prior Work:** v1 — "Auditing the BQNN" (DOI: [10.5281/zenodo.21566035](https://doi.org/10.5281/zenodo.21566035))  
**Status:** Phase 0 — Initialized (2026-07-27)

## Overview

The v1 BQNN audit (published 2026-07-25) evaluated Lakhdar-Hamina et al. (2025, PRL) and found all three quantum advantage claims **NOT SUSTAINED**. The paper never ran a competitive classical baseline.

**This v2 project builds that baseline.** A classical binarized MLP with stochastic regularization (dropout, Gaussian noise injection, MC ensemble) is implemented, benchmarked head-to-head against BQNN's MNIST results, and the calibration prediction CAL-BQNN-05 is executed.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib scipy

# Reproduce baseline
python src/binarized_mlp.py

# Run full benchmark
python src/benchmark.py
```

## Project Structure

```
bqnn-classical-baseline/
├── README.md
├── PROJECT-PLAN.md
├── .gitignore
├── paper.md              # Final publication
├── paper.pdf             # Rendered PDF
├── docs/                 # Source documents, prior work PDFs
├── artifacts/            # Phase deliverables (due diligence, implementation, benchmarking)
├── notebooks/            # Working notes, calculation notebooks
├── releases/             # Zenodo-ready bundles
└── src/                  # Python source code
```

## License

QNFO Unified License Agreement (QNFO-ULA)
