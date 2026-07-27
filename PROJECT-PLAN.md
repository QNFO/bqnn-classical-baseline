# PROJECT-PLAN: BQNN Classical Competitive Baseline — v2

| Field | Value |
|:------|:------|
| **Project** | BQNN Classical Competitive Baseline |
| **Series** | The Qubit Delusion — Quantum Advantage Audit |
| **Prior Work** | v1 — "Auditing the BQNN" (DOI: 10.5281/zenodo.21566035) |
| **Status** | Phase 0 — INITIALIZED |
| **Created** | 2026-07-27 |
| **License** | QNFO Unified License Agreement (QNFO-ULA) |

---

## §1 Charter

### 1.1 Problem Statement

The v1 BQNN audit found that all three quantum advantage claims in Lakhdar-Hamina et al. (2025, PRL) are **NOT SUSTAINED**. The paper's own methodology is valuable (multi-platform QNN benchmarking), but its advantage claims rest on: (a) self-comparison without any competitive classical baseline, (b) N = 55 images with overlapping error bars, (c) a single cherry-picked NY image for noise-as-advantage, and (d) a fully separable (classically simulable) circuit.

The BQNN authors never ran the competitive classical baseline that would test their null hypothesis: that classical stochastic regularization — dropout, Gaussian noise injection at inference, MC ensemble averaging — achieves equivalent or superior performance at near-zero energy cost. This v2 project **constructs that missing baseline**.

### 1.2 Core Claim (Locked at Phase 0)

> **CC-01:** A classical binarized multilayer perceptron (BinaryConnect architecture, Courbariaux et al. 2015) with tuned stochastic regularization — specifically: (a) Bernoulli dropout at training, (b) Gaussian noise injection at inference, and (c) Monte Carlo ensemble averaging — **matches or exceeds** BQNN's validation rate on the same MNIST subset under the same experimental protocol, with a joules-per-solution ratio of ≤ 10⁻⁶ compared to BQNN's quantum inference.

> **Falsification condition:** If the classical baseline, after fair hyperparameter tuning and identical train/test splits, achieves validation rate ≥ 2% below BQNN's quantum inference at a = 0.5 AND this gap is statistically significant (p < 0.05 after Bonferroni correction for the 3 classical methods tested), then CC-01 is falsified and quantum advantage at inference warrants further investigation.

### 1.3 What v2 Adds Beyond v1

| v1 (Audit) | v2 (Constructive) |
|:-----------|:------------------|
| Identified gap (no classical baseline) | **Builds** the classical baseline |
| Estimated joules-per-solution theoretically | **Measures** joules-per-solution empirically |
| Registered CAL-BQNN-05 prediction | **Executes** CAL-BQNN-05 |
| Conclusion: "NOT SUSTAINED" | Conclusion: empirical verdict + revised calibration |
| 5 calibration predictions registered | Updates calibration register with actual results |

---

## §2 Work Breakdown Structure

### Phase 0: Project Initialization
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 0.1 | Scaffold directory structure (docs/, artifacts/, notebooks/, releases/, src/) | HARD | ✅ |
| 0.2 | Create PROJECT-PLAN.md with charter, WBS, risk register | HARD | ✅ |
| 0.3 | Create .gitignore, git init, GitHub repo | HARD | pending |
| 0.4 | Core claim lock (§1.2) | HARD | ✅ |
| 0.5 | Commit + tag v0.1-phase0 + push | HARD | pending |

### Phase 1: Due Diligence — Classical Baselines Literature
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 1.1 | KG/D1/Vectorize query — prior QNFO classical ML benchmarking | HARD | pending |
| 1.2 | External literature — classical binarized NN, stochastic regularization, competitive baselines for QML | HARD | pending |
| 1.3 | Gap analysis — identify exact architecture and hyperparameter search space | SOFT | pending |
| 1.4 | Due diligence report (artifacts/phase1-due-diligence.md) | HARD | pending |

### Phase 2: Implementation — Classical Baseline
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 2.1 | Reproduce BQNN's exact MNIST subset (55 images, same indices if available) | HARD | pending |
| 2.2 | Implement BinaryConnect MLP in PyTorch (3 layers, 16 neurons/layer, ±1 activations) | HARD | pending |
| 2.3 | Implement BQNN-equivalent training protocol (STE, SGD, identical epochs) | HARD | pending |
| 2.4 | Implement 3 stochastic regularization methods: (a) Bernoulli dropout, (b) Gaussian inference noise, (c) MC ensemble | HARD | pending |
| 2.5 | Hyperparameter sweep over dropout rate, noise σ, ensemble size | HARD | pending |
| 2.6 | Document implementation in artifacts/phase2-implementation.md | HARD | pending |

### Phase 3: Benchmarking — Head-to-Head Comparison
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 3.1 | Run classical baseline on BQNN's MNIST subset | HARD | pending |
| 3.2 | Re-extract BQNN's reported metrics from arXiv:2507.21222v2 | HARD | pending |
| 3.3 | Statistical comparison: validation rate, error bars, significance tests | HARD | pending |
| 3.4 | Joules-per-solution measurement (wall-clock time × TDP for GPU vs ion-trap) | HARD | pending |
| 3.5 | Noise-as-advantage reproduction: test on BQNN's image 6929 (if identifiable) | SOFT | pending |
| 3.6 | Benchmarking report (artifacts/phase3-benchmarking.md) | HARD | pending |

### Phase 4: Analysis & Synthesis
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 4.1 | Statistical analysis: CC-01 verdict (falsified or sustained?) | HARD | pending |
| 4.2 | CAL-BQNN-05 execution: register results | HARD | pending |
| 4.3 | Update calibration register (add CAL-BQNN-06 through CAL-BQNN-08) | HARD | pending |
| 4.4 | Red-team self-audit (5-adversary protocol) | HARD | pending |
| 4.5 | Analysis report (artifacts/phase4-analysis.md) | HARD | pending |

### Phase 5: Publication
| ID | Task | Gate | Status |
|:---|:-----|:-----|:-------|
| 5.1 | Write paper.md (Springer Nature professional standards) | HARD | pending |
| 5.2 | Publication Language Gate + Physics Writing Standards + credential scan | HARD | pending |
| 5.3 | Build PDF via build-paper.py (zero U+FFFD/U+FFFF) | HARD GATE P5.PDF | pending |
| 5.4 | Zenodo newversion deposit + PROVENANCE-BUNDLE.zip | HARD | pending |
| 5.5 | D1 living-paper insert + papers-server verification | HARD | pending |
| 5.6 | R2 archive upload | HARD | pending |
| 5.7 | Buffer social media (Twitter, LinkedIn) | SOFT | pending |
| 5.8 | GitHub push + tag + release | HARD | pending |

---

## §3 Deliverable Registry

| ID | Deliverable | Path | R2 Path | Status |
|:---|:------------|:-----|:--------|:-------|
| D-01 | PROJECT-PLAN.md | PROJECT-PLAN.md | qnfo-releases/releases/2026/07/bqnn-classical-baseline/ | ✅ |
| D-02 | Due Diligence Report | artifacts/phase1-due-diligence.md | same | pending |
| D-03 | Implementation Report | artifacts/phase2-implementation.md | same | pending |
| D-04 | Source Code | src/*.py | same | pending |
| D-05 | Benchmarking Report | artifacts/phase3-benchmarking.md | same | pending |
| D-06 | Analysis Report | artifacts/phase4-analysis.md | same | pending |
| D-07 | Paper (markdown) | paper.md | same | pending |
| D-08 | Paper (PDF) | paper.pdf | same | pending |
| D-09 | PROVENANCE-BUNDLE.zip | PROVENANCE-BUNDLE.zip | same | pending |

---

## §4 Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|:---|:-----|:-----------|:-------|:-----------|
| R1 | BQNN's exact MNIST image indices not published → cannot reproduce identical test set | HIGH | MEDIUM | Sample 55 random MNIST images with same class distribution; document discrepancy |
| R2 | BQNN's training hyperparameters (learning rate, epochs, optimizer config) not fully specified | MEDIUM | MEDIUM | Standard STE/SGD defaults; sensitivity analysis across parameter ranges |
| R3 | Classical baseline underperforms due to insufficient hyperparameter tuning → false positive for quantum | MEDIUM | HIGH | Comprehensive sweep; report all results including worst-case classical |
| R4 | BQNN's image 6929 not identifiable from paper → cannot test noise-as-advantage claim directly | HIGH | LOW | Test on ALL MNIST images; report proportion where noise injection helps |
| R5 | Classical baseline requires GPU cluster to match BQNN's training → energy comparison skewed | LOW | MEDIUM | Measure GPU energy with nvidia-smi; report total training + inference cost |
| R6 | D1 / Zenodo / R2 publish failures (known infrastructure fragility per KIF history) | LOW | CRITICAL | Build-paper.py pre-verification; d1-query.py auto-discovery; credential-scan pre-flight |

---

## §5 Success Criteria

1. **CC-01 verdict delivered** — classical baseline compared against BQNN with statistical rigor
2. **CAL-BQNN-05 executed** — prediction checked, dated result registered
3. **Empirical joules-per-solution measured** — not theoretical estimate, actual measurement
4. **All 5 calibration predictions updated** — check status, register new predictions
5. **Paper published with Zenodo DOI** — v2.0 newversion of concept 10.5281/zenodo.21566035
6. **Code and data archived** — src/ + experimental logs in R2

---

## §6 Version History

| Version | Date | Changes |
|:--------|:-----|:--------|
| v0.1 | 2026-07-27 | Phase 0 scaffold — charter, WBS, risk register, core claim lock |
