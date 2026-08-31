# Adversarial Graph Contrastive Learning with Topology-Aware Multimodal Fusion for Cross-Domain Emotion Recognition

<div align="center">

**Joon Young Lee**, **Dae Hyeon Kim**, **Young-Seok Choi**<sup>*</sup>

Department of Electronics and Communications Engineering, Kwangwoon University, Seoul, Republic of Korea

![Journal](https://img.shields.io/badge/Journal-Knowledge--Based%20Systems-2f6f9f.svg)
![Status](https://img.shields.io/badge/Status-Under%20Revision-f39c12.svg)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776ab.svg)

<sup>*</sup>Corresponding author

</div>

---

## 📢 Manuscript status

> **Important:** The manuscript has been submitted to *Knowledge-Based Systems* and is currently under revision. The manuscript itself has not yet been updated to reflect the latest revision work, so some descriptions may temporarily differ from this code release. This notice does not indicate acceptance or publication.

The title and author information above follow the current manuscript draft and may be updated as the revision proceeds.

## 🔍 Overview

This repository contains the single-model implementation of **Multimodal Domain Adversarial Graph Contrastive Learning (MDA-GCL)** for cross-subject and cross-session emotion recognition using electroencephalography (EEG) and eye-movement (EM) features.

MDA-GCL combines three components:

1. **Symmetric Similarity Network Fusion (SSNF)** constructs a sample-level multimodal affinity graph from EEG and EM feature views.
2. **Graph Contrastive Learning (GCL)** encourages representations that remain stable across stochastic graph perturbations.
3. **Domain-Adversarial Neural Network (DANN)** reduces subject- and session-specific domain bias.

The graph represents similarities between temporal sample windows. It is not an anatomical brain-connectivity or EEG-attention map.

## 📝 Abstract
---
Multimodal emotion recognition leveraging electroencephalography (EEG) and eye-movement (EM) signals is a fundamental component in affective computing. However, modeling the cross-modal correlations between heterogeneous modalities and mitigating the domain shifts caused by individual physiological variability and temporal measurement fluctuations remain significant challenges. In this paper, we propose a framework, Multimodal Domain Adversarial Graph Contrastive Learning (MDA-GCL), to address these issues in cross-subject and cross-session scenarios. Specifically, we first introduce a Symmetric Similarity Network Fusion (SSNF) strategy to construct a unified multimodal graph, which serves as a structured sample-level affinity manifold representing the complementary physiological dynamics between the central and peripheral nervous systems. To counteract the risk of negative transfer—where forced domain alignment inadvertently blurs emotion class boundaries—we integrate a Domain-Adversarial Neural Network (DANN) with Graph Contrastive Learning (GCL). While GCL extracts task-relevant perturbation-invariant representations to preserve class separability, DANN simultaneously reduces domain-specific nuisance biases via minimax optimization. Experiments on the SEED-IV and SEED-V datasets show that MDA-GCL achieves 76.1% and 83.8% on SEED-IV, and 75.6% and 82.5% on SEED-V under batch-wise transductive cross-subject and cross-session protocols, respectively. The topology and module ablations indicate that the performance gains arise from the combined effect of the SSNF-based graph representation and the downstream GCL--DANN joint learning objective rather than from any single component alone. The implementation of MDA-GCL and the experimental configurations used in this study are publicly available at GitHub: https://github.com/LeeJY-KW/MDA-GCL.

---

## 📦 Repository contents

```text
MDA-GCL-release-v3/
├── configs/                    # Four dataset/protocol configurations
├── docs/
│   ├── DATA.md                 # External array layout and validation contract
│   └── REPRODUCIBILITY.md      # Complete execution workflow
├── src/mda_gcl/                # Model, graph, protocol, training, and CLI code
├── tests/                      # Unit and synthetic pipeline tests
└── pyproject.toml
```

Datasets, preprocessing code, checkpoints, generated outputs, plots, and experimental records are not included. Feature arrays must be supplied separately.

## ⚙️ Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m mda_gcl.cli --help
```

Real training requires a CUDA-capable PyTorch environment. The synthetic smoke test can run on CPU.

## 🗂️ Data preparation

The runner expects externally supplied NumPy arrays:

| Dataset | Feature views | Classes | Evaluation protocols |
| --- | --- | ---: | --- |
| SEED-IV | DE, PSD, eye movement | 4 | Cross-subject, cross-session |
| SEED-V | DE, eye movement | 5 | Cross-subject, cross-session |

See [docs/DATA.md](docs/DATA.md) for filenames, dimensions, label encoding, and directory layout.

## 🚀 Quick start

Run a deterministic in-memory smoke test without dataset files:

```bash
.venv/bin/python -m mda_gcl.cli \
  --config configs/seed_v_cross_session.yaml \
  --data-root . \
  --output-dir /tmp/mda-gcl-smoke \
  --device cpu \
  --dry-run
```

Validate external arrays and protocol folds without training:

```bash
.venv/bin/python -m mda_gcl.cli \
  --config configs/seed_v_cross_session.yaml \
  --data-root /path/to/data \
  --output-dir /tmp/mda-gcl-preflight \
  --device cpu \
  --preflight-only
```

Run one complete experiment:

```bash
.venv/bin/python -m mda_gcl.cli \
  --config configs/seed_v_cross_session.yaml \
  --data-root /path/to/data \
  --output-dir /path/to/output \
  --device cuda:0
```

Available configurations:

- `configs/seed_iv_cross_subject.yaml`
- `configs/seed_iv_cross_session.yaml`
- `configs/seed_v_cross_subject.yaml`
- `configs/seed_v_cross_session.yaml`

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete four-configuration workflow.

## 🔬 Evaluation contract

- Source and target feature views are standardized **separately** for every fold.
- Target feature statistics may be used transductively.
- Target emotion labels are not used for graph construction, normalization, training, or checkpoint selection.
- The configured final epoch is evaluated once on the clean normalized graph.
- The same MDA-GCL architecture is used for all four dataset/protocol configurations.

## 📄 Outputs

Each completed run writes only to the requested output directory:

- `results.json`: effective configuration, provenance, fold outputs, and aggregate metrics.
- `fold_results.csv`: one row per fold with metrics, predictions, targets, and provenance.

## 🧪 Local test

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider -W error
```

## 📝 Citation

The final citation will be added after the journal revision is completed and the manuscript metadata has been updated. Until then, the current title and bibliographic information should be treated as provisional.
