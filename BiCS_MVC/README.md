# BiCS-MVC

**Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering**

Official PyTorch implementation. This repository contains the complete training code, evaluation pipeline, and experiment scripts for the BiCS-MVC paper.

---

## Project Structure

```
BiCS_MVC/
├── README.md
├── __init__.py          # Module entry; exports all public APIs
├── config.py            # Hyperparameters, paths, device selection
├── dataset.py           # MultiViewDataset — loads .mat files
├── losses.py            # L_bi-con + L_sem
├── model.py             # BiCSMVC model
├── metrics.py           # ACC (Hungarian), NMI, ARI, F-score
├── trainer.py           # Training loop + K-Means evaluation
└── run_repeat.py        # Entry point
```

---

## Quick Start

```bash
# Train on all datasets
python run_repeat.py

# Train on specific datasets
python run_repeat.py Hdigit
```

---
