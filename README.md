# A Transformer Emulator for Multiple Spreading Processes on Networks

Code, datasets, trained models, and results for the paper:

> **A Transformer Emulator for Multiple Spreading Processes: From the Zealot Voter Model to Epidemics and Bounded Confidence**
> Vahid Moeinifar, AGH University of Krakow.

This repository contains everything needed to reproduce the paper. The core model is the **ZealotTransformer (ZT)**: a graph Transformer encoder with dual zealot/non-zealot pooling and an LSTM trajectory decoder. The same architecture, retrained without structural change, emulates three distinct dynamical families:

- **Zealot Voter Model** — opinion copying; observable: magnetization `m(t)`.
- **SIS epidemic model** — simple contagion with permanently infected seeds; observable: infected fraction `i(t)`.
- **Deffuant–Weisbuch model** — bounded-confidence clustering with stubborn agents; observable: number of opinion clusters `n_clust(t)`.

The voter-model experiments were run on the **LUMI** supercomputer (AMD MI250X GPUs). The additional SIS and Deffuant experiments were run on **Kaggle** (single NVIDIA T4 GPU), reflecting their smaller training distributions and the modest model size (~4M parameters).

---

## Repository structure

```
.
├── 1-Kaggle/                     # SIS & Deffuant experiments (NVIDIA T4, Kaggle) 
│   ├── scripts files             # Kaggle notebooks / Python scripts
│   ├── datasets and models/      # Generated trajectory datasets + trained ZT weights
│   └── results/                  # Output tables (xlsx / txt) for SIS & Deffuant
│
└── 2-LUMI/                       # Voter-model experiments (AMD MI250X, LUMI)
    ├── scripts + .sh files       # Training/evaluation scripts and Slurm batch files
    ├── result/                   # Output tables for the voter model
    └── saved model/              # Trained voter-model weights
```

---

## 1-Kaggle — SIS and Deffuant

This folder reproduces the multi-process results (the cross-process generalization tables in the paper).

### Contents
- **`scripts/`** — the kernels (Monte-Carlo simulators), trainers, and evaluators:
  - SIS: dataset kernel, trainer, single-cell evaluator.
  - Deffuant: dataset kernel, trainer, single-cell evaluator.
- **`datasets and models/`** — the `.npz` trajectory datasets and the trained ZT weights (`.pt`) for each process.
- **`results/`** — the evaluation outputs (`.xlsx` and `.txt`) used to build the paper tables.

### How the processes are simulated
- **SIS:** infection rate `β = 1.5·γ/λ_max(A)`, recovery rate `γ = 0.5` (endemic regime). Seeds are permanently infected. Observable `i(t) ∈ [0,1]`.
- **Deffuant:** confidence threshold `ε = 0.15`, convergence rate `μ = 0.5`. Stubborn agents hold a fixed opinion. Observable is the cluster count `n_clust(t)`, normalised by `C_max = 20` during training and rescaled to clusters at evaluation.

### Training distribution (Kaggle processes)
To fit the available compute, SIS and Deffuant were trained on a reduced distribution relative to the voter model:
- Topologies: BA, ER, WS
- Sizes: `N ∈ {256, 512, 1024}`
- Seeds/stubborn: `Z = 8`
- Placements: hub and random
- MC runs per trajectory: **128 for SIS**, **32 for Deffuant** (the cluster-count trajectory is smooth; 32 vs 128 runs differ by < 0.2 clusters).

Training uses **size-bucketed batching** with a fixed node budget per batch (effective batch size from 32 at `N=1024` to 128 at `N=256`) to bound the `O(N²)` attention memory.

### Reproduce (Kaggle)
1. **Generate a dataset** (CPU session): run the relevant kernel's `generate_dataset(...)` to produce e.g. `sis_dataset.npz` / `deffuant_dataset.npz`.
2. **Train** (GPU session): run the matching trainer, pointing `DATA_PATH` at the dataset; it saves `zt_sis.pt` / `zt_deffuant.pt`.
3. **Evaluate**: run the single-cell evaluator with `WEIGHTS` set to the trained `.pt`; it writes the two result tables (main + out-of-distribution) to `results/`.

Each evaluator is fully self-contained — paste it into a single Kaggle cell and commit.

---

## 2-LUMI — Voter model

This folder reproduces the main voter-model results (the five-model comparison, placement, size, and OOD tables).

### Contents
- **Scripts + `.sh` files** — the training/evaluation code and the Slurm batch scripts used on LUMI.
- **`result/`** — voter-model evaluation tables.
- **`saved model/`** — trained voter-model weights, including the full ZealotTransformer and the lightweight BA-only variant.

### Models
Five progressive emulators are included:
1. **Specialist-Low** — GAT trained on `Z=2` only.
2. **Global-GAT** — GAT trained on all `Z`.
3. **SpectralLSTM** — 8-D spectral descriptor + LSTM decoder.
4. **PA-LSTM** — 11-D placement-aware descriptor + LSTM decoder.
5. **ZealotTransformer (ZT)** — the main contribution.

### Training distribution (voter model)
- Topologies: BA, ER, WS
- Sizes: `N ∈ {256, 512, 1024, 2048}`
- Zealots: `Z ∈ {2, 8, 16, 32}`
- Placements: hub and random (bridge is held out for OOD testing)
- Evaluation: 128 MC runs (32 for large `N`), 10 graphs per configuration.

### Reproduce (LUMI / any GPU)
Submit the provided `.sh` Slurm scripts, or run the Python scripts directly on a CUDA/ROCm GPU. Trained weights are saved to `saved model/` and evaluation tables to `result/`.

---

## Node features

Every process uses the same five node features per node `i`:
`x_i = [z_i, k̄_i, f_i, π_i, c_i]`
where `z_i` is the seed/zealot/stubborn indicator, `k̄_i` the normalised degree, `f_i` the Fiedler-vector coordinate, `π_i` the normalised PageRank, and `c_i` the local clustering coefficient.

## Model summary

- Encoder: node-feature embedding → multi-head self-attention Transformer (D=256, 4 heads, 3 layers, pre-norm, GELU).
- Pooling: separate mean-pool over seed nodes and non-seed nodes → concatenated context vector.
- Decoder: 2-layer LSTM, autoregressive over `T=50` steps, sigmoid output head.
- Loss: time-weighted RMSE (later steps weighted more heavily).

## Known limitation

Across **all three processes**, the Random Geometric Graph (RGG) topology is the consistent weak point: the node-feature set encodes the bottleneck structure of BA/ER/WS graphs but not the spatial embedding of RGG. This is reported honestly in the paper.

## Requirements

- Python 3.10+
- PyTorch, NumPy, NetworkX, pandas, openpyxl
- A CUDA or ROCm GPU is recommended for training (the model is small enough to also run on CPU for inference).

## Citation

If you use this code or data, please cite the paper (full reference above). A `CITATION` entry will be added once the DOI is assigned.

## Contact

Vahid Moeinifar — vmoeinifar@agh.edu.pl
AGH University of Krakow, Faculty of Electrical Engineering, Automatics, Computer Science and Biomedical Engineering.
