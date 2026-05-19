"""
SpectralZealot Trajectory Predictor
====================================
Architecture improvements over PA-LSTM:

1. Spectral zealot embedding  — phi_Z = U_k^T z  (k=8 eigenvectors)
   replaces weak scalar hub/bridge/fiedler scores
2. Mixed training sizes       — N in {256, 512, 1024, 2048}
3. Scale-aware graph features — diameter, avg_path, spectral_radius,
                                modularity estimate added to descriptor
4. Non-autoregressive decoder — MLP predicts full T-step trajectory at
                                once (no error accumulation)
5. Uncertainty output         — predicts mu(t) and log_sigma(t) per step,
                                trained with Gaussian NLL loss

Final descriptor dimension:
  Base (8):     rho_Z, lambda_2, mean_deg/N, CV(deg), C,
                topo_ba, topo_er, topo_ws
  Scale (4):    diameter/N, avg_path/N, spectral_radius/N, modularity
  Placement (k=8):  phi_Z = U_k^T z  (spectral zealot embedding)
  Total:  8 + 4 + 8 = 20D

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
torch.set_num_threads(NUM_CPUS)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ALL_Z        = [2, 8, 16, 32]
TRAIN_N_LIST = [256, 512, 1024, 2048]   # mixed sizes during training
T_STEPS      = 50
MC_RUNS      = 20
K_EIGS       = 8     # number of Laplacian eigenvectors for zealot embedding
BASE_DIM     = 8
SCALE_DIM    = 4
PLACEMENT_DIM = K_EIGS
DESC_DIM     = BASE_DIM + SCALE_DIM + PLACEMENT_DIM   # = 20


# ─────────────────────────────────────────────────────────────
# Zealot placement strategies
# ─────────────────────────────────────────────────────────────

def place_hubs(G, num_zealots, rng=None):
    return set(n for n, _ in
               sorted(G.degree(), key=lambda x: x[1], reverse=True)[:num_zealots])

def place_random(G, num_zealots, rng):
    nodes = list(G.nodes())
    return set(int(n) for n in rng.choice(nodes, size=num_zealots, replace=False))


# ─────────────────────────────────────────────────────────────
# Graph descriptors — 20D
# ─────────────────────────────────────────────────────────────

def compute_laplacian_eigs(G, k):
    """
    Returns (eigenvalues [k], eigenvectors [N, k]).
    Sorted by eigenvalue ascending (lambda_0=0 excluded, so indices 1..k).
    Falls back to zeros on failure.
    """
    n = G.number_of_nodes()
    try:
        L    = nx.laplacian_matrix(G).astype(float)
        nev  = min(k + 1, n - 1)
        vals, vecs = eigsh(L, k=nev, which='SM', tol=1e-3, maxiter=3000)
        order = np.argsort(vals)
        vals  = vals[order]
        vecs  = vecs[:, order]
        # Drop the zero eigenvector (index 0), keep next k
        vals  = vals[1:k+1]
        vecs  = vecs[:, 1:k+1]
        # Pad if fewer than k eigenvalues
        if vals.shape[0] < k:
            pad_v = np.zeros(k - vals.shape[0])
            pad_e = np.zeros((n, k - vals.shape[0]))
            vals  = np.concatenate([vals, pad_v])
            vecs  = np.concatenate([vecs, pad_e], axis=1)
        return vals.astype(np.float32), vecs.astype(np.float32)
    except Exception:
        return np.zeros(k, dtype=np.float32), np.zeros((n, k), dtype=np.float32)


def spectral_zealot_embedding(vecs, zealot_set, n):
    """
    phi_Z = U_k^T z  where z is the zealot indicator vector (normalised).
    Returns vector of shape (k,).
    """
    z = np.zeros(n, dtype=np.float32)
    for node in zealot_set:
        z[node] = 1.0
    z /= (np.linalg.norm(z) + 1e-8)   # L2 normalise
    return vecs.T @ z   # (k,)


def compute_scale_features(G):
    """
    4 scale-aware features, all normalised by N.
    Uses approximations for speed on large graphs.
    """
    n = G.number_of_nodes()

    # Diameter — sample-based approximation (exact too slow for N=2048)
    try:
        sample_nodes = list(G.nodes())[:min(50, n)]
        eccs = [nx.eccentricity(G, v=v) for v in sample_nodes]
        diameter = float(max(eccs)) / n
    except Exception:
        diameter = 0.0

    # Average shortest path — sample-based
    try:
        sample = random.sample(list(G.nodes()), min(100, n))
        lengths = []
        for src in sample:
            sp = nx.single_source_shortest_path_length(G, src)
            lengths.extend(sp.values())
        avg_path = float(np.mean(lengths)) / n
    except Exception:
        avg_path = 0.0

    # Spectral radius of adjacency matrix (largest eigenvalue / N)
    try:
        A    = nx.adjacency_matrix(G).astype(float)
        vals = eigsh(A, k=1, which='LM', return_eigenvectors=False,
                     tol=1e-3, maxiter=1000)
        spectral_radius = float(vals[0]) / n
    except Exception:
        spectral_radius = 0.0

    # Modularity estimate via degree-based null model
    try:
        degrees  = np.array([d for _, d in G.degree()], dtype=np.float64)
        m_edges  = G.number_of_edges()
        # Greedy communities approximation
        communities = nx.community.greedy_modularity_communities(G)
        mod = nx.community.modularity(G, communities)
        modularity = float(np.clip(mod, -1.0, 1.0))
    except Exception:
        modularity = 0.0

    return np.array([diameter, avg_path, spectral_radius, modularity],
                    dtype=np.float32)


def compute_descriptor(G, zealot_set, topo_type, k=K_EIGS):
    """
    Full 20D descriptor.
    G            : networkx Graph
    zealot_set   : set of zealot node indices
    topo_type    : 'ba' | 'er' | 'ws'
    k            : number of Laplacian eigenvectors
    """
    n        = G.number_of_nodes()
    Z        = len(zealot_set)
    degrees  = np.array([d for _, d in G.degree()], dtype=np.float64)
    mean_deg = degrees.mean()

    # ── Base 8 features ──────────────────────────────────────
    rho_z      = Z / n
    eig_vals, eig_vecs = compute_laplacian_eigs(G, k)
    lambda_2   = float(eig_vals[0]) if len(eig_vals) > 0 else 0.0
    norm_deg   = mean_deg / n
    cv_deg     = degrees.std() / (mean_deg + 1e-8)
    clustering = nx.average_clustering(G)
    topo_ba = 1.0 if topo_type == 'ba' else 0.0
    topo_er = 1.0 if topo_type == 'er' else 0.0
    topo_ws = 1.0 if topo_type == 'ws' else 0.0

    base = np.array([rho_z, lambda_2, norm_deg, cv_deg, clustering,
                     topo_ba, topo_er, topo_ws], dtype=np.float32)

    # ── Scale features (4) ───────────────────────────────────
    scale = compute_scale_features(G)

    # ── Spectral zealot embedding (k=8) ──────────────────────
    phi_z = spectral_zealot_embedding(eig_vecs, zealot_set, n)

    return np.concatenate([base, scale, phi_z]).astype(np.float32)


def normalize_descriptors(descriptors, stats=None):
    if stats is None:
        mean  = descriptors.mean(axis=0)
        std   = descriptors.std(axis=0) + 1e-8
        stats = (mean, std)
    mean, std = stats
    return (descriptors - mean) / std, stats


# ─────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────

def make_graph(topo, n, m=8, seed=None):
    if topo == 'ba':
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == 'er':
        p = min(2 * m / (n - 1), 1.0)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(n, p, seed=(seed + attempt) if seed else None)
            if nx.is_connected(G):
                break
    elif topo == 'ws':
        G = nx.watts_strogatz_graph(n, max(4, 2*m), p=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown topo: {topo}")
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G, topo


# ─────────────────────────────────────────────────────────────
# Voter model simulation
# ─────────────────────────────────────────────────────────────

def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=MC_RUNS, seed=None):
    rng       = np.random.default_rng(seed)
    n         = G.number_of_nodes()
    adj       = [list(G.neighbors(i)) for i in range(n)]
    is_zealot = np.zeros(n, dtype=bool)
    for z in zealot_set:
        is_zealot[z] = True
    non_zealots = np.where(~is_zealot)[0]
    all_trajs   = []
    for _ in range(mc_runs):
        ops = rng.choice([-1.0, 1.0], size=n)
        ops[is_zealot] = 1.0
        traj = []
        for _ in range(T):
            traj.append(float(ops.mean()))
            chosen = rng.choice(non_zealots, size=len(non_zealots), replace=True)
            for node in chosen:
                nbrs = adj[node]
                if nbrs:
                    ops[node] = ops[rng.integers(0, len(nbrs))]
            ops[is_zealot] = 1.0
        all_trajs.append(traj)
    return np.mean(all_trajs, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

def _build_one_sample(args_tuple):
    """Worker function — builds 2 samples (hub + random) for one graph."""
    topo, n, num_zealots, g_idx, T, mc_runs = args_tuple
    seed = hash((topo, n, num_zealots, g_idx)) % (2**31)
    rng  = np.random.default_rng(seed)
    try:
        G, topo_type = make_graph(topo, n, m=8, seed=seed)
        results = []
        for strategy in ('hub', 'random'):
            zealot_set = (place_hubs(G, num_zealots)
                          if strategy == 'hub'
                          else place_random(G, num_zealots, rng))
            desc = compute_descriptor(G, zealot_set, topo_type, k=K_EIGS)
            traj = simulate_trajectory(G, zealot_set, T, mc_runs, seed=seed + 1)
            results.append((desc, traj))
        return results
    except Exception as e:
        print(f"  Worker error ({topo} N={n} Z={num_zealots} g={g_idx}): {e}",
              flush=True)
        return []


def build_dataset(zealot_list, num_graphs, T, mc_runs, num_workers=None):
    import multiprocessing as mp

    if num_workers is None:
        num_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 48)

    print("\nBuilding SpectralZealot dataset...", flush=True)
    print(f"  Z={zealot_list} | topologies=ba,er,ws | "
          f"N∈{TRAIN_N_LIST} | {num_graphs} graphs each", flush=True)
    print(f"  Placements: hub + random | desc_dim={DESC_DIM}", flush=True)
    print(f"  Parallel workers: {num_workers}", flush=True)

    topos = ['ba', 'er', 'ws']
    # Build full task list
    tasks = [
        (topo, n, num_zealots, g_idx, T, mc_runs)
        for topo in topos
        for n in TRAIN_N_LIST
        for num_zealots in zealot_list
        for g_idx in range(num_graphs)
    ]
    total_tasks   = len(tasks)
    total_samples = total_tasks * 2
    print(f"  Total tasks: {total_tasks}  "
          f"Total samples: {total_samples}", flush=True)

    t0 = time.time()
    descs, trajs = [], []

    # Use spawn context to avoid issues with CUDA + fork on LUMI
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(
                _build_one_sample, tasks, chunksize=4)):
            for desc, traj in results:
                descs.append(desc)
                trajs.append(traj)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (total_tasks - i - 1) / (rate + 1e-8)
                print(f"  {i+1}/{total_tasks} tasks  "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                      flush=True)

    # Shuffle
    idx = list(range(len(descs)))
    random.shuffle(idx)
    descs = [descs[i] for i in idx]
    trajs = [trajs[i] for i in idx]

    desc_arr = np.stack(descs)
    traj_arr = np.stack(trajs)
    print(f"\nDataset: {len(desc_arr)} samples in "
          f"{time.time()-t0:.1f}s", flush=True)
    print(f"  Descriptor shape: {desc_arr.shape}", flush=True)
    return desc_arr, traj_arr


# ─────────────────────────────────────────────────────────────
# Model — MLP decoder with uncertainty output
# ─────────────────────────────────────────────────────────────

class SpectralZealotPredictor(nn.Module):
    """
    Non-autoregressive trajectory predictor with uncertainty.

    Architecture:
      Descriptor encoder: DESC_DIM → 256 → 512 → 512  (shared trunk)
      Mean head:          512 → 256 → T  (sigmoid, trajectory in [0,1])
      Log-sigma head:     512 → 256 → T  (log standard deviation)

    Outputs mu(t) and log_sigma(t) for each time step.
    At inference, returns mu(t) (converted to [-1,1]).
    """
    def __init__(self, desc_dim=DESC_DIM, hidden_dim=512, T=T_STEPS,
                 dropout=0.1):
        super().__init__()
        self.T = T

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean head — predicts full trajectory at once
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, T),
            nn.Sigmoid()   # output in [0,1]
        )

        # Log-sigma head — predicts uncertainty
        self.logsigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, T),
        )

    def forward(self, x):
        """
        x : (B, desc_dim)
        Returns:
          mu       : (B, T)  in [0,1]
          logsigma : (B, T)  unconstrained
        """
        h        = self.encoder(x)
        mu       = self.mean_head(h)
        logsigma = self.logsigma_head(h)
        return mu, logsigma

    def predict_mean(self, x):
        """Inference-only: return mean trajectory in [-1,1]."""
        mu, _ = self.forward(x)
        return mu * 2 - 1   # (B, T) in [-1,1]


# ─────────────────────────────────────────────────────────────
# Loss — Gaussian NLL with time weighting
# ─────────────────────────────────────────────────────────────

def gaussian_nll_loss(mu, logsigma, target, T):
    """
    Gaussian NLL: -log N(target | mu, sigma^2)
    With time weighting: later steps penalised more (w_t = 1 + t/T).
    mu, logsigma, target: (B, T) in [0,1]
    """
    weights = torch.linspace(1.0, 2.0, T, device=mu.device)
    weights = weights / weights.sum()

    sigma2  = torch.exp(2 * logsigma).clamp(min=1e-6)
    nll     = 0.5 * (torch.log(sigma2) + (target - mu)**2 / sigma2)
    loss    = (nll * weights.unsqueeze(0)).mean()
    return loss


# ─────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, desc_t, traj_t, optimizer, device, batch_size):
    model.train()
    M   = desc_t.shape[0]
    idx = torch.randperm(M)
    total, count = 0.0, 0
    T = traj_t.shape[1]

    for start in range(0, M, batch_size):
        batch_idx  = idx[start:start + batch_size]
        desc_batch = desc_t[batch_idx].to(device)
        traj_batch = traj_t[batch_idx].to(device)

        optimizer.zero_grad()
        mu, logsigma = model(desc_batch)
        loss = gaussian_nll_loss(mu, logsigma, traj_batch, T)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); count += 1

    return total / count


@torch.no_grad()
def evaluate_rmse(model, desc_t, traj_t, device, batch_size):
    model.eval()
    M     = desc_t.shape[0]
    sq, n = 0.0, 0

    for start in range(0, M, batch_size):
        desc_batch = desc_t[start:start + batch_size].to(device)
        traj_batch = traj_t[start:start + batch_size].to(device)
        mu, _      = model(desc_batch)
        # Convert [0,1] → [-1,1] for RMSE
        p = mu.cpu().numpy() * 2 - 1
        t = traj_batch.cpu().numpy() * 2 - 1
        sq += np.sum((p - t)**2)
        n  += p.size

    return float(np.sqrt(sq / n))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",   type=int,   required=True)
    p.add_argument("--epochs",       type=int,   required=True)
    p.add_argument("--lr",           type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--num_graphs",   type=int,   required=True)
    p.add_argument("--mc_runs",      type=int,   required=True)
    p.add_argument("--T",            type=int,   required=True)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--save_dir",     type=str,   default="saved_models")
    p.add_argument("--save_name",    type=str,   default="spectral_zealot.pt")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\nSpectralZealot Predictor", flush=True)
    print(f"  desc_dim={DESC_DIM}  (base={BASE_DIM} + "
          f"scale={SCALE_DIM} + embedding={PLACEMENT_DIM})", flush=True)
    print(f"  k_eigs={K_EIGS}  train_N={TRAIN_N_LIST}", flush=True)
    print(f"  Non-autoregressive MLP decoder + Gaussian NLL", flush=True)
    print(f"Epochs={args.epochs} | Batch={args.batch_size} | "
          f"T={args.T} | mc_runs={args.mc_runs}", flush=True)

    # ── Dataset ───────────────────────────────────────────────
    desc_arr, traj_arr = build_dataset(ALL_Z, args.num_graphs,
                                       args.T, args.mc_runs,
                                       num_workers=args.num_workers)

    # Normalise trajectories to [0,1]
    traj_arr = (traj_arr + 1) / 2.0

    # Normalise descriptors
    desc_arr, norm_stats = normalize_descriptors(desc_arr)

    M     = len(desc_arr)
    split = int(0.8 * M)
    desc_t = torch.tensor(desc_arr[:split],  dtype=torch.float32)
    traj_t = torch.tensor(traj_arr[:split],  dtype=torch.float32)
    desc_v = torch.tensor(desc_arr[split:],  dtype=torch.float32)
    traj_v = torch.tensor(traj_arr[split:],  dtype=torch.float32)
    print(f"Train: {split} | Val: {M - split}", flush=True)

    # ── Model ─────────────────────────────────────────────────
    model = SpectralZealotPredictor(
        desc_dim=DESC_DIM,
        hidden_dim=args.hidden_dim,
        T=args.T,
        dropout=args.dropout
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup = 20
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_rmse  = float("inf")
    best_state = None

    print("\nTraining started\n", flush=True)
    print(f"  {'Ep':>5}  {'NLL Loss':>10}  {'RMSE':>8}  "
          f"{'Best':>8}  {'LR':>10}  {'Time':>7}", flush=True)
    print(f"  {'-'*60}", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, desc_t, traj_t, optimizer, device,
                           args.batch_size)
        rmse = evaluate_rmse(model, desc_v, traj_v, device, args.batch_size)
        scheduler.step()

        if rmse < best_rmse:
            best_rmse  = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch <= 10 or epoch % 20 == 0:
            print(f"  {epoch:>5}  {loss:>10.4f}  {rmse:>8.4f}  "
                  f"{best_rmse:>8.4f}  {lr_now:>10.2e}  "
                  f"{time.time()-t0:>6.1f}s", flush=True)

        if epoch % 50 == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "best_val_rmse":    best_rmse,
                "epoch":            epoch,
                "norm_stats":       norm_stats,
                "hyperparams": {
                    "desc_dim":     DESC_DIM,
                    "base_dim":     BASE_DIM,
                    "scale_dim":    SCALE_DIM,
                    "placement_dim": PLACEMENT_DIM,
                    "k_eigs":       K_EIGS,
                    "hidden_dim":   args.hidden_dim,
                    "T":            args.T,
                    "trained_Z":    ALL_Z,
                    "trained_N":    TRAIN_N_LIST,
                    "topologies":   ["ba", "er", "ws"],
                    "model_type":   "SpectralZealotPredictor",
                    "placements":   ["hub", "random"],
                    "decoder":      "MLP_nonAutoregressive",
                    "loss":         "GaussianNLL",
                }
            }, os.path.join(args.save_dir, args.save_name))
            print(f"  [checkpoint saved at epoch {epoch}]", flush=True)

    # ── Final save ────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    torch.save({
        "model_state_dict": best_state,
        "best_val_rmse":    best_rmse,
        "norm_stats":       norm_stats,
        "args":             vars(args),
        "hyperparams": {
            "desc_dim":      DESC_DIM,
            "base_dim":      BASE_DIM,
            "scale_dim":     SCALE_DIM,
            "placement_dim": PLACEMENT_DIM,
            "k_eigs":        K_EIGS,
            "hidden_dim":    args.hidden_dim,
            "T":             args.T,
            "trained_Z":     ALL_Z,
            "trained_N":     TRAIN_N_LIST,
            "topologies":    ["ba", "er", "ws"],
            "model_type":    "SpectralZealotPredictor",
            "placements":    ["hub", "random"],
            "decoder":       "MLP_nonAutoregressive",
            "loss":          "GaussianNLL",
        }
    }, ckpt_path)
    print(f"\n✓ Saved to {ckpt_path}", flush=True)
    print(f"  Best Val RMSE: {best_rmse:.4f}", flush=True)
    print(f"  desc_dim={DESC_DIM}: base={BASE_DIM} + "
          f"scale={SCALE_DIM} + embedding={PLACEMENT_DIM}", flush=True)


if __name__ == "__main__":
    main()