"""
Universal Magnetization Trajectory Predictor — Placement-Aware
===============================================================
Extension of SpectralLSTM that adds zealot placement descriptors
to the 8D graph descriptor, making it sensitive to *where* zealots
sit in the network, not just how many there are.

New descriptor (11D):
  Original 8D:
    1. rho_Z              — zealot density Z/N
    2. lambda_2           — algebraic connectivity
    3. mean_degree / N    — size-normalised mean degree
    4. CV(degree)         — degree heterogeneity
    5. global_clustering  — Watts-Strogatz C
    6. topo_ba            — topology one-hot
    7. topo_er            — topology one-hot
    8. topo_ws            — topology one-hot

  New placement features (+3D):
    9.  mean_degree_zealots / mean_degree_all  — hub score
                (>1 means zealots sit on hubs, ~1 means random)
   10.  mean_betweenness_zealots / mean_betweenness_all  — bridge score
   11.  zealot_fiedler_score  — |projection of zealot indicator onto
                                 Fiedler vector| / Z
                (captures how well zealots straddle the graph bottleneck)

Training covers two placement strategies:
  - Hub placement      (top-Z degree nodes, as before)
  - Random placement   (uniformly random non-hub nodes)

This forces the model to learn the placement → trajectory mapping
rather than relying on implicit hub-placement assumption.

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.sparse.linalg import eigsh

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

ALL_Z      = [2, 8, 16, 32]
N          = 1024
T_STEPS    = 50
MC_RUNS    = 20
DESC_DIM   = 11   # extended from 8


# ─────────────────────────────────────────────────────────────
# Zealot placement strategies
# ─────────────────────────────────────────────────────────────

def place_zealots_hubs(G, num_zealots, rng=None):
    """Top-Z nodes by degree (original strategy)."""
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:num_zealots])


def place_zealots_random(G, num_zealots, rng):
    """Uniformly random placement (new strategy)."""
    nodes = list(G.nodes())
    chosen = rng.choice(nodes, size=num_zealots, replace=False)
    return set(int(n) for n in chosen)


# ─────────────────────────────────────────────────────────────
# Graph descriptors — extended to 11D
# ─────────────────────────────────────────────────────────────

def compute_spectral_gap_and_fiedler(G):
    """
    Returns (lambda_2, fiedler_vector).
    fiedler_vector is the eigenvector for lambda_2 (shape N,).
    Falls back to zeros if computation fails.
    """
    try:
        L = nx.laplacian_matrix(G).astype(float)
        vals, vecs = eigsh(L, k=2, which='SM', tol=1e-3, maxiter=2000)
        order = np.argsort(vals)
        lambda_2 = float(max(vals[order[1]], 0.0))
        fiedler  = vecs[:, order[1]]          # (N,)
        return lambda_2, fiedler
    except Exception:
        n = G.number_of_nodes()
        return 0.0, np.zeros(n, dtype=np.float64)


def compute_descriptors(G, num_zealots, topo_type, zealot_set,
                        betweenness=None):
    """
    Compute 11D descriptor vector.

    Parameters
    ----------
    G           : networkx Graph
    num_zealots : int
    topo_type   : 'ba' | 'er' | 'ws'
    zealot_set  : set of zealot node indices (from any placement strategy)
    betweenness : precomputed betweenness dict or None (computed here if None)

    Returns
    -------
    desc : np.ndarray shape (11,)
    """
    N_g      = G.number_of_nodes()
    degrees  = np.array([d for _, d in G.degree()], dtype=np.float64)

    # ── Original 8 features ──────────────────────────────────
    rho_z      = num_zealots / N_g
    mean_deg   = degrees.mean()
    norm_deg   = mean_deg / N_g
    cv_deg     = degrees.std() / (mean_deg + 1e-8)
    clustering = nx.average_clustering(G)
    lambda_2, fiedler = compute_spectral_gap_and_fiedler(G)

    topo_ba = 1.0 if topo_type == 'ba' else 0.0
    topo_er = 1.0 if topo_type == 'er' else 0.0
    topo_ws = 1.0 if topo_type == 'ws' else 0.0

    # ── New placement features (9, 10, 11) ───────────────────
    zealot_list = list(zealot_set)

    # Feature 9: hub score — mean degree of zealots / mean degree of all
    zealot_degrees = degrees[zealot_list]
    hub_score = float(zealot_degrees.mean() / (mean_deg + 1e-8))

    # Feature 10: bridge score — mean betweenness of zealots / mean overall
    if betweenness is None:
        betweenness = nx.betweenness_centrality(G)
    btwn_vals       = np.array(list(betweenness.values()), dtype=np.float64)
    mean_btwn_all   = btwn_vals.mean()
    zealot_btwn     = np.array([betweenness[n] for n in zealot_list],
                                dtype=np.float64)
    bridge_score = float(zealot_btwn.mean() / (mean_btwn_all + 1e-10))

    # Feature 11: Fiedler projection score
    # How strongly do zealots span the Fiedler cut?
    # |mean |fiedler[zealots]|| — large = zealots straddle the bottleneck
    if len(fiedler) == N_g and len(zealot_list) > 0:
        fiedler_score = float(np.abs(fiedler[zealot_list]).mean())
    else:
        fiedler_score = 0.0

    return np.array([
        rho_z,
        lambda_2,
        norm_deg,
        cv_deg,
        clustering,
        topo_ba,
        topo_er,
        topo_ws,
        hub_score,       # 9
        bridge_score,    # 10
        fiedler_score,   # 11
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────

def make_ba(n, m=8, seed=None):
    return nx.barabasi_albert_graph(n, m, seed=seed), 'ba'

def make_er(n, avg_degree=16, seed=None):
    p = avg_degree / (n - 1)
    for attempt in range(20):
        G = nx.erdos_renyi_graph(n, p,
                seed=(seed + attempt) if seed is not None else None)
        if nx.is_connected(G):
            return G, 'er'
    return nx.erdos_renyi_graph(n, p + 0.01, seed=seed), 'er'

def make_ws(n, k=16, p=0.1, seed=None):
    return nx.watts_strogatz_graph(n, k, p, seed=seed), 'ws'

GRAPH_MAKERS = {
    'ba': lambda seed: make_ba(N, m=8,          seed=seed),
    'er': lambda seed: make_er(N, avg_degree=16, seed=seed),
    'ws': lambda seed: make_ws(N, k=16, p=0.1,  seed=seed),
}


# ─────────────────────────────────────────────────────────────
# Voter model simulation
# ─────────────────────────────────────────────────────────────

def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=MC_RUNS, seed=None):
    """
    Run mc_runs independent MC simulations and return mean magnetization
    trajectory m(0..T) as numpy array of shape (T,).
    zealot_set: set of zealot node indices.
    """
    rng       = np.random.default_rng(seed)
    N_g       = G.number_of_nodes()
    adj       = [list(G.neighbors(n)) for n in range(N_g)]
    is_zealot = np.zeros(N_g, dtype=bool)
    for z in zealot_set:
        is_zealot[z] = True
    non_zealots = np.where(~is_zealot)[0]

    all_trajs = []
    for _ in range(mc_runs):
        opinions = rng.choice([-1.0, 1.0], size=N_g)
        opinions[is_zealot] = 1.0
        traj = []
        for _ in range(T):
            traj.append(float(opinions.mean()))
            chosen = rng.choice(non_zealots, size=len(non_zealots), replace=True)
            for node in chosen:
                nbrs = adj[node]
                if nbrs:
                    opinions[node] = opinions[rng.integers(0, len(nbrs))]
            opinions[is_zealot] = 1.0
        all_trajs.append(traj)

    return np.mean(all_trajs, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Dataset — both placement strategies
# ─────────────────────────────────────────────────────────────

def build_dataset(zealot_list, num_graphs, T, mc_runs):
    """
    Each sample covers both hub and random zealot placement,
    doubling the dataset size relative to the original script.

    Returns:
      descriptors  : (M, 11)
      trajectories : (M, T)
    """
    print("\nBuilding placement-aware trajectory dataset...", flush=True)
    print(f"  Z={zealot_list} | topologies=ba,er,ws | "
          f"{num_graphs} graphs each | T={T} | mc_runs={mc_runs}", flush=True)
    print(f"  Placement strategies: hub + random  "
          f"(dataset size = 2x original)", flush=True)

    descriptors_list = []
    trajectory_list  = []
    t0    = time.time()
    total = len(zealot_list) * len(GRAPH_MAKERS) * num_graphs * 2
    count = 0

    for topo_name, make_fn in GRAPH_MAKERS.items():
        for num_zealots in zealot_list:
            print(f"  {topo_name}  Z={num_zealots}...", flush=True)
            for g_idx in range(num_graphs):
                seed = hash((topo_name, num_zealots, g_idx)) % (2**31)
                rng  = np.random.default_rng(seed)
                G, topo_type = make_fn(seed)

                # Betweenness is expensive — compute once per graph
                btwn = nx.betweenness_centrality(G)

                for strategy in ('hub', 'random'):
                    if strategy == 'hub':
                        zealot_set = place_zealots_hubs(G, num_zealots)
                    else:
                        zealot_set = place_zealots_random(G, num_zealots, rng)

                    desc = compute_descriptors(G, num_zealots, topo_type,
                                               zealot_set, betweenness=btwn)
                    traj = simulate_trajectory(G, zealot_set, T, mc_runs,
                                               seed=seed + 1)

                    descriptors_list.append(desc)
                    trajectory_list.append(traj)
                    count += 1

                if count % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"    {count}/{total} samples  "
                          f"({elapsed:.0f}s elapsed)", flush=True)

    # Shuffle
    idx = list(range(len(descriptors_list)))
    random.shuffle(idx)
    descriptors_list = [descriptors_list[i] for i in idx]
    trajectory_list  = [trajectory_list[i]  for i in idx]

    descriptors  = np.stack(descriptors_list)   # (M, 11)
    trajectories = np.stack(trajectory_list)    # (M, T)

    print(f"\nDataset ready: {len(descriptors)} samples in "
          f"{time.time()-t0:.1f}s", flush=True)
    print(f"  Descriptor shape: {descriptors.shape}", flush=True)
    print(f"  Trajectory shape: {trajectories.shape}", flush=True)

    return descriptors, trajectories


def normalize_descriptors(descriptors, stats=None):
    if stats is None:
        mean  = descriptors.mean(axis=0)
        std   = descriptors.std(axis=0) + 1e-8
        stats = (mean, std)
    mean, std = stats
    return (descriptors - mean) / std, stats


# ─────────────────────────────────────────────────────────────
# Model — same LSTM architecture, wider encoder input only
# ─────────────────────────────────────────────────────────────

class TrajectoryLSTM(nn.Module):
    """
    Identical to original SpectralLSTM except desc_dim=11.
    MLP encoder: 11 → 128 → 256 → 2LH
    """
    def __init__(self, desc_dim=DESC_DIM, hidden_dim=256, num_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T          = T

        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2)
        )

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, descriptors, teacher_forcing_ratio=0.0,
                target_traj=None):
        B   = descriptors.shape[0]
        enc = self.encoder(descriptors)
        enc = enc.view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=descriptors.device)

        predictions = []
        h, c = h0, c0
        for t in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            pred_t = self.output_head(out.squeeze(1))
            predictions.append(pred_t)
            if (teacher_forcing_ratio > 0.0 and
                    target_traj is not None and
                    random.random() < teacher_forcing_ratio):
                inp = target_traj[:, t].unsqueeze(1).unsqueeze(2)
            else:
                inp = pred_t.detach().unsqueeze(1)

        return torch.cat(predictions, dim=1)   # (B, T)


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def trajectory_loss(pred, target):
    T       = pred.shape[1]
    weights = torch.linspace(1.0, 2.0, T, device=pred.device)
    weights = weights / weights.sum()
    mse     = ((pred - target) ** 2 * weights.unsqueeze(0)).mean()
    return torch.sqrt(mse + 1e-8)


# ─────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, desc_t, traj_t, optimizer, device, batch_size, tf_ratio):
    model.train()
    M   = desc_t.shape[0]
    idx = torch.randperm(M)
    total, count = 0.0, 0

    for start in range(0, M, batch_size):
        batch_idx  = idx[start:start + batch_size]
        desc_batch = desc_t[batch_idx].to(device)
        traj_batch = traj_t[batch_idx].to(device)
        optimizer.zero_grad()
        pred = model(desc_batch,
                     teacher_forcing_ratio=tf_ratio,
                     target_traj=traj_batch)
        loss = trajectory_loss(pred, traj_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); count += 1

    return total / count


@torch.no_grad()
def evaluate_rmse(model, desc_t, traj_t, device, batch_size):
    model.eval()
    M       = desc_t.shape[0]
    sq, n   = 0.0, 0

    for start in range(0, M, batch_size):
        desc_batch = desc_t[start:start + batch_size].to(device)
        traj_batch = traj_t[start:start + batch_size].to(device)
        pred       = model(desc_batch, teacher_forcing_ratio=0.0)
        p = pred.cpu().numpy() * 2 - 1
        t = traj_batch.cpu().numpy() * 2 - 1
        sq += np.sum((p - t) ** 2)
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
    p.add_argument("--hidden_dim",   type=int,   required=True)
    p.add_argument("--num_layers",   type=int,   required=True)
    p.add_argument("--dropout",      type=float, required=True)
    p.add_argument("--num_workers",  type=int,   required=True)
    p.add_argument("--save_dir",     type=str,   default="saved_models")
    p.add_argument("--save_name",    type=str,   default="pa-lstm.pt")
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

    print(f"\nPlacement-Aware SpectralLSTM  (desc_dim={DESC_DIM})", flush=True)
    print(f"  New features: hub_score, bridge_score, fiedler_score", flush=True)
    print(f"  Placement strategies: hub + random", flush=True)
    print(f"Epochs={args.epochs} | Batch={args.batch_size} | "
          f"T={args.T} | mc_runs={args.mc_runs}", flush=True)

    # ── Dataset ───────────────────────────────────────────────
    descriptors, trajectories = build_dataset(
        ALL_Z, args.num_graphs, args.T, args.mc_runs)

    trajectories = (trajectories + 1) / 2.0   # [-1,1] → [0,1]
    descriptors, norm_stats = normalize_descriptors(descriptors)

    M     = len(descriptors)
    split = int(0.8 * M)
    desc_t = torch.tensor(descriptors[:split],  dtype=torch.float32)
    traj_t = torch.tensor(trajectories[:split], dtype=torch.float32)
    desc_v = torch.tensor(descriptors[split:],  dtype=torch.float32)
    traj_v = torch.tensor(trajectories[split:], dtype=torch.float32)
    print(f"Train: {split} | Val: {M - split}", flush=True)

    # ── Model ─────────────────────────────────────────────────
    model = TrajectoryLSTM(
        desc_dim=DESC_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        T=args.T,
        dropout=args.dropout
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"  {'Ep':>5}  {'Loss':>8}  {'RMSE':>8}  "
          f"{'Best':>8}  {'TF':>6}  {'LR':>10}  {'Time':>7}", flush=True)
    print(f"  {'-'*62}", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Teacher forcing: 1.0 → 0.0 over first half of training
        tf_ratio = max(0.0, 1.0 - 2.0 * epoch / args.epochs)

        t0   = time.time()
        loss = train_epoch(model, desc_t, traj_t, optimizer, device,
                           args.batch_size, tf_ratio)
        rmse = evaluate_rmse(model, desc_v, traj_v, device, args.batch_size)
        scheduler.step()

        if rmse < best_rmse:
            best_rmse  = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch <= 10 or epoch % 20 == 0:
            print(f"  {epoch:>5}  {loss:>8.4f}  {rmse:>8.4f}  "
                  f"{best_rmse:>8.4f}  {tf_ratio:>6.3f}  {lr_now:>10.2e}  "
                  f"{time.time()-t0:>6.1f}s", flush=True)

        if epoch % 50 == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "best_val_rmse":    best_rmse,
                "epoch":            epoch,
                "norm_stats":       norm_stats,
                "hyperparams": {
                    "desc_dim":    DESC_DIM,
                    "hidden_dim":  args.hidden_dim,
                    "num_layers":  args.num_layers,
                    "T":           args.T,
                    "trained_Z":   ALL_Z,
                    "topologies":  ["ba", "er", "ws"],
                    "model_type":  "PlacementAwareLSTM",
                    "placements":  ["hub", "random"],
                    "new_features": ["hub_score", "bridge_score",
                                     "fiedler_score"],
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
            "desc_dim":    DESC_DIM,
            "hidden_dim":  args.hidden_dim,
            "num_layers":  args.num_layers,
            "T":           args.T,
            "trained_Z":   ALL_Z,
            "topologies":  ["ba", "er", "ws"],
            "model_type":  "PlacementAwareLSTM",
            "placements":  ["hub", "random"],
            "new_features": ["hub_score", "bridge_score", "fiedler_score"],
        }
    }, ckpt_path)
    print(f"\n✓ Saved to {ckpt_path}", flush=True)
    print(f"  Best Val RMSE: {best_rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
