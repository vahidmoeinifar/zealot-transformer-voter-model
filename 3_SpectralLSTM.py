"""
Universal Magnetization Trajectory Predictor
=============================================

Input (8D conditioning vector per graph):
  1. rho_Z              — zealot density Z/N
  2. lambda_2           — spectral gap (algebraic connectivity)
  3. mean_degree / N    — normalised mean degree
  4. CV(degree)         — degree heterogeneity (std/mean)
  5. global_clustering  — Watts-Strogatz C
  6. topo_ba            — topology one-hot: Barabasi-Albert
  7. topo_er            — topology one-hot: Erdos-Renyi
  8. topo_ws            — topology one-hot: Watts-Strogatz

Output:
  m(t) for t = 0..T  — full magnetization trajectory

Architecture: LSTM over time steps, conditioned on graph descriptors.
Loss: RMSE on trajectory (regression, not classification).

Trained on BA, ER, WS graphs × all Z in {2,8,16,32}.

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

ALL_Z    = [2, 8, 16, 32]
N        = 1024
T_STEPS  = 50
MC_RUNS  = 20


# ─────────────────────────────────────────────────────────────
# Graph Descriptors
# ─────────────────────────────────────────────────────────────

def compute_spectral_gap(G):
    try:
        L = nx.laplacian_matrix(G).astype(float)
        # Get 2 smallest eigenvalues
        vals = eigsh(L, k=2, which='SM', return_eigenvectors=False,
                     tol=1e-3, maxiter=1000)
        lambda_2 = float(np.sort(vals)[1])
        return max(lambda_2, 0.0)
    except Exception:
        return 0.0


def compute_descriptors(G, num_zealots, topo_type):

    N_g     = G.number_of_nodes()
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)

    rho_z      = num_zealots / N_g
    mean_deg   = degrees.mean()
    norm_deg   = mean_deg / N_g
    cv_deg     = degrees.std() / (mean_deg + 1e-8)   # degree heterogeneity
    clustering = nx.average_clustering(G)
    lambda_2   = compute_spectral_gap(G)

    topo_ba = 1.0 if topo_type == 'ba' else 0.0
    topo_er = 1.0 if topo_type == 'er' else 0.0
    topo_ws = 1.0 if topo_type == 'ws' else 0.0

    return np.array([
        rho_z,
        lambda_2,
        norm_deg,
        cv_deg,
        clustering,
        topo_ba,
        topo_er,
        topo_ws,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Graph Generators
# ─────────────────────────────────────────────────────────────

def make_ba(N, m=8, seed=None):
    return nx.barabasi_albert_graph(N, m, seed=seed), 'ba'

def make_er(N, avg_degree=16, seed=None):
    p = avg_degree / (N - 1)
    for attempt in range(20):
        G = nx.erdos_renyi_graph(N, p,
                seed=(seed + attempt) if seed is not None else None)
        if nx.is_connected(G):
            return G, 'er'
    return nx.erdos_renyi_graph(N, p + 0.01, seed=seed), 'er'

def make_ws(N, k=16, p=0.1, seed=None):
    return nx.watts_strogatz_graph(N, k, p, seed=seed), 'ws'

GRAPH_MAKERS = {
    'ba': lambda seed: make_ba(N, m=8,         seed=seed),
    'er': lambda seed: make_er(N, avg_degree=16, seed=seed),
    'ws': lambda seed: make_ws(N, k=16, p=0.1,  seed=seed),
}


# ─────────────────────────────────────────────────────────────
# Voter Model Simulation → trajectory
# ─────────────────────────────────────────────────────────────

def place_zealots_hubs(G, num_zealots):
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return set(n for n, _ in sorted_nodes[:num_zealots])

def simulate_trajectory(G, num_zealots, T=50, mc_runs=20, seed=None):
    """
    Run mc_runs independent MC simulations and return mean magnetization
    trajectory m(0..T) as numpy array of shape (T,).
    """
    rng = np.random.default_rng(seed)
    N_g = G.number_of_nodes()
    adj = [list(G.neighbors(n)) for n in range(N_g)]

    zealot_set  = place_zealots_hubs(G, num_zealots)
    is_zealot   = np.zeros(N_g, dtype=bool)
    for z in zealot_set:
        is_zealot[z] = True
    non_zealots = np.where(~is_zealot)[0]

    all_trajs = []
    for run in range(mc_runs):
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

    return np.mean(all_trajs, axis=0).astype(np.float32)  # (T,)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

def build_dataset(zealot_list, num_graphs, T, mc_runs):

    print("\nBuilding trajectory dataset...", flush=True)
    print(f"  Z={zealot_list} | topologies=ba,er,ws | "
          f"{num_graphs} graphs each | T={T} | mc_runs={mc_runs}", flush=True)

    descriptors_list = []
    trajectory_list  = []
    t0 = time.time()
    total = len(zealot_list) * len(GRAPH_MAKERS) * num_graphs
    count = 0

    for topo_name, make_fn in GRAPH_MAKERS.items():
        for num_zealots in zealot_list:
            print(f"  {topo_name} Z={num_zealots}...", flush=True)
            for g_idx in range(num_graphs):
                seed = hash((topo_name, num_zealots, g_idx)) % (2**31)
                G, topo_type = make_fn(seed)

                desc = compute_descriptors(G, num_zealots, topo_type)
                traj = simulate_trajectory(G, num_zealots, T, mc_runs,
                                           seed=seed + 1)

                descriptors_list.append(desc)
                trajectory_list.append(traj)
                count += 1

    # Shuffle
    idx = list(range(len(descriptors_list)))
    random.shuffle(idx)
    descriptors_list = [descriptors_list[i] for i in idx]
    trajectory_list  = [trajectory_list[i]  for i in idx]

    descriptors = np.stack(descriptors_list)   # (M, 8)
    trajectories = np.stack(trajectory_list)   # (M, T)

    print(f"\nDataset ready: {len(descriptors)} samples in "
          f"{time.time()-t0:.1f}s", flush=True)
    print(f"  Descriptor shape: {descriptors.shape}", flush=True)
    print(f"  Trajectory shape: {trajectories.shape}", flush=True)

    return descriptors, trajectories


def normalize_descriptors(descriptors, stats=None):
    if stats is None:
        mean = descriptors.mean(axis=0)
        std  = descriptors.std(axis=0) + 1e-8
        stats = (mean, std)
    mean, std = stats
    return (descriptors - mean) / std, stats


# ─────────────────────────────────────────────────────────────
# Model: LSTM trajectory predictor
# ─────────────────────────────────────────────────────────────
class TrajectoryLSTM(nn.Module):

    def __init__(self, desc_dim=8, hidden_dim=128, num_layers=2,
                 T=50, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T          = T

        # Encode graph descriptor into LSTM initial state
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2)  # h0 and c0
        )

        # LSTM: input is previous m(t-1), output predicts m(t)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()   # magnetization in [0, 1] (normalized from [-1,1])
        )

    def forward(self, descriptors, teacher_forcing_ratio=0.0,
                target_traj=None):

        B = descriptors.shape[0]

        # Encode descriptor -> initial LSTM state
        enc = self.encoder(descriptors)  # (B, num_layers * hidden * 2)
        enc = enc.view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()

        # Initial input: m(t=-1) = 0.5 (neutral, [0,1] scale)
        inp = torch.full((B, 1, 1), 0.5, device=descriptors.device)

        predictions = []
        h, c = h0, c0

        for t in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))   # out: (B, 1, hidden)
            pred_t = self.output_head(out.squeeze(1))  # (B, 1)
            predictions.append(pred_t)

            # Next input: teacher forcing or model prediction
            if (teacher_forcing_ratio > 0.0 and
                    target_traj is not None and
                    random.random() < teacher_forcing_ratio):
                inp = target_traj[:, t].unsqueeze(1).unsqueeze(2)
            else:
                inp = pred_t.detach().unsqueeze(1)

        pred_traj = torch.cat(predictions, dim=1)  # (B, T)
        return pred_traj


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def trajectory_loss(pred, target):
    T = pred.shape[1]
    # Weight later steps more heavily to penalize drift
    weights = torch.linspace(1.0, 2.0, T, device=pred.device)
    weights = weights / weights.sum()
    mse = ((pred - target) ** 2 * weights.unsqueeze(0)).mean()
    return torch.sqrt(mse + 1e-8)


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, desc_t, traj_t, optimizer, device,
                batch_size, tf_ratio):
    model.train()
    M = desc_t.shape[0]
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
    M  = desc_t.shape[0]
    sq, n = 0.0, 0

    for start in range(0, M, batch_size):
        desc_batch = desc_t[start:start+batch_size].to(device)
        traj_batch = traj_t[start:start+batch_size].to(device)
        pred = model(desc_batch, teacher_forcing_ratio=0.0)

        # Convert [0,1] back to [-1,1] for RMSE comparison
        p = (pred.cpu().numpy() * 2 - 1)
        t = (traj_batch.cpu().numpy() * 2 - 1)
        sq += np.sum((p - t) ** 2)
        n  += p.size

    return float(np.sqrt(sq / n))


# ─────────────────────────────────────────────────────────────
# Main
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
    p.add_argument("--save_name",    type=str,   default="SpectralLSTM.pt")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"Epochs={args.epochs} | Batch={args.batch_size} | "
          f"T={args.T} | mc_runs={args.mc_runs}", flush=True)

    # Build dataset
    descriptors, trajectories = build_dataset(
        ALL_Z, args.num_graphs, args.T, args.mc_runs)

    # Normalize trajectories to [0,1] (from [-1,1])
    trajectories = (trajectories + 1) / 2.0

    # Normalize descriptors
    descriptors, norm_stats = normalize_descriptors(descriptors)

    # Train/val split
    M     = len(descriptors)
    split = int(0.8 * M)
    desc_t = torch.tensor(descriptors[:split],  dtype=torch.float32)
    traj_t = torch.tensor(trajectories[:split], dtype=torch.float32)
    desc_v = torch.tensor(descriptors[split:],  dtype=torch.float32)
    traj_v = torch.tensor(trajectories[split:], dtype=torch.float32)

    print(f"Train: {split} | Val: {M-split}", flush=True)

    # Model
    model  = TrajectoryLSTM(
        desc_dim=8, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, T=args.T,
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
    print(f"  {'-'*60}", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Teacher forcing: 1.0 -> 0.0 over first half of training
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
                    "desc_dim":   8,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "T":          args.T,
                    "trained_Z":  ALL_Z,
                    "topologies": ["ba", "er", "ws"],
                    "model_type": "TrajectoryLSTM"
                }
            }, os.path.join(args.save_dir, args.save_name))
            print(f"  [checkpoint saved at epoch {epoch}]", flush=True)

    # Final save
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    torch.save({
        "model_state_dict": best_state,
        "best_val_rmse":    best_rmse,
        "norm_stats":       norm_stats,
        "args":             vars(args),
        "hyperparams": {
            "desc_dim":   8,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "T":          args.T,
            "trained_Z":  ALL_Z,
            "topologies": ["ba", "er", "ws"],
            "model_type": "TrajectoryLSTM"
        }
    }, ckpt_path)
    print(f"\n✓ Saved to {ckpt_path}", flush=True)
    print(f"  Best Val RMSE: {best_rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()