"""
train_baselines.py  —  Train and save baseline models
======================================================

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

os.makedirs("saved_models", exist_ok=True)
os.makedirs("result",       exist_ok=True)

# ── Must match compare script exactly ─────────────
T_STEPS    = 50
N          = 1024
M_PARAM    = 8
ALL_Z      = [2, 8, 16, 32]
TOPOLOGIES = ['ba', 'er', 'ws']
MLP_DESC_DIM = 8


# ═════════════════════════════════════════════════════════════
# Graph helpers  (identical to main script)
# ═════════════════════════════════════════════════════════════

def build_graph(topo, n, m, seed):
    if topo == 'ba':
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == 'er':
        G = nx.erdos_renyi_graph(n, min(2*m/(n-1), 1.0), seed=seed)
    elif topo == 'ws':
        G = nx.watts_strogatz_graph(n, max(2, 2*m), p=0.1, seed=seed)
    else:
        raise ValueError(topo)
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G


def place_hubs(G, Z):
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:Z]]


def compute_spectral_gap(G):
    try:
        L    = nx.laplacian_matrix(G).astype(float)
        vals = eigsh(L, k=2, which='SM', return_eigenvectors=False,
                     tol=1e-3, maxiter=1000)
        return max(float(np.sort(vals)[1]), 0.0)
    except Exception:
        return 0.0


def compute_spectral_descriptor(G, num_zealots, topo_type):
    """8-D descriptor — identical to compare_models_extended.py."""
    N_g     = G.number_of_nodes()
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    rho_z   = num_zealots / N_g
    mean_deg = degrees.mean()
    return np.array([
        rho_z,
        compute_spectral_gap(G),
        mean_deg / N_g,
        degrees.std() / (mean_deg + 1e-8),
        nx.average_clustering(G),
        1.0 if topo_type == 'ba' else 0.0,
        1.0 if topo_type == 'er' else 0.0,
        1.0 if topo_type == 'ws' else 0.0,
    ], dtype=np.float32)


def fast_mc_mean_traj(G, zealot_nodes, T, mc_runs, seed):
    """MC trajectory — identical to compare_models_extended.py."""
    rng         = np.random.default_rng(seed)
    N_g         = G.number_of_nodes()
    adj         = [list(G.neighbors(i)) for i in range(N_g)]
    is_zealot   = np.zeros(N_g, dtype=bool)
    for z in zealot_nodes:
        is_zealot[z] = True
    non_zealots = np.where(~is_zealot)[0]
    all_t = []
    for _ in range(mc_runs):
        ops = rng.choice([-1.0, 1.0], size=N_g)
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
        all_t.append(traj)
    return np.mean(all_t, axis=0).astype(np.float32)


# ═════════════════════════════════════════════════════════════
# Dataset builder
# ═════════════════════════════════════════════════════════════

def build_training_data(n_graphs, mc_runs):
    print(f"\nBuilding training dataset  "
          f"({len(TOPOLOGIES)} topos × {len(ALL_Z)} Z × {n_graphs} graphs "
          f"× {mc_runs} MC runs)...", flush=True)
    descs, trajs = [], []
    total = len(TOPOLOGIES) * len(ALL_Z) * n_graphs
    done  = 0
    for topo in TOPOLOGIES:
        for Z in ALL_Z:
            for g_idx in range(n_graphs):
                seed = hash((topo, Z, g_idx)) % (2**31)
                G    = build_graph(topo, N, M_PARAM, seed)
                zn   = place_hubs(G, Z)
                desc = compute_spectral_descriptor(G, Z, topo)
                traj = fast_mc_mean_traj(G, zn, T_STEPS, mc_runs, seed + 1)
                descs.append(desc)
                trajs.append(traj)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total}", flush=True)

    desc_arr = np.stack(descs)   # (n_samples, 8)
    traj_arr = np.stack(trajs)   # (n_samples, T)
    return desc_arr, traj_arr


def compute_norm_stats(desc_arr):
    mean = desc_arr.mean(axis=0).astype(np.float32)
    std  = (desc_arr.std(axis=0)  + 1e-8).astype(np.float32)
    return mean, std


# ═════════════════════════════════════════════════════════════
# Model A — MLP-Descriptor
# ═════════════════════════════════════════════════════════════

class MLPDescriptor(nn.Module):
    def __init__(self, desc_dim=8, T=50, hidden=256):
        super().__init__()
        self.T   = T
        self.net = nn.Sequential(
            nn.Linear(desc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   T),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(desc_arr, traj_arr, norm_stats, device, epochs, lr=1e-3):
    print(f"\n[MLP-Descriptor] Training for {epochs} epochs ...", flush=True)
    mean, std = norm_stats
    desc_n = (desc_arr - mean) / std
    traj_n = (traj_arr + 1.0) / 2.0          # spin → [0,1]

    desc_t = torch.tensor(desc_n, dtype=torch.float32).to(device)
    traj_t = torch.tensor(traj_n, dtype=torch.float32).to(device)

    model = MLPDescriptor(desc_dim=MLP_DESC_DIM, T=T_STEPS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_loss, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(desc_t)
        loss = torch.sqrt(((pred - traj_t)**2).mean())
        loss.backward()
        opt.step()
        sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sd   = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 50 == 0:
            print(f"  ep {ep:>4}/{epochs}  loss={loss.item():.5f}", flush=True)

    model.load_state_dict(best_sd)
    print(f"  Best training loss: {best_loss:.5f}")
    return model.eval()


def save_mlp(model, norm_stats, path):
    mean, std = norm_stats
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": {
            "desc_dim": MLP_DESC_DIM,
            "T":        T_STEPS,
            "hidden":   256,
        },
        # Store as plain lists so JSON-serialisable if needed
        "norm_stats": (mean.tolist(), std.tolist()),
        "training_config": {
            "N": N, "M_param": M_PARAM,
            "topologies": TOPOLOGIES,
            "Z_values":   ALL_Z,
            "placement":  "hubs",
        },
    }, path)
    print(f"  Saved MLP-Descriptor → {path}")


def load_mlp(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    hp    = ckpt["hyperparams"]
    model = MLPDescriptor(desc_dim=hp["desc_dim"], T=hp["T"], hidden=hp["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    mean, std = ckpt["norm_stats"]
    norm_stats = (np.array(mean, dtype=np.float32),
                  np.array(std,  dtype=np.float32))
    print(f"  Loaded MLP-Descriptor ← {path}")
    return model.to(device).eval(), norm_stats


# ═════════════════════════════════════════════════════════════
# Model B — Mean-Field ODE
# ═════════════════════════════════════════════════════════════

def meanfield_ode(m, t, alpha, beta, rho_z):
    return -alpha * m + beta * rho_z * (1.0 - m)


def fit_meanfield(desc_arr, traj_arr):
    print("\n[Mean-Field ODE] Fitting parameters ...", flush=True)
    t_arr = np.arange(T_STEPS, dtype=np.float64)
    rho_z_vals = desc_arr[:, 0]          # first descriptor feature = rho_z

    def residuals(params):
        alpha = max(params[0], 1e-6)
        beta  = max(params[1], 1e-6)
        errs  = []
        for rho_z, traj in zip(rho_z_vals, traj_arr):
            m0   = float(traj[0])
            pred = odeint(meanfield_ode, m0, t_arr,
                          args=(alpha, beta, float(rho_z)))[:, 0]
            errs.append(np.mean((pred - traj)**2))
        return float(np.mean(errs))

    result = minimize(residuals, x0=[0.05, 0.5],
                      method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    alpha = max(result.x[0], 1e-6)
    beta  = max(result.x[1], 1e-6)
    print(f"  alpha={alpha:.6f}  beta={beta:.6f}  residual={result.fun:.6f}")
    return float(alpha), float(beta)


def save_mf(alpha, beta, norm_stats, path):
    mean, std = norm_stats
    payload = {
        "alpha": alpha,
        "beta":  beta,
        "norm_stats": {
            "mean": mean.tolist(),
            "std":  std.tolist(),
        },
        "training_config": {
            "N": N, "M_param": M_PARAM,
            "topologies": TOPOLOGIES,
            "Z_values":   ALL_Z,
            "placement":  "hubs",
            "T":          T_STEPS,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved Mean-Field ODE  → {path}")


def load_mf(path):
    with open(path) as f:
        d = json.load(f)
    norm_stats = (
        np.array(d["norm_stats"]["mean"], dtype=np.float32),
        np.array(d["norm_stats"]["std"],  dtype=np.float32),
    )
    print(f"  Loaded Mean-Field ODE ← {path}  "
          f"(alpha={d['alpha']:.6f}  beta={d['beta']:.6f})")
    return float(d["alpha"]), float(d["beta"]), norm_stats


# ═════════════════════════════════════════════════════════════
# Norm-stats alignment with LSTM checkpoint
# ═════════════════════════════════════════════════════════════

def try_load_lstm_norm_stats(lstm_path, device):
    """
    If universal_lstm.pt exists, extract its norm_stats so all baselines
    share the exact same normalisation as the LSTM.
    Returns (mean, std) or None.
    """
    if not os.path.exists(lstm_path):
        return None
    try:
        ckpt = torch.load(lstm_path, map_location=device, weights_only=False)
        ns   = ckpt.get("norm_stats")
        if ns is not None:
            mean = np.array(ns[0], dtype=np.float32)
            std  = np.array(ns[1], dtype=np.float32)
            print(f"  Using norm_stats from {lstm_path}")
            return mean, std
    except Exception as e:
        print(f"  WARNING: could not load norm_stats from {lstm_path}: {e}")
    return None


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train and save MLP-Descriptor and Mean-Field ODE baselines")
    parser.add_argument("--n-graphs", type=int, default=50,
                        help="Graphs per (topo, Z) cell (default: 50)")
    parser.add_argument("--mc-runs",  type=int, default=20,
                        help="MC runs per graph (default: 20)")
    parser.add_argument("--epochs",   type=int, default=300,
                        help="MLP training epochs (default: 300)")
    parser.add_argument("--lr",       type=float, default=1e-3,
                        help="MLP learning rate (default: 1e-3)")
    parser.add_argument("--lstm-path", default="saved_models/universal_lstm.pt",
                        help="Path to LSTM checkpoint for norm_stats alignment")
    parser.add_argument("--out-dir",  default="saved_models",
                        help="Output directory (default: saved_models)")
    parser.add_argument("--force",    action="store_true",
                        help="Retrain even if checkpoints already exist")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_mlp = os.path.join(args.out_dir, "baseline_mlp.pt")
    out_mf  = os.path.join(args.out_dir, "baseline_mf.json")

    # Check if already trained
    mlp_exists = os.path.exists(out_mlp)
    mf_exists  = os.path.exists(out_mf)
    if mlp_exists and mf_exists and not args.force:
        print("Both baseline files already exist:")
        print(f"  {out_mlp}")
        print(f"  {out_mf}")
        print("Use --force to retrain.")
        return

    print(f"Device : {device}")
    print(f"Config : n_graphs={args.n_graphs}  mc_runs={args.mc_runs}  "
          f"epochs={args.epochs}  lr={args.lr}")

    # ── Build dataset ─────────────────────────────────────────
    desc_arr, traj_arr = build_training_data(args.n_graphs, args.mc_runs)

    # ── Norm stats — prefer LSTM's if available ───────────────
    lstm_norm = try_load_lstm_norm_stats(args.lstm_path, device)
    if lstm_norm is not None:
        norm_stats = lstm_norm
        print("  norm_stats: aligned with LSTM checkpoint")
    else:
        norm_stats = compute_norm_stats(desc_arr)
        print("  norm_stats: computed from this training set")

    # ── Train / fit ───────────────────────────────────────────
    if not mlp_exists or args.force:
        mlp_model = train_mlp(desc_arr, traj_arr, norm_stats, device,
                              epochs=args.epochs, lr=args.lr)
        save_mlp(mlp_model, norm_stats, out_mlp)
    else:
        print(f"\n[MLP-Descriptor] Already exists, skipping ({out_mlp})")

    if not mf_exists or args.force:
        alpha, beta = fit_meanfield(desc_arr, traj_arr)
        save_mf(alpha, beta, norm_stats, out_mf)
    else:
        print(f"\n[Mean-Field ODE] Already exists, skipping ({out_mf})")

    print("\n" + "="*55)
    print("Baselines saved. In compare_models_extended.py, replace")
    print("the build_baseline_training_data + train_mlp_baseline +")
    print("fit_meanfield calls with:")
    print()
    print('  mlp_model, mlp_norm = load_mlp("saved_models/baseline_mlp.pt", device)')
    print('  mf_alpha, mf_beta, _ = load_mf("saved_models/baseline_mf.json")')
    print()
    print("  baseline_params = {")
    print('      "mlp": (mlp_model, mlp_norm),')
    print('      "mf":  (mf_alpha,  mf_beta),')
    print("  }")
    print("="*55)


if __name__ == "__main__":
    main()