"""
ZealotTransformer — Enhanced Training with Multi-Metric Validation
==================================================================
Saves checkpoints every N epochs for manual convergence checking.
Reports validation RMSE on:
  - BA / hub placement (training distribution)
  - BA / random placement
  - ER / hub placement
  - WS / hub placement


"""

import os, random, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool, cpu_count
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
torch.set_num_threads(min(4, NUM_CPUS))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ALL_Z        = [2, 8, 16, 32]
TRAIN_N_LIST = [256, 512, 1024, 2048]
TOPOLOGIES   = ['ba', 'er', 'ws']
T_STEPS      = 50
MC_RUNS      = 20
NODE_FEAT_DIM = 5

# Validation parameters
VAL_RUNS = 20          # Number of MC runs for validation
VAL_GRAPHS = 10        # Number of graphs per validation configuration
CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
EARLY_STOPPING_PATIENCE = 30  # Stop if no improvement for N epochs


# ═════════════════════════════════════════════════════════════
# DATA GENERATION FUNCTIONS (Same as before)
# ═════════════════════════════════════════════════════════════

def compute_fiedler_vector(G):
    """Returns Fiedler vector (N,) — 2nd smallest Laplacian eigenvector."""
    n = G.number_of_nodes()
    if n > 500:
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        fiedler = degrees / (degrees.max() + 1e-8)
        return fiedler.astype(np.float32)
    try:
        L = nx.laplacian_matrix(G).astype(float)
        nev = min(3, n - 1)
        vals, vecs = eigsh(L, k=nev, which='SM', tol=1e-2, maxiter=1000)
        order = np.argsort(vals)
        fiedler = vecs[:, order[1]]
        mx = np.abs(fiedler).max()
        if mx > 1e-8:
            fiedler /= mx
        return fiedler.astype(np.float32)
    except Exception:
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        fiedler = degrees / (degrees.max() + 1e-8)
        return fiedler.astype(np.float32)


def compute_node_features(G, zealot_set):
    N_g = G.number_of_nodes()
    degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
    deg_norm = degrees / (degrees.max() + 1e-8)
    z_i = np.zeros(N_g, dtype=np.float32)
    for node in zealot_set:
        z_i[node] = 1.0
    fiedler = compute_fiedler_vector(G)
    try:
        if N_g > 1000:
            pr_arr = deg_norm.copy()
        else:
            pr = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
            pr_arr = np.array([pr[i] for i in range(N_g)], dtype=np.float32)
            pr_arr /= (pr_arr.max() + 1e-8)
    except Exception:
        pr_arr = deg_norm.copy()
    try:
        clust_dict = nx.clustering(G)
        clust = np.array([clust_dict[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        clust = np.zeros(N_g, dtype=np.float32)
    X = np.stack([z_i, deg_norm, fiedler, pr_arr, clust], axis=1)
    return X.astype(np.float32)


def make_graph(topo, n, m=8, seed=None):
    if topo == 'ba':
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == 'er':
        p = min(2 * m / (n - 1), 1.0)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(n, p, seed=(seed + attempt if seed is not None else None))
            if nx.is_connected(G):
                break
    elif topo == 'ws':
        G = nx.watts_strogatz_graph(n, max(4, 2 * m), p=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topo}")
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G


def place_hubs(G, Z):
    return set(n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:Z])


def place_random(G, Z, rng):
    return set(int(n) for n in rng.choice(list(G.nodes()), size=Z, replace=False))


def place_bridges(G, Z):
    """Place zealots on nodes with highest betweenness centrality."""
    btwn = nx.betweenness_centrality(G, normalized=True)
    return set(sorted(btwn, key=btwn.get, reverse=True)[:Z])


def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=MC_RUNS, seed=None):
    rng = np.random.default_rng(seed)
    N_g = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N_g)]
    is_zealot = np.zeros(N_g, dtype=bool)
    for z in zealot_set:
        is_zealot[z] = True
    non_zealots = np.where(~is_zealot)[0]
    all_trajs = np.zeros((mc_runs, T), dtype=np.float32)
    for run in range(mc_runs):
        ops = rng.choice([-1.0, 1.0], size=N_g).astype(np.float32)
        ops[is_zealot] = 1.0
        for t in range(T):
            all_trajs[run, t] = float(ops.mean())
            chosen = rng.choice(non_zealots, size=len(non_zealots), replace=True)
            for node in chosen:
                nbrs = adj[node]
                if nbrs:
                    ops[node] = ops[nbrs[rng.integers(0, len(nbrs))]]
            ops[is_zealot] = 1.0
    return np.mean(all_trajs, axis=0).astype(np.float32)


def _generate_single_sample(args):
    topo, n, Z, g_idx, strategy, T, mc_runs = args
    m = max(4, int(8 * n / 1024))
    seed = hash((topo, n, Z, g_idx)) % (2**31)
    rng = np.random.default_rng(seed)
    try:
        G = make_graph(topo, n, m=m, seed=seed)
    except Exception:
        return None
    if strategy == 'hub':
        zealot_set = place_hubs(G, Z)
    elif strategy == 'random':
        zealot_set = place_random(G, Z, rng)
    elif strategy == 'bridges':
        zealot_set = place_bridges(G, Z)
    else:
        zealot_set = place_hubs(G, Z)
    X = compute_node_features(G, zealot_set)
    traj = simulate_trajectory(G, zealot_set, T, mc_runs, seed=seed + 1)
    z_mask = np.zeros(n, dtype=bool)
    for nd in zealot_set:
        z_mask[nd] = True
    return {
        "X": X,
        "z_mask": z_mask,
        "traj": traj,
        "N": n,
        "Z": Z,
        "topo": topo,
        "placement": strategy,
    }


def build_dataset(zealot_list, num_graphs_per_cell, T, mc_runs):
    print("\nBuilding ZealotTransformer dataset...", flush=True)
    tasks = []
    for topo in TOPOLOGIES:
        for n in TRAIN_N_LIST:
            for Z in zealot_list:
                for g_idx in range(num_graphs_per_cell):
                    for strategy in ('hub', 'random'):
                        tasks.append((topo, n, Z, g_idx, strategy, T, mc_runs))
    total_tasks = len(tasks)
    n_workers = max(1, NUM_CPUS - 1)
    samples = []
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_generate_single_sample, tasks)):
            if result is not None:
                samples.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total_tasks - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{total_tasks} tasks ({elapsed:.0f}s elapsed, ETA: {eta:.0f}s) {len(samples)} valid", flush=True)
    random.shuffle(samples)
    print(f"\nDataset ready: {len(samples)} samples in {time.time()-t0:.1f}s", flush=True)
    return samples


# ═════════════════════════════════════════════════════════════
# DATASET CLASS AND COLLATE FUNCTION
# ═════════════════════════════════════════════════════════════

class GraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_graphs(batch):
    max_nodes = max(s['N'] for s in batch)
    B = len(batch)
    X_batch = torch.zeros(B, max_nodes, NODE_FEAT_DIM)
    z_mask_batch = torch.zeros(B, max_nodes, dtype=torch.bool)
    padding_mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    traj_batch = torch.zeros(B, T_STEPS)
    for i, s in enumerate(batch):
        N = s['N']
        X_batch[i, :N] = torch.from_numpy(s['X'])
        z_mask_batch[i, :N] = torch.from_numpy(s['z_mask'])
        padding_mask[i, :N] = True
        traj_batch[i] = torch.from_numpy((s['traj'] + 1) / 2.0)
    return {
        'X': X_batch,
        'z_mask': z_mask_batch,
        'padding_mask': padding_mask,
        'traj': traj_batch,
        'batch_size': B
    }


# ═════════════════════════════════════════════════════════════
# MODEL (Same as before)
# ═════════════════════════════════════════════════════════════

class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128,
                 nhead=4, num_transformer_layers=3,
                 lstm_hidden=256, lstm_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.T = T
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers,
            enable_nested_tensor=False)
        ctx_dim = 2 * d_model
        self.ctx_projector = nn.Sequential(
            nn.Linear(ctx_dim, lstm_hidden * 2),
            nn.GELU(),
            nn.Linear(lstm_hidden * 2, lstm_layers * lstm_hidden * 2)
        )
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_graph_batch(self, X_batch, z_mask_batch, padding_mask):
        H = self.node_encoder(X_batch)
        H = self.transformer(H, src_key_padding_mask=~padding_mask)
        zealot_mask = z_mask_batch & padding_mask
        H_zealot = H * zealot_mask.unsqueeze(-1)
        zealot_sum = H_zealot.sum(dim=1)
        zealot_count = zealot_mask.sum(dim=1, keepdim=True).clamp(min=1)
        zealot_pool = zealot_sum / zealot_count
        non_zealot_mask = (~z_mask_batch) & padding_mask
        H_non_zealot = H * non_zealot_mask.unsqueeze(-1)
        non_zealot_sum = H_non_zealot.sum(dim=1)
        non_zealot_count = non_zealot_mask.sum(dim=1, keepdim=True).clamp(min=1)
        non_zealot_pool = non_zealot_sum / non_zealot_count
        context = torch.cat([zealot_pool, non_zealot_pool], dim=-1)
        return context

    def decode_batch(self, context):
        B = context.shape[0]
        proj = self.ctx_projector(context)
        proj = proj.view(B, self.lstm_layers, self.lstm_hidden * 2)
        h0 = proj[:, :, :self.lstm_hidden].transpose(0, 1).contiguous()
        c0 = proj[:, :, self.lstm_hidden:].transpose(0, 1).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=context.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            pred_t = self.output_head(out)
            preds.append(pred_t)
            inp = pred_t.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, z_mask):
        context = self.encode_graph_batch(
            X.unsqueeze(0),
            z_mask.unsqueeze(0),
            torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        )
        traj = self.decode_batch(context)
        return traj.squeeze(0)


# ═════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS (NEW)
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_on_configuration(model, config, device):
    """
    Evaluate model on a specific configuration.
    config: dict with keys: topo, placement, Z, n
    """
    topo = config['topo']
    placement = config['placement']
    Z = config['Z']
    n = config['n']
    
    preds = []
    trues = []
    
    for run in range(VAL_RUNS):
        seed = run + 10000
        rng = np.random.default_rng(seed)
        
        try:
            G = make_graph(topo, n, m=8, seed=int(rng.integers(0, 99999)))
            
            if placement == 'hub':
                zealots = list(place_hubs(G, Z))
            elif placement == 'random':
                zealots = list(place_random(G, Z, rng))
            elif placement == 'bridges':
                zealots = list(place_bridges(G, Z))
            else:
                zealots = list(place_hubs(G, Z))
            
            gt = simulate_trajectory(G, set(zealots), T=T_STEPS, mc_runs=5, seed=seed+1)
            pred = rollout_single(model, G, zealots, device)
            
            trues.append(gt.tolist())
            preds.append(pred.tolist())
        except Exception as e:
            continue
    
    if not preds:
        return float('nan'), float('nan')
    
    per_run_rmse = [float(np.sqrt(np.mean((np.array(p) - np.array(t))**2))) 
                    for p, t in zip(preds, trues)]
    return float(np.mean(per_run_rmse)), float(np.std(per_run_rmse))


@torch.no_grad()
def rollout_single(model, G, zealot_nodes, device):
    """Single rollout for evaluation."""
    zealot_set = set(zealot_nodes)
    X = compute_node_features(G, zealot_set)
    z_mask = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zealot_set:
        z_mask[nd] = True
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    z_t = torch.tensor(z_mask, dtype=torch.bool).to(device)
    pred = model(X_t, z_t).cpu().numpy()
    return pred * 2 - 1  # → [-1, 1]


def validate_model(model, device, epoch, current_best, patience_counter):
    """
    Run comprehensive validation on multiple configurations.
    Returns updated best RMSE and patience counter.
    """
    val_configs = [
        # Training distribution (BA, hub placement)
        {'topo': 'ba', 'placement': 'hub', 'Z': Z, 'n': 1024} for Z in ALL_Z
    ] + [
        # BA, random placement (generalization test)
        {'topo': 'ba', 'placement': 'random', 'Z': Z, 'n': 1024} for Z in ALL_Z
    ] + [
        # ER, hub placement (cross-topology)
        {'topo': 'er', 'placement': 'hub', 'Z': Z, 'n': 1024} for Z in ALL_Z
    ] + [
        # WS, hub placement (cross-topology)
        {'topo': 'ws', 'placement': 'hub', 'Z': Z, 'n': 1024} for Z in ALL_Z
    ]
    
    results = {}
    total_rmse = 0.0
    n_valid = 0
    
    print(f"\n  Validating at epoch {epoch}...", flush=True)
    
    for config in val_configs:
        mean_rmse, std_rmse = evaluate_on_configuration(model, config, device)
        if not np.isnan(mean_rmse):
            key = f"{config['topo']}_{config['placement']}_Z{config['Z']}"
            results[key] = {'mean': mean_rmse, 'std': std_rmse}
            total_rmse += mean_rmse
            n_valid += 1
    
    avg_rmse = total_rmse / n_valid if n_valid > 0 else float('inf')
    
    # Print summary table
    print(f"\n  {'─'*70}")
    print(f"  {'Config':<35} {'RMSE':<12} {'Std':<10}")
    print(f"  {'─'*70}")
    for key, vals in sorted(results.items()):
        print(f"  {key:<35} {vals['mean']:>8.4f}    {vals['std']:>8.4f}")
    print(f"  {'─'*70}")
    print(f"  {'AVERAGE (all configs)':<35} {avg_rmse:>8.4f}")
    print(f"  {'BEST SO FAR':<35} {current_best:>8.4f}")
    print(f"  {'─'*70}\n", flush=True)
    
    # Update best
    if avg_rmse < current_best:
        current_best = avg_rmse
        patience_counter = 0
    else:
        patience_counter += 1
    
    return avg_rmse, current_best, patience_counter, results


# ═════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═════════════════════════════════════════════════════════════

def trajectory_loss(pred, target, T):
    w = torch.linspace(1.0, 2.0, T, device=pred.device)
    w = w / w.sum()
    loss = torch.sqrt(((pred - target) ** 2 * w.unsqueeze(0)).sum(dim=1).mean() + 1e-8)
    return loss


def train_epoch(model, samples, optimizer, device, batch_size, scaler=None):
    model.train()
    random.shuffle(samples)
    total_loss = 0.0
    num_batches = 0
    
    dataset = GraphDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=min(4, NUM_CPUS),
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=2
    )
    
    for batch in dataloader:
        X = batch['X'].to(device, non_blocking=True)
        z_mask = batch['z_mask'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        traj_target = batch['traj'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                context = model.encode_graph_batch(X, z_mask, padding_mask)
                pred = model.decode_batch(context)
                loss = trajectory_loss(pred, traj_target, model.T)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            context = model.encode_graph_batch(X, z_mask, padding_mask)
            pred = model.decode_batch(context)
            loss = trajectory_loss(pred, traj_target, model.T)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, epoch, avg_rmse, best_rmse, results, args, is_best=False):
    """Save checkpoint with full validation results."""
    os.makedirs(args.save_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_val_rmse": avg_rmse,
        "best_val_rmse": best_rmse,
        "val_results": results,
        "hyperparams": {
            "node_feat_dim": NODE_FEAT_DIM,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_transformer_layers": args.tf_layers,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "T": args.T,
            "dropout": args.dropout,
            "epochs_trained": epoch,
            "stopped_epoch": epoch,
            "final_val_rmse": avg_rmse,
        }
    }
    
    # Regular checkpoint
    torch.save(checkpoint, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # Best model
    if is_best:
        torch.save(checkpoint, os.path.join(args.save_dir, args.save_name))
        print(f"    ★ New best model saved! (RMSE={best_rmse:.4f})")
    
    # Also save a running log
    log_entry = {
        "epoch": epoch,
        "avg_val_rmse": avg_rmse,
        "best_val_rmse": best_rmse,
        "val_results": results
    }
    log_path = os.path.join(args.save_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_graphs",    type=int,   default=30)
    p.add_argument("--mc_runs",       type=int,   default=20)
    p.add_argument("--T",             type=int,   default=T_STEPS)
    p.add_argument("--d_model",       type=int,   default=128)
    p.add_argument("--nhead",         type=int,   default=4)
    p.add_argument("--tf_layers",     type=int,   default=3)
    p.add_argument("--lstm_hidden",   type=int,   default=256)
    p.add_argument("--lstm_layers",   type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--save_dir",      type=str,   default="saved_models")
    p.add_argument("--save_name",     type=str,   default="zealot_transformer.pt")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--compile_model", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nZealotTransformer Training (Multi-Metric Validation)")
    print(f"  Checkpoint every {CHECKPOINT_EVERY} epochs")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # Build dataset
    samples = build_dataset(ALL_Z, args.num_graphs, args.T, args.mc_runs)
    split = int(0.8 * len(samples))
    train_data = samples[:split]
    val_data = samples[split:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # Model
    model = ZealotTransformer(
        node_feat_dim=NODE_FEAT_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.tf_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        T=args.T,
        dropout=args.dropout
    ).to(device)
    
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_lambda(epoch):
        warmup = 15
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    best_val_rmse = float('inf')
    patience_counter = 0
    stopped_epoch = None
    training_log = []

    print(f"\n  {'Ep':>5}  {'Train Loss':>10}  {'Val RMSE':>10}  {'Best':>10}  {'Pat':>5}  {'Time':>7}")
    print(f"  {'-'*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        loss = train_epoch(model, train_data, optimizer, device, args.batch_size, scaler)
        
        # Validate (only every 5 epochs to save time, or every epoch if you prefer)
        if epoch % 5 == 0 or epoch == 1:
            avg_rmse, best_val_rmse, patience_counter, val_results = validate_model(
                model, device, epoch, best_val_rmse, patience_counter
            )
            is_best = (avg_rmse == best_val_rmse)
        else:
            avg_rmse = float('nan')
            is_best = False
        
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {loss:10.4f}  {avg_rmse:10.4f}  {best_val_rmse:10.4f}  {patience_counter:5d}  {elapsed:6.1f}s")
        else:
            print(f"  {epoch:5d}  {loss:10.4f}  {'—':>10}  {best_val_rmse:10.4f}  {patience_counter:5d}  {elapsed:6.1f}s")
        
        # Save checkpoint every CHECKPOINT_EVERY epochs
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, avg_rmse, best_val_rmse, 
                           val_results if epoch % 5 == 0 else {}, args, is_best)
            print(f"    → Checkpoint saved (epoch {epoch})")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            stopped_epoch = epoch
            print(f"\n  ⏹️ Early stopping triggered at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    # Final save
    if stopped_epoch is None:
        stopped_epoch = args.epochs
    
    # Save final model with stopping info
    final_checkpoint = {
        "epoch": stopped_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_rmse": best_val_rmse,
        "stopped_epoch": stopped_epoch,
        "early_stopping": patience_counter >= EARLY_STOPPING_PATIENCE,
        "hyperparams": {
            "node_feat_dim": NODE_FEAT_DIM,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_transformer_layers": args.tf_layers,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "T": args.T,
            "dropout": args.dropout,
            "total_epochs_trained": stopped_epoch,
        }
    }
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(final_checkpoint, os.path.join(args.save_dir, args.save_name))
    
    print(f"\n✓ Training completed!")
    print(f"  Stopped at epoch: {stopped_epoch}")
    print(f"  Best validation RMSE: {best_val_rmse:.4f}")
    print(f"  Model saved to: {os.path.join(args.save_dir, args.save_name)}")
    
    # Save a summary file for your paper
    summary = {
        "stopped_epoch": stopped_epoch,
        "best_val_rmse": best_val_rmse,
        "early_stopping_triggered": patience_counter >= EARLY_STOPPING_PATIENCE,
        "patience_used": EARLY_STOPPING_PATIENCE,
        "total_epochs_scheduled": args.epochs,
        "checkpoint_frequency": CHECKPOINT_EVERY,
        "validation_frequency": 5,
    }
    with open(os.path.join(args.save_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Training summary saved to: {os.path.join(args.save_dir, 'training_summary.json')}")


if __name__ == "__main__":
    main()