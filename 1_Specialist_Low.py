"""
Specialist GAT Models — Specialist-Low (Z=2) and Specialist-High (Z=32)
========================================================================
Trains two specialist models, each on a single zealot count:
  - Specialist-Low:  trained exclusively on Z=2  (rho_Z = 0.002) --> Just this one used
  - Specialist-High: trained exclusively on Z=32 (rho_Z = 0.031)

Architecture identical to Global-GAT:
  - 4x GATConv layers, residual connections, LayerNorm, dropout=0.1
  - Node features: [s_i, z_i, deg_norm]
  - Loss: BCE + magnetisation consistency term

Both models are trained sequentially in a single job.

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SEED = 42
N    = 1024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SPECIALIST_CONFIGS = {
    'low':  {'z': 2,  'save_name': 'specialist_low_z2.pt'},
    'high': {'z': 32, 'save_name': 'specialist_high_z32.pt'},
}


def get_device():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"[device] {n_gpus} GPU(s) available:")
        for i in range(n_gpus):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  VRAM={p.total_memory/1e9:.1f}GB")
    else:
        n_gpus = 0
        device = torch.device("cpu")
        print("[device] No GPU — running on CPU.")
    return device, n_gpus


# ─────────────────────────────────────────────────────────────
# Graph + Simulation
# ─────────────────────────────────────────────────────────────

def build_ba_graph(N=1024, m=8, seed=None):
    return nx.barabasi_albert_graph(N, m, seed=seed)


def build_graph_tensors(G):
    n_nodes    = G.number_of_nodes()
    edges      = list(G.edges())
    ei         = [[u, v] for u, v in edges] + [[v, u] for u, v in edges]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    adj_list   = [list(G.neighbors(n)) for n in range(n_nodes)]
    degrees    = torch.tensor([len(a) for a in adj_list], dtype=torch.long)
    max_deg    = int(degrees.max().item())
    adj_pad    = torch.zeros(n_nodes, max_deg, dtype=torch.long)
    for i, a in enumerate(adj_list):
        if len(a) > 0:
            adj_pad[i, :len(a)] = torch.tensor(a, dtype=torch.long)
    deg_norm = degrees.float() / (degrees.float().max() + 1e-8)
    return edge_index, adj_pad, degrees, deg_norm


def place_zealots_hubs(G, num_zealots):
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    zealot_idx   = torch.tensor([n for n, _ in sorted_nodes[:num_zealots]])
    zealot_mask  = torch.zeros(G.number_of_nodes(), dtype=torch.bool)
    zealot_mask[zealot_idx] = True
    return zealot_mask


@torch.no_grad()
def simulate_cpu(adj_pad, degrees, zealot_mask, mc_steps=40):
    n_nodes        = adj_pad.shape[0]
    non_zealot_idx = torch.where(~zealot_mask)[0]
    n_nz           = non_zealot_idx.shape[0]
    opinions       = torch.randint(0, 2, (n_nodes,)).float() * 2 - 1
    opinions[zealot_mask] = 1.0
    snapshots = [opinions.clone()]
    for _ in range(mc_steps):
        chosen       = non_zealot_idx[torch.randint(0, n_nz, (n_nz,))]
        deg_chosen   = degrees[chosen].clamp(min=1)
        rand_pos     = (torch.rand(n_nz) * deg_chosen.float()).long()
        rand_pos     = rand_pos.clamp(0, adj_pad.shape[1] - 1)
        neighbor_ids = adj_pad[chosen, rand_pos]
        opinions[chosen] = opinions[neighbor_ids]
        opinions[zealot_mask] = 1.0
        snapshots.append(opinions.clone())
    return snapshots


def build_dataset(num_zealots, num_graphs, mc_steps, n_nodes, m):
    """Build dataset for a single zealot count (specialist training)."""
    print(f"\n  Building dataset for Z={num_zealots} ...", flush=True)
    dataset = []
    for g_idx in range(num_graphs):
        G = build_ba_graph(n_nodes, m, seed=g_idx)
        edge_index, adj_pad, degrees, deg_norm = build_graph_tensors(G)
        zealot_mask = place_zealots_hubs(G, num_zealots)
        snapshots   = simulate_cpu(adj_pad, degrees, zealot_mask, mc_steps)
        for t in range(len(snapshots) - 1):
            s_t  = (snapshots[t]     + 1) / 2
            s_t1 = (snapshots[t + 1] + 1) / 2
            x    = torch.stack([s_t, zealot_mask.float(), deg_norm], dim=1)
            dataset.append(Data(x=x, edge_index=edge_index, y=s_t1))
    random.shuffle(dataset)
    print(f"  Dataset size: {len(dataset):,}", flush=True)
    return dataset


# ─────────────────────────────────────────────────────────────
# Model  (identical architecture to Global-GAT)
# ─────────────────────────────────────────────────────────────

class SpecialistGAT(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.dropout   = dropout
        out_channels   = hidden_dim // 4   # per-head dim so concat = hidden_dim

        self.conv1 = GATConv(3,          out_channels, heads=4, dropout=dropout)
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_channels, heads=4, dropout=dropout)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_channels, heads=4, dropout=dropout)
        self.ln3   = nn.LayerNorm(hidden_dim)
        self.conv4 = GATConv(hidden_dim, out_channels, heads=4, dropout=dropout)
        self.ln4   = nn.LayerNorm(hidden_dim)
        self.out   = GATConv(hidden_dim, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Layer 1 — no residual (input dim mismatch)
        h = F.elu(self.conv1(x, edge_index))
        h = self.ln1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layers 2-4 — residual connections
        h1 = F.elu(self.conv2(h, edge_index))
        h  = self.ln2(h + h1)
        h  = F.dropout(h, p=self.dropout, training=self.training)

        h2 = F.elu(self.conv3(h, edge_index))
        h  = self.ln3(h + h2)
        h  = F.dropout(h, p=self.dropout, training=self.training)

        h3 = F.elu(self.conv4(h, edge_index))
        h  = self.ln4(h + h3)
        h  = F.dropout(h, p=self.dropout, training=self.training)

        return torch.sigmoid(self.out(h, edge_index)).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def voter_loss(pred, target):
    return (F.binary_cross_entropy(pred, target) +
            0.5 * (pred.mean() - target.mean()) ** 2)


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred  = model(batch.x, batch.edge_index)
        loss  = voter_loss(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sq, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch.x, batch.edge_index)
        p     = pred.cpu().numpy() * 2 - 1
        t     = batch.y.cpu().numpy() * 2 - 1
        sq   += float(np.sum((p - t) ** 2))
        n    += len(p)
    return float(np.sqrt(sq / max(n, 1)))


# ─────────────────────────────────────────────────────────────
# Train one specialist
# ─────────────────────────────────────────────────────────────

def train_specialist(label, num_zealots, save_name, args, device, n_gpus):
    print("\n" + "=" * 55)
    print(f"  Specialist-{label.capitalize()}  (Z={num_zealots})")
    print("=" * 55)
    print(f"  rho_Z       : {num_zealots/N:.4f}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  num_graphs  : {args.num_graphs}")
    print(f"  mc_steps    : {args.mc_steps}")
    print(f"  hidden_dim  : {args.hidden_dim}")
    print(f"  dropout     : {args.dropout}")
    print("=" * 55)

    # ── Dataset ──────────────────────────────────────────────
    dataset = build_dataset(num_zealots, args.num_graphs,
                             args.mc_steps, N, m=8)
    split   = int(0.8 * len(dataset))

    effective_batch = args.batch_size * max(1, n_gpus)
    train_loader = DataLoader(
        dataset[:split], batch_size=effective_batch,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        dataset[split:], batch_size=effective_batch,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))

    print(f"Train: {split:,}  |  Val: {len(dataset)-split:,}")

    # ── Model ────────────────────────────────────────────────
    model = SpecialistGAT(hidden_dim=args.hidden_dim,
                           dropout=args.dropout).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}\n")

    # ── Optimizer & Scheduler ────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup = 20
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────
    best_rmse  = float("inf")
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)
    save_path  = os.path.join(args.save_dir, save_name)

    print(f"  {'Ep':>5}  {'Loss':>8}  {'RMSE':>8}  {'Best':>8}  "
          f"{'LR':>10}  {'Time':>7}")
    print(f"  {'-'*55}")

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        rmse = evaluate(model, val_loader, device)
        scheduler.step()

        if rmse < best_rmse:
            best_rmse  = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch <= 10 or epoch % 20 == 0:
            print(f"  {epoch:>5}  {loss:>8.4f}  {rmse:>8.4f}  "
                  f"{best_rmse:>8.4f}  {lr_now:>10.2e}  "
                  f"{time.time()-t0:>6.1f}s", flush=True)

        if epoch % 50 == 0 and best_state is not None:
            torch.save({
                "model_state_dict": best_state,
                "best_val_rmse":    best_rmse,
                "epoch":            epoch,
                "hyperparams": {
                    "hidden_dim":  args.hidden_dim,
                    "dropout":     args.dropout,
                    "trained_Z":   [num_zealots],
                    "N":           N,
                    "model_type":  f"SpecialistGAT_{label}",
                }
            }, save_path)
            print(f"  [checkpoint at epoch {epoch}  RMSE={best_rmse:.4f}]",
                  flush=True)

    # ── Final save ───────────────────────────────────────────
    if best_state is not None:
        torch.save({
            "model_state_dict": best_state,
            "best_val_rmse":    best_rmse,
            "hyperparams": {
                "hidden_dim":  args.hidden_dim,
                "dropout":     args.dropout,
                "trained_Z":   [num_zealots],
                "N":           N,
                "model_type":  f"SpecialistGAT_{label}",
            }
        }, save_path)
        print(f"\n✓ Specialist-{label.capitalize()} saved → {save_path}")
        print(f"  Best Val RMSE: {best_rmse:.4f}")

    return best_rmse


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Specialist GAT — trains Low (Z=2) then High (Z=32)")
    parser.add_argument("--batch_size",   type=int,   required=True)
    parser.add_argument("--epochs",       type=int,   required=True)
    parser.add_argument("--lr",           type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--num_graphs",   type=int,   required=True)
    parser.add_argument("--mc_steps",     type=int,   required=True)
    parser.add_argument("--num_workers",  type=int,   required=True)
    parser.add_argument("--hidden_dim",   type=int,   required=True)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--save_dir",     type=str,   default="saved_models")
    args = parser.parse_args()

    device, n_gpus = get_device()

    results = {}
    for label, cfg in SPECIALIST_CONFIGS.items():
        rmse = train_specialist(
            label, cfg['z'], cfg['save_name'], args, device, n_gpus)
        results[label] = rmse

    print("\n" + "=" * 55)
    print("  Final Summary")
    print("=" * 55)
    for label, rmse in results.items():
        cfg = SPECIALIST_CONFIGS[label]
        print(f"  Specialist-{label.capitalize():<5} "
              f"(Z={cfg['z']:>2})  Best Val RMSE: {rmse:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()