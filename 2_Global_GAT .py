"""
Global GAT Model (no conditioning) — LUMI GPU Version
======================================================
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

SEED  = 42
ALL_Z = [2, 8, 16, 32]
N     = 1024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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
    n_nodes  = G.number_of_nodes()
    edges    = list(G.edges())
    ei       = [[u, v] for u, v in edges] + [[v, u] for u, v in edges]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    adj_list = [list(G.neighbors(n)) for n in range(n_nodes)]
    degrees  = torch.tensor([len(a) for a in adj_list], dtype=torch.long)
    max_deg  = int(degrees.max().item())
    adj_pad  = torch.zeros(n_nodes, max_deg, dtype=torch.long)
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


def build_dataset(zealot_list, num_graphs, mc_steps, n_nodes, m):
    print("\nBuilding dataset ...", flush=True)
    dataset = []
    for z in zealot_list:
        print(f"  Z={z}", flush=True)
        for g_idx in range(num_graphs):
            G = build_ba_graph(n_nodes, m, seed=g_idx)
            edge_index, adj_pad, degrees, deg_norm = build_graph_tensors(G)
            zealot_mask = place_zealots_hubs(G, z)
            snapshots   = simulate_cpu(adj_pad, degrees, zealot_mask, mc_steps)
            for t in range(len(snapshots) - 1):
                s_t  = (snapshots[t]     + 1) / 2
                s_t1 = (snapshots[t + 1] + 1) / 2
                x = torch.stack([s_t, zealot_mask.float(), deg_norm], dim=1)
                dataset.append(Data(x=x, edge_index=edge_index, y=s_t1))
    random.shuffle(dataset)
    print(f"Dataset size: {len(dataset):,}", flush=True)
    return dataset


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class GlobalGATModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        out_channels = hidden_dim // 4
        
        self.conv1 = GATConv(3, out_channels, heads=4)
        self.conv2 = GATConv(hidden_dim, out_channels, heads=4)
        self.conv3 = GATConv(hidden_dim, out_channels, heads=4)
        self.out   = GATConv(hidden_dim, 1,  heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        return torch.sigmoid(self.out(x, edge_index)).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / count


@torch.no_grad()
def evaluate(model, loader, device):
    base_model = model.module if hasattr(model, 'module') else model
    base_model.eval()
    sq, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred  = base_model(batch.x, batch.edge_index)
        p     = pred.cpu().numpy() * 2 - 1
        t     = batch.y.cpu().numpy() * 2 - 1
        sq   += float(np.sum((p - t) ** 2))
        n    += len(p)
    return float(np.sqrt(sq / n))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",   type=int,   required=True)
    parser.add_argument("--epochs",       type=int,   required=True)
    parser.add_argument("--lr",           type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--num_graphs",   type=int,   required=True)
    parser.add_argument("--mc_steps",     type=int,   required=True)
    parser.add_argument("--num_workers",  type=int,   required=True)
    parser.add_argument("--hidden_dim",   type=int,   required=True)
    parser.add_argument("--save_dir",     type=str,   default="saved_models")
    parser.add_argument("--save_name",    type=str,   default="Global-GAT.pt")
    args = parser.parse_args()

    device, n_gpus = get_device()

    print("\n" + "=" * 55)
    print("  Global GAT (no conditioning) — LUMI GPU")
    print("=" * 55)
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr}")
    print(f"  num_graphs  : {args.num_graphs}")
    print(f"  mc_steps    : {args.mc_steps}")
    print(f"  hidden_dim  : {args.hidden_dim}") # Added for logging
    print(f"  n_gpus      : {n_gpus}")
    print("=" * 55)

    # ── Dataset ──────────────────────────────────────────────
    dataset = build_dataset(ALL_Z, args.num_graphs, args.mc_steps, N, m=8)
    split   = int(0.8 * len(dataset))

    effective_batch = args.batch_size * max(1, n_gpus)
    print(f"\nEffective batch size: {effective_batch}")

    train_loader = DataLoader(
        dataset[:split], batch_size=effective_batch,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        dataset[split:], batch_size=effective_batch,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"))

    print(f"Train: {split:,}  |  Val: {len(dataset)-split:,}")

    # ── Model ────────────────────────────────────────────────
    model = GlobalGATModel(hidden_dim=args.hidden_dim)
    if n_gpus > 1:
        print(f"\nUsing DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}\n")

    # ── Optimizer & Scheduler ────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────
    best_rmse  = float("inf")
    best_state = None
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    print(f"  {'Ep':>5}  {'Loss':>8}  {'RMSE':>8}  {'Best':>8}  {'LR':>10}  {'Time':>7}")
    print(f"  {'-'*55}")

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        rmse = evaluate(model, val_loader, device)
        scheduler.step()

        if rmse < best_rmse:
            best_rmse  = rmse
            base_model = model.module if hasattr(model, 'module') else model
            best_state = {k: v.clone() for k, v in base_model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch <= 10 or epoch % 20 == 0:
            print(f"  {epoch:>5}  {loss:>8.4f}  {rmse:>8.4f}  "
                  f"{best_rmse:>8.4f}  {lr_now:>10.2e}  "
                  f"{time.time()-t0:>6.1f}s", flush=True)

        if epoch % 50 == 0 and best_state is not None:
            torch.save({"model_state_dict": best_state,
                        "best_val_rmse": best_rmse,
                        "epoch": epoch}, save_path)
            print(f"  [checkpoint at epoch {epoch}  RMSE={best_rmse:.4f}]")

    # ── Final save ───────────────────────────────────────────
    torch.save({"model_state_dict": best_state,
                "best_val_rmse":    best_rmse}, save_path)
    print(f"\n✓ Saved to {save_path}")
    print(f"  Best Val RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
