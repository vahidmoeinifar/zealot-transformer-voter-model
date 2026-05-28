#!/usr/bin/env python3
"""
eval_ood_comparison.py — Evaluate Full ZT vs BA-only ZT on OOD configurations
================================================================================
Evaluates:
  • Supplement: RGG topology (unseen), N=1024, Z∈{2,32}, hub placement
  • Supplement: BA large networks (unseen sizes), N∈{4096,8192}, Z∈{64,128}, hub placement

Compares:
  - Full ZealotTransformer (trained on BA/ER/WS, multi-size, hub+random)
  - BA-only ZealotTransformer (trained on BA only, N=1024 only, hub only)
  - All baselines (SpectralLSTM, PA-LSTM, GlobalGAT, SpecLow, Persistence, MeanField)

Usage:
  python eval_ood_comparison.py \\
      --full_zt_checkpoint       saved_models/zealot_transformer.pt \\
      --ba_only_zt_checkpoint    saved_models/zealot_transformer_1024_BA.pt \\
      --spectral_lstm_checkpoint saved_models/universal_lstm.pt \\
      --pa_lstm_checkpoint       saved_models/pa-lstm.pt \\
      --spec_low_checkpoint      saved_models/specialist_low.pt \\
      --global_gat_checkpoint    saved_models/global_gat.pt \\
      --out_dir result_ood \\
      --mc_runs 128 \\
      --workers 16
"""

import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.integrate import odeint
from scipy.sparse.linalg import eigsh
from torch_geometric.nn import GATConv
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
T_STEPS       = 50
NODE_FEAT_DIM = 5
MC_RUNS       = 128
VAL_GRAPHS    = 10

# Large BA evaluation — UPDATED
BA_LARGE_SIZES = [4096, 8192]
BA_LARGE_Z     = [64, 128]

# RGG evaluation
RGG_SIZES      = [1024]
RGG_Z          = [2, 32]

# Model names for output
MODEL_NAMES = [
    "Persistence", "MeanField",
    "SpecLow", "GlobalGAT",
    "SpectralLSTM", "PA-LSTM",
    "ZealotTransformer_Full",
    "ZealotTransformer_BAonly",
]


# ═════════════════════════════════════════════════════════════
# Graph helpers
# ═════════════════════════════════════════════════════════════

def make_graph(topo, n, m=8, seed=None):
    rng_np = np.random.default_rng(seed)
    if topo == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == "rgg":
        r = np.sqrt(2 * m / (n * np.pi))
        for attempt in range(15):
            pos = {i: (float(rng_np.random()), float(rng_np.random()))
                   for i in range(n)}
            G = nx.random_geometric_graph(n, r, pos=pos)
            if nx.is_connected(G): break
            r *= 1.05
        G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(f"Unknown topology: {topo}")
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G


def place_hubs(G, Z):
    return set(n for n, _ in
               sorted(G.degree(), key=lambda x: x[1], reverse=True)[:Z])


def get_zealot_set(G, placement, Z, rng):
    if placement == "hub":
        return place_hubs(G, Z)
    else:
        raise ValueError(f"Only hub placement supported, got: {placement}")


# ═════════════════════════════════════════════════════════════
# Parallel MC — batched WITHIN each graph
# ═════════════════════════════════════════════════════════════

def _mc_batch(args):
    adj, is_z_arr, non_z, n_runs, T, seed = args
    rng   = np.random.default_rng(seed)
    N_g   = len(adj)
    is_z  = is_z_arr.astype(bool)
    trajs = np.zeros((n_runs, T), dtype=np.float32)
    for r in range(n_runs):
        ops = rng.choice([-1., 1.], size=N_g).astype(np.float32)
        ops[is_z] = 1.
        for t in range(T):
            trajs[r, t] = ops.mean()
            chosen = rng.choice(non_z, size=len(non_z), replace=True)
            for nd in chosen:
                nbrs = adj[nd]
                if nbrs:
                    ops[nd] = ops[nbrs[rng.integers(0, len(nbrs))]]
            ops[is_z] = 1.
    return trajs.mean(axis=0)


def simulate_graph_parallel(G, zealot_set, mc_runs, workers, seed):
    N_g   = G.number_of_nodes()
    adj   = [list(G.neighbors(i)) for i in range(N_g)]
    is_z  = np.zeros(N_g, dtype=np.float32)
    for z in zealot_set:
        is_z[z] = 1.0
    non_z = np.where(is_z == 0)[0]

    chunk_size = max(1, mc_runs // workers)
    chunks     = []
    remaining  = mc_runs
    chunk_seed = seed
    while remaining > 0:
        n = min(chunk_size, remaining)
        chunks.append((adj, is_z, non_z, n, T_STEPS, chunk_seed))
        remaining  -= n
        chunk_seed += 1

    with Pool(processes=min(workers, len(chunks))) as pool:
        batch_means = pool.map(_mc_batch, chunks)

    sizes  = [c[3] for c in chunks]
    total  = sum(sizes)
    result = sum(m * s / total for m, s in zip(batch_means, sizes))
    return result.astype(np.float32)


def build_graph_record(topo, placement, Z, n, g_idx, seed_base, mc_runs, workers):
    seed = seed_base + g_idx * 1000 + abs(hash((topo, placement, Z, n))) % 9999
    rng  = np.random.default_rng(seed)
    try:
        G  = make_graph(topo, n, m=8, seed=int(rng.integers(0, 99999)))
        zs = get_zealot_set(G, placement, Z, rng)
        gt = simulate_graph_parallel(G, zs, mc_runs, workers, seed + 1)

        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        return {
            "g_idx": g_idx, "topo": topo, "placement": placement,
            "Z": Z, "n": n, "seed": seed,
            "gt": gt.tolist(),
            "m0": float(gt[0]),
            "rho": Z / G.number_of_nodes(),
            "degrees": degrees.tolist(),
            "zealot_list": list(zs),
            "edges": list(G.edges()),
            "N_g": G.number_of_nodes(),
        }
    except Exception as e:
        print(f"    [build_graph_record] g_idx={g_idx}: {e}")
        return None


# ═════════════════════════════════════════════════════════════
# Descriptors
# ═════════════════════════════════════════════════════════════

def _spectral_gap(G):
    try:
        n    = G.number_of_nodes()
        L    = nx.laplacian_matrix(G).astype(float)
        vals = eigsh(L, k=min(3, n-1), which="SM",
                     return_eigenvectors=False, tol=1e-2, maxiter=1000)
        return float(sorted(vals)[1]) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0


def _fiedler_vector(G):
    n   = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    return deg / (deg.max() + 1e-8)


def compute_spectral_8d(G, Z, n, topo):
    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    mu  = deg.mean()
    return np.array([
        Z / n, _spectral_gap(G), mu / n,
        deg.std() / (mu + 1e-8), nx.average_clustering(G),
        1.0 if topo == "ba" else 0.0,
        0.0,
        0.0,
    ], dtype=np.float32)


def compute_pa_11d(G, Z, n, topo, zealot_set):
    base = compute_spectral_8d(G, Z, n, topo)
    deg  = np.array([d for _, d in G.degree()], dtype=np.float64)
    mu   = deg.mean()
    zl   = list(zealot_set)
    hub_s = float(deg[zl].mean() / (mu + 1e-8)) if zl else 1.0
    try:
        btwn = nx.betweenness_centrality(G, normalized=True, endpoints=False)
        bv   = np.array(list(btwn.values()))
        bridge_s = float(np.mean([btwn[z] for z in zl]) /
                         (bv.mean() + 1e-10)) if zl else 1.0
    except Exception:
        bridge_s = 1.0
    fiedler   = _fiedler_vector(G)
    fiedler_s = float(np.abs(fiedler[zl]).mean()) \
                if zl and len(fiedler) == n else 0.0
    return np.concatenate([base, [hub_s, bridge_s, fiedler_s]]).astype(np.float32)


def compute_node_features_5d(G, zealot_set):
    N_g   = G.number_of_nodes()
    deg   = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn    = deg / (deg.max() + 1e-8)
    z_i   = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set:
        z_i[nd] = 1.0
    fiedler = _fiedler_vector(G)
    pr    = dn.copy()
    clust = np.zeros(N_g, dtype=np.float32)
    try:
        if N_g <= 2000:
            cd = nx.clustering(G)
            clust = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        pass
    return np.stack([z_i, dn, fiedler, pr, clust], axis=1).astype(np.float32)


def normalize_desc(desc, stats):
    if stats is None:
        return desc
    mean = np.asarray(stats[0], dtype=np.float32)
    std  = np.asarray(stats[1], dtype=np.float32)
    return (desc - mean) / (std + 1e-8)


# ═════════════════════════════════════════════════════════════
# Model architectures
# ═════════════════════════════════════════════════════════════

class LocalGATModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        out_ch = hidden_dim // 4
        self.conv1 = GATConv(3, out_ch, heads=4, dropout=dropout)
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln3   = nn.LayerNorm(hidden_dim)
        self.conv4 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln4   = nn.LayerNorm(hidden_dim)
        self.out   = GATConv(hidden_dim, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        h  = F.elu(self.conv1(x, edge_index))
        h  = self.ln1(h)
        h  = F.dropout(h, p=self.dropout, training=self.training)
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


class GlobalGATModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.0):
        super().__init__()
        out_ch = hidden_dim // 4
        self.conv1 = GATConv(3, out_ch, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.out   = GATConv(hidden_dim, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        return torch.sigmoid(self.out(x, edge_index)).squeeze(-1)


class TrajectoryLSTM(nn.Module):
    def __init__(self, desc_dim=8, hidden_dim=256, num_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = T
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2))
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, d):
        B   = d.shape[0]
        enc = self.encoder(d).view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=d.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out.squeeze(1))
            preds.append(p)
            inp = p.detach().unsqueeze(1)
        return torch.cat(preds, dim=1)


class PALSTMModel(nn.Module):
    def __init__(self, desc_dim=11, hidden_dim=256, num_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T = T
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2))
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, d):
        B   = d.shape[0]
        enc = self.encoder(d).view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=d.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out.squeeze(1))
            preds.append(p)
            inp = p.detach().unsqueeze(1)
        return torch.cat(preds, dim=1)


class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128, nhead=4,
                 num_transformer_layers=3, lstm_hidden=256, lstm_layers=2,
                 T=T_STEPS, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.T = T
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_transformer_layers,
            enable_nested_tensor=False)
        self.ctx_projector = nn.Sequential(
            nn.Linear(2 * d_model, lstm_hidden * 2), nn.GELU(),
            nn.Linear(lstm_hidden * 2, lstm_layers * lstm_hidden * 2))
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H = self.node_encoder(X)
        H = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1, keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(B, self.lstm_layers, self.lstm_hidden * 2)
        h0 = proj[:, :, :self.lstm_hidden].transpose(0, 1).contiguous()
        c0 = proj[:, :, self.lstm_hidden:].transpose(0, 1).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=ctx.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out)
            preds.append(p)
            inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0) * 2 - 1


# ═════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════

def _clean(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()}


def load_models(args, device):
    loaded = {}

    def _try(key, path, strict=True, pair=False):
        if not (path and os.path.isfile(path)):
            print(f"  - {key}: not found")
            return
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            hp = ckpt.get("hyperparams", {})
            if key in ["SpecLow"]:
                m = LocalGATModel(hidden_dim=hp.get("hidden_dim", 256),
                                  dropout=hp.get("dropout", 0.1))
            elif key in ["GlobalGAT"]:
                m = GlobalGATModel(hidden_dim=hp.get("hidden_dim", 256))
            elif key in ["SpectralLSTM"]:
                m = TrajectoryLSTM(desc_dim=hp.get("desc_dim", 8),
                                   hidden_dim=hp.get("hidden_dim", 256),
                                   num_layers=hp.get("num_layers", 2),
                                   T=hp.get("T", T_STEPS))
            elif key in ["PA-LSTM"]:
                m = PALSTMModel(desc_dim=hp.get("desc_dim", 11),
                                hidden_dim=hp.get("hidden_dim", 256),
                                num_layers=hp.get("num_layers", 2),
                                T=hp.get("T", T_STEPS))
            elif key in ["ZealotTransformer_Full", "ZealotTransformer_BAonly"]:
                m = ZealotTransformer(
                    node_feat_dim=hp.get("node_feat_dim", NODE_FEAT_DIM),
                    d_model=hp.get("d_model", 128),
                    nhead=hp.get("nhead", 4),
                    num_transformer_layers=hp.get("num_transformer_layers", 3),
                    lstm_hidden=hp.get("lstm_hidden", 256),
                    lstm_layers=hp.get("lstm_layers", 2),
                    T=hp.get("T", T_STEPS))
            else:
                return
            m.load_state_dict(_clean(ckpt["model_state_dict"]), strict=strict)
            m = m.to(device).eval()
            loaded[key] = (m, ckpt.get("norm_stats", None)) if pair else m
            print(f"  ✓ {key}: {path}")
        except Exception as e:
            print(f"  ✗ {key}: {e}")

    _try("SpecLow", args.spec_low_checkpoint, strict=True)
    _try("GlobalGAT", args.global_gat_checkpoint, strict=True)
    _try("SpectralLSTM", args.spectral_lstm_checkpoint, strict=False, pair=True)
    _try("PA-LSTM", args.pa_lstm_checkpoint, strict=False, pair=True)
    _try("ZealotTransformer_Full", args.full_zt_checkpoint, strict=True)
    _try("ZealotTransformer_BAonly", args.ba_only_zt_checkpoint, strict=True)

    return loaded


# ═════════════════════════════════════════════════════════════
# Baselines & inference
# ═════════════════════════════════════════════════════════════

def predict_persistence(m0):
    return np.full(T_STEPS, m0, dtype=np.float32)


def predict_meanfield(rho, m0, alpha=0.08, beta=2.5):
    def ode(m, t):
        return -alpha * m + beta * rho * (1 - m)
    sol = odeint(ode, [m0], np.linspace(0, T_STEPS - 1, T_STEPS))
    return np.clip(sol.squeeze(), -1, 1).astype(np.float32)


def _rebuild_graph(rec):
    G = nx.Graph()
    G.add_nodes_from(range(rec["N_g"]))
    G.add_edges_from(rec["edges"])
    return G


def _edge_index_from_rec(rec, device):
    edges = rec["edges"]
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0).to(device)


@torch.no_grad()
def infer_gat(model, rec, device):
    N_g = rec["N_g"]
    deg = np.array(rec["degrees"], dtype=np.float32)
    dn = deg / (deg.max() + 1e-8)
    z_i = np.zeros(N_g, dtype=np.float32)
    for nd in rec["zealot_list"]:
        z_i[nd] = 1.0
    ops = np.random.choice([-1., 1.], size=N_g).astype(np.float32)
    ops[z_i == 1] = 1.
    s_n = (ops + 1) / 2.
    x = torch.tensor(np.stack([s_n, z_i, dn], axis=1),
                     dtype=torch.float32).to(device)
    ei = _edge_index_from_rec(rec, device)
    zm = torch.tensor(z_i, dtype=torch.float32).to(device)
    preds = []
    for _ in range(T_STEPS):
        probs = model(x, ei)
        samp = torch.bernoulli(probs)
        spin = samp * 2 - 1
        spin[zm == 1] = 1.
        preds.append(spin.mean().item())
        ns = samp.clone()
        ns[zm == 1] = 1.
        x = torch.stack([ns, x[:, 1], x[:, 2]], dim=1)
    return np.array(preds, dtype=np.float32)


@torch.no_grad()
def infer_spectral_lstm(model_ns, rec, device):
    model, ns = model_ns
    G = _rebuild_graph(rec)
    desc = compute_spectral_8d(G, rec["Z"], rec["n"], rec["topo"])
    desc = normalize_desc(desc[np.newaxis, :], ns)
    pred = model(torch.tensor(desc, dtype=torch.float32).to(device))
    return (pred.squeeze(0).cpu().numpy() * 2 - 1).astype(np.float32)


@torch.no_grad()
def infer_pa_lstm(model_ns, rec, device):
    model, ns = model_ns
    G = _rebuild_graph(rec)
    zs = set(rec["zealot_list"])
    desc = compute_pa_11d(G, rec["Z"], rec["n"], rec["topo"], zs)
    desc = normalize_desc(desc[np.newaxis, :], ns)
    pred = model(torch.tensor(desc, dtype=torch.float32).to(device))
    return (pred.squeeze(0).cpu().numpy() * 2 - 1).astype(np.float32)


@torch.no_grad()
def infer_zt(model, rec, device):
    G = _rebuild_graph(rec)
    zs = set(rec["zealot_list"])
    X = compute_node_features_5d(G, zs)
    zm = np.zeros(rec["N_g"], dtype=bool)
    for nd in zs:
        zm[nd] = True
    pred = model(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(zm, dtype=torch.bool).to(device))
    return pred.cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════
# Cell evaluation
# ═════════════════════════════════════════════════════════════

def evaluate_cell(topo, placement, Z, n, loaded, device,
                  seed_base, workers, mc_runs):
    all_preds = {m: [] for m in MODEL_NAMES}
    gt_list = []

    for g_idx in range(VAL_GRAPHS):
        print(f"      graph {g_idx+1}/{VAL_GRAPHS}...", end="", flush=True)
        t_g = time.time()
        rec = build_graph_record(topo, placement, Z, n,
                                 g_idx, seed_base, mc_runs, workers)
        if rec is None:
            print(" skip")
            continue

        gt = np.array(rec["gt"])
        m0 = rec["m0"]
        rho = rec["rho"]
        gt_list.append(gt)

        all_preds["Persistence"].append(predict_persistence(m0))
        all_preds["MeanField"].append(predict_meanfield(rho, m0))

        # SpecLow
        if "SpecLow" in loaded:
            try:
                p = infer_gat(loaded["SpecLow"], rec, device)
                all_preds["SpecLow"].append(p)
            except Exception as e:
                print(f"\n    [SpecLow] error: {e}")
                all_preds["SpecLow"].append(predict_persistence(m0))

        # GlobalGAT
        if "GlobalGAT" in loaded:
            try:
                p = infer_gat(loaded["GlobalGAT"], rec, device)
                all_preds["GlobalGAT"].append(p)
            except Exception as e:
                print(f"\n    [GlobalGAT] error: {e}")
                all_preds["GlobalGAT"].append(predict_persistence(m0))

        # SpectralLSTM
        if "SpectralLSTM" in loaded:
            try:
                p = infer_spectral_lstm(loaded["SpectralLSTM"], rec, device)
                all_preds["SpectralLSTM"].append(p)
            except Exception as e:
                print(f"\n    [SpectralLSTM] error: {e}")
                all_preds["SpectralLSTM"].append(predict_persistence(m0))

        # PA-LSTM
        if "PA-LSTM" in loaded:
            try:
                p = infer_pa_lstm(loaded["PA-LSTM"], rec, device)
                all_preds["PA-LSTM"].append(p)
            except Exception as e:
                print(f"\n    [PA-LSTM] error: {e}")
                all_preds["PA-LSTM"].append(predict_persistence(m0))

        # ZealotTransformer Full
        if "ZealotTransformer_Full" in loaded:
            try:
                p = infer_zt(loaded["ZealotTransformer_Full"], rec, device)
                all_preds["ZealotTransformer_Full"].append(p)
            except Exception as e:
                print(f"\n    [ZealotTransformer_Full] error: {e}")
                all_preds["ZealotTransformer_Full"].append(predict_persistence(m0))

        # ZealotTransformer BA-only
        if "ZealotTransformer_BAonly" in loaded:
            try:
                p = infer_zt(loaded["ZealotTransformer_BAonly"], rec, device)
                all_preds["ZealotTransformer_BAonly"].append(p)
            except Exception as e:
                print(f"\n    [ZealotTransformer_BAonly] error: {e}")
                all_preds["ZealotTransformer_BAonly"].append(predict_persistence(m0))

        print(f" {time.time()-t_g:.1f}s")

    if not gt_list:
        return {m: (float("nan"), float("nan")) for m in MODEL_NAMES}

    results = {}
    for mname in MODEL_NAMES:
        preds = all_preds[mname]
        if not preds:
            results[mname] = (float("nan"), float("nan"))
            continue
        per = [float(np.sqrt(np.mean((np.array(p) - np.array(g)) ** 2)))
               for p, g in zip(preds, gt_list)]
        results[mname] = (float(np.mean(per)), float(np.std(per)))
    return results


# ═════════════════════════════════════════════════════════════
# Flat text table writer
# ═════════════════════════════════════════════════════════════

def _f(mean, std):
    if mean is None or np.isnan(float(mean)):
        return "   N/A      "
    return f"{mean:.3f} ± {std:.3f}"


def write_table(title, meta_lines, col_models, row_keys, row_label_fn,
                results, out_path):
    col_w = 18
    hdr = f"{'':40}" + "".join(f"{m:>{col_w}}" for m in col_models)
    sep = "-" * len(hdr)
    rows = []
    for key in row_keys:
        label = row_label_fn(key)
        cell = results.get(key, {})
        row = f"{label:<40}"
        for m in col_models:
            ms = cell.get(m, (float("nan"), float("nan")))
            row += f"{_f(*ms):>{col_w}}"
        rows.append(row)
    lines = ["=" * len(sep), title, "=" * len(sep)]
    lines += meta_lines
    lines += [sep, hdr, sep]
    lines += rows
    lines += [sep, ""]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════
# Evaluation plan
# ═════════════════════════════════════════════════════════════

def build_plan():
    cells = []
    # Large BA: N∈{4096,8192}, Z∈{64,128}, hub placement
    for Z in BA_LARGE_Z:
        for n in BA_LARGE_SIZES:
            cells.append(("hub", "ba", Z, n))
    # RGG: N=1024, Z∈{2,32}, hub placement
    for Z in RGG_Z:
        for n in RGG_SIZES:
            cells.append(("hub", "rgg", Z, n))
    return cells


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_zt_checkpoint", type=str, required=True,
                   help="Full ZealotTransformer (BA/ER/WS, multi-size)")
    p.add_argument("--ba_only_zt_checkpoint", type=str, required=True,
                   help="BA-only ZealotTransformer (N=1024 only)")
    p.add_argument("--spectral_lstm_checkpoint", type=str, default=None)
    p.add_argument("--pa_lstm_checkpoint", type=str, default=None)
    p.add_argument("--spec_low_checkpoint", type=str, default=None)
    p.add_argument("--global_gat_checkpoint", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="result_ood")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc_runs", type=int, default=MC_RUNS)
    p.add_argument("--val_graphs", type=int, default=VAL_GRAPHS)
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    return p.parse_args()


def main():
    args = parse_args()
    global MC_RUNS, VAL_GRAPHS
    MC_RUNS = args.mc_runs
    VAL_GRAPHS = args.val_graphs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_properties(0).name}")
    print(f"Workers: {args.workers}  MC runs: {MC_RUNS}  "
          f"Val graphs: {VAL_GRAPHS}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tbl_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(tbl_dir, exist_ok=True)

    print("\nLoading models...")
    loaded = load_models(args, device)

    plan = build_plan()
    total = len(plan)
    print(f"\nPlan: {total} cells")
    print(f"  BA Large: N∈{BA_LARGE_SIZES}, Z∈{BA_LARGE_Z} → {len(BA_LARGE_SIZES) * len(BA_LARGE_Z)} cells")
    print(f"  RGG: N={RGG_SIZES}, Z∈{RGG_Z} → {len(RGG_SIZES) * len(RGG_Z)} cells\n")

    results = {}
    raw_export = {}
    t0 = time.time()

    for i, (pl, topo, Z, n) in enumerate(plan, 1):
        print(f"\n  [{i:3d}/{total}]  pl={pl:8} topo={topo:5} Z={Z:4} N={n}",
              flush=True)
        t1 = time.time()
        res = evaluate_cell(topo, pl, Z, n, loaded, device,
                            args.seed, args.workers, MC_RUNS)
        results[(pl, topo, Z, n)] = res
        best = min((v[0] for v in res.values() if not np.isnan(v[0])),
                   default=float("nan"))
        print(f"  → best_RMSE={best:.4f}  total={time.time()-t1:.1f}s",
              flush=True)
        for mname, (m, s) in res.items():
            key = f"{pl}_{topo}_Z{Z}_N{n}_{mname}"
            raw_export[key] = {"mean_rmse": m, "std_rmse": s}

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min")
    print("\nWriting tables...")

    meta = [f"MC runs: {MC_RUNS}   Val graphs: {VAL_GRAPHS}"]

    # Table 1: RGG comparison
    write_table(
        "SUPPLEMENT: RGG Topology — Hub Placement (Unseen During Training)",
        meta + ["RGG never seen during training. Comparing Full ZT vs BA-only ZT."],
        MODEL_NAMES,
        [(Z, 1024) for Z in RGG_Z],
        lambda k: f"RGG  Hub  Z={k[0]:4}  N={k[1]}",
        {(Z, 1024): results.get(("hub", "rgg", Z, 1024), {}) for Z in RGG_Z},
        os.path.join(tbl_dir, "suppl_rgg_comparison.txt"))

    # Table 2: BA Large N comparison
    write_table(
        "SUPPLEMENT: BA Large Networks — Hub Placement (Unseen Sizes)",
        meta + [f"N∈{BA_LARGE_SIZES}, Z∈{BA_LARGE_Z} outside training distribution"],
        MODEL_NAMES,
        [(Z, n) for Z in BA_LARGE_Z for n in BA_LARGE_SIZES],
        lambda k: f"BA   Hub  Z={k[0]:4}  N={k[1]}",
        {(Z, n): results.get(("hub", "ba", Z, n), {})
         for Z in BA_LARGE_Z for n in BA_LARGE_SIZES},
        os.path.join(tbl_dir, "suppl_ba_large_comparison.txt"))

    json_path = os.path.join(args.out_dir, "results_ood.json")
    with open(json_path, "w") as f:
        json.dump(raw_export, f, indent=2)
    print(f"\n  Raw JSON: {json_path}")
    print(f"\nDone. Tables in {tbl_dir}/")


if __name__ == "__main__":
    main()