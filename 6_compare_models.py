#!/usr/bin/env python3
"""
compare_models.py — LUMI version
=================================
Compares all neural models + baselines. Outputs 3 plain text tables + figures.

Tables:
  table1_cross_topology.txt    — BA/ER/WS × Z ∈ {2,8,16,32}, hub placement
  table2_zealot_placement.txt  — BA, Hub / Bridge / Random × Z ∈ {2,8,16,32}
  table3_size_generalization.txt — all topologies × N ∈ {256,512,1024,2048,4096}

Author: Vahid Moeinifar (AGH University of Science and Technology)
"""

import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse.linalg import eigsh
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
ALL_Z         = [2, 8, 16, 32]
SIZE_LIST     = [256, 512, 1024, 2048, 4096]
TRAIN_N       = 1024
T_STEPS       = 50
NODE_FEAT_DIM = 5
TOPOLOGIES    = ["ba", "er", "ws"]

TOPO_LABELS = {
    "ba": "Barabási–Albert",
    "er": "Erdős–Rényi",
    "ws": "Watts–Strogatz",
}

# Short names for table column headers
TABLE_MODELS = [
    "Persistence", "Mean-Field ODE",
    "Specialist-Low", "Global-GAT",
    "SpectralLSTM", "PA-LSTM", "ZealotTransformer",
]
TABLE_HEADERS = {
    "Persistence":       "Persist.",
    "Mean-Field ODE":    "Mean-Field",
    "Specialist-Low":    "Spec-Low",
    "Global-GAT":        "Global-GAT",
    "SpectralLSTM":      "Spec-LSTM",
    "PA-LSTM":           "PA-LSTM",
    "ZealotTransformer": "ZT",
}

FIG_W_IN = 180 / 25.4


# ═════════════════════════════════════════════════════════════
# Model architectures — must exactly match training scripts
# ═════════════════════════════════════════════════════════════

class LocalGATModel(nn.Module):
    """Specialist-Low / Specialist-High"""
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        out_ch = hidden_dim // 4
        self.conv1 = GATConv(3,          out_ch, heads=4, dropout=dropout)
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.conv3 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln3   = nn.LayerNorm(hidden_dim)
        self.conv4 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.ln4   = nn.LayerNorm(hidden_dim)
        self.out   = GATConv(hidden_dim, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        h  = F.elu(self.conv1(x, edge_index)); h = self.ln1(h)
        h  = F.dropout(h, p=self.dropout, training=self.training)
        h1 = F.elu(self.conv2(h, edge_index)); h = self.ln2(h + h1)
        h  = F.dropout(h, p=self.dropout, training=self.training)
        h2 = F.elu(self.conv3(h, edge_index)); h = self.ln3(h + h2)
        h  = F.dropout(h, p=self.dropout, training=self.training)
        h3 = F.elu(self.conv4(h, edge_index)); h = self.ln4(h + h3)
        h  = F.dropout(h, p=self.dropout, training=self.training)
        return torch.sigmoid(self.out(h, edge_index)).squeeze(-1)


class GlobalGATModel(nn.Module):
    """Global-GAT — no LayerNorm"""
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        out_ch = hidden_dim // 4
        self.conv1 = GATConv(3,          out_ch, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, out_ch, heads=4, dropout=dropout)
        self.out   = GATConv(hidden_dim, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        return torch.sigmoid(self.out(x, edge_index)).squeeze(-1)


class TrajectoryLSTM(nn.Module):
    """SpectralLSTM — universal_lstm.pt"""
    def __init__(self, desc_dim=8, hidden_dim=256, num_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T          = T
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2))
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, descriptors):
        B   = descriptors.shape[0]
        enc = self.encoder(descriptors)
        enc = enc.view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=descriptors.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            pred_t = self.output_head(out.squeeze(1))
            preds.append(pred_t)
            inp = pred_t.detach().unsqueeze(1)
        return torch.cat(preds, dim=1)   # (B, T) in [0,1]


class PALSTMModel(nn.Module):
    """PA-LSTM — pa-lstm.pt, 11D descriptor"""
    def __init__(self, desc_dim=11, hidden_dim=256, num_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T          = T
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_layers * hidden_dim * 2))
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, descriptors):
        B   = descriptors.shape[0]
        enc = self.encoder(descriptors)
        enc = enc.view(B, self.num_layers, self.hidden_dim * 2)
        h0  = enc[:, :, :self.hidden_dim].permute(1, 0, 2).contiguous()
        c0  = enc[:, :, self.hidden_dim:].permute(1, 0, 2).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=descriptors.device)
        preds = []
        h, c = h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            pred_t = self.output_head(out.squeeze(1))
            preds.append(pred_t)
            inp = pred_t.detach().unsqueeze(1)
        return torch.cat(preds, dim=1)


class ZealotTransformer(nn.Module):
    """ZealotTransformer — zealot_transformer.pt"""
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128,
                 nhead=4, num_transformer_layers=3,
                 lstm_hidden=256, lstm_layers=2, T=T_STEPS, dropout=0.0):
        super().__init__()
        self.d_model     = d_model
        self.T           = T
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.LayerNorm(d_model), nn.GELU())
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
            nn.Linear(lstm_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H   = self.node_encoder(X)
        H   = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp  = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1, keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B    = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(
            B, self.lstm_layers, self.lstm_hidden * 2)
        h0  = proj[:, :, :self.lstm_hidden].transpose(0, 1).contiguous()
        c0  = proj[:, :, self.lstm_hidden:].transpose(0, 1).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=ctx.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out)
            preds.append(p)
            inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm  = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0) * 2 - 1   # → [-1,1]


# ═════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════

def _clean(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()}


def load_local_gat(path, device, label):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    hp    = ckpt.get("hyperparams", {})
    model = LocalGATModel(hidden_dim=hp.get("hidden_dim", 256),
                          dropout=hp.get("dropout", 0.1))
    model.load_state_dict(_clean(ckpt["model_state_dict"]), strict=True)
    print(f"  ✓ {label}: {path}")
    return model.to(device).eval()


def load_global_gat(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    hp    = ckpt.get("hyperparams", {})
    model = GlobalGATModel(hidden_dim=hp.get("hidden_dim", 256),
                           dropout=hp.get("dropout", 0.0))
    model.load_state_dict(_clean(ckpt["model_state_dict"]), strict=True)
    print(f"  ✓ Global-GAT: {path}")
    return model.to(device).eval()


def load_spectral_lstm(path, device):
    ckpt       = torch.load(path, map_location=device, weights_only=False)
    hp         = ckpt.get("hyperparams", {})
    norm_stats = ckpt.get("norm_stats", None)
    model      = TrajectoryLSTM(
        desc_dim   = hp.get("desc_dim",   8),
        hidden_dim = hp.get("hidden_dim", 256),
        num_layers = hp.get("num_layers", 2),
        T          = hp.get("T", T_STEPS),
        dropout    = 0.1)
    model.load_state_dict(_clean(ckpt["model_state_dict"]), strict=False)
    print(f"  ✓ SpectralLSTM: {path}")
    if norm_stats is None:
        print("    WARNING: no norm_stats in checkpoint")
    return model.to(device).eval(), norm_stats


def load_pa_lstm(path, device):
    ckpt       = torch.load(path, map_location=device, weights_only=False)
    hp         = ckpt.get("hyperparams", {})
    norm_stats = ckpt.get("norm_stats", None)
    model      = PALSTMModel(
        desc_dim   = hp.get("desc_dim",   11),
        hidden_dim = hp.get("hidden_dim", 256),
        num_layers = hp.get("num_layers", 2),
        T          = hp.get("T", T_STEPS),
        dropout    = 0.1)
    model.load_state_dict(_clean(ckpt["model_state_dict"]), strict=False)
    print(f"  ✓ PA-LSTM: {path}  (desc_dim={hp.get('desc_dim', 11)})")
    if norm_stats is None:
        print("    WARNING: no norm_stats in checkpoint")
    return model.to(device).eval(), norm_stats


def load_zt(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    hp    = ckpt.get("hyperparams", {})
    model = ZealotTransformer(
        node_feat_dim         = hp.get("node_feat_dim", NODE_FEAT_DIM),
        d_model               = hp.get("d_model", 128),
        nhead                 = hp.get("nhead", 4),
        num_transformer_layers= hp.get("num_transformer_layers", 3),
        lstm_hidden           = hp.get("lstm_hidden", 256),
        lstm_layers           = hp.get("lstm_layers", 2),
        T                     = hp.get("T", T_STEPS),
        dropout               = 0.0)
    model.load_state_dict(_clean(ckpt["model_state_dict"]), strict=True)
    best = ckpt.get("best_val_rmse", "?")
    print(f"  ✓ ZealotTransformer: {path}  "
          f"(epoch={ckpt.get('epoch','?')}  best_RMSE={best})")
    return model.to(device).eval()


# ═════════════════════════════════════════════════════════════
# Graph / simulation helpers
# ═════════════════════════════════════════════════════════════

def make_graph(topo, n, m=8, seed=None):
    if topo == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == "er":
        p = min(2 * m / (n - 1), 1.0)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(
                n, p, seed=(seed + attempt if seed is not None else None))
            if nx.is_connected(G):
                break
    elif topo == "ws":
        G = nx.watts_strogatz_graph(n, max(4, 2 * m), p=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topo}")
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G


def place_hubs(G, Z):
    return set(n for n, _ in
               sorted(G.degree(), key=lambda x: x[1], reverse=True)[:Z])


def place_bridges(G, Z):
    btwn = nx.betweenness_centrality(G, normalized=True)
    return set(sorted(btwn, key=btwn.get, reverse=True)[:Z])


def place_random(G, Z, rng):
    return set(int(n) for n in
               rng.choice(list(G.nodes()), size=Z, replace=False))


def get_zealot_set(G, placement, Z, rng):
    if placement == "hub":
        return place_hubs(G, Z)
    elif placement == "bridge":
        return place_bridges(G, Z)
    else:
        return place_random(G, Z, rng)


def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=128, seed=None):
    rng   = np.random.default_rng(seed)
    N_g   = G.number_of_nodes()
    adj   = [list(G.neighbors(i)) for i in range(N_g)]
    is_z  = np.zeros(N_g, dtype=bool)
    for z in zealot_set:
        is_z[z] = True
    non_z = np.where(~is_z)[0]
    trajs = np.zeros((mc_runs, T), dtype=np.float32)
    for r in range(mc_runs):
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


def compute_spectral_desc_8d(G, Z, n, topo):
    """8D descriptor for SpectralLSTM — must match universal_lstm training."""
    deg  = np.array([d for _, d in G.degree()], dtype=np.float64)
    mu   = deg.mean()
    return np.array([
        Z / n,
        _spectral_gap(G),
        mu / n,
        deg.std() / (mu + 1e-8),
        nx.average_clustering(G),
        1.0 if topo == "ba" else 0.0,
        1.0 if topo == "er" else 0.0,
        1.0 if topo == "ws" else 0.0,
    ], dtype=np.float32)


def _fiedler_vector(G):
    """Returns Fiedler vector (N,), normalised to [-1,1]."""
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    if n > 500:
        return deg / (deg.max() + 1e-8)
    try:
        L    = nx.laplacian_matrix(G).astype(float)
        vals, vecs = eigsh(L, k=min(3, n-1), which="SM",
                           tol=1e-2, maxiter=1000)
        f  = vecs[:, np.argsort(vals)[1]]
        mx = np.abs(f).max()
        return (f / mx).astype(np.float32) if mx > 1e-8 else f.astype(np.float32)
    except Exception:
        return deg / (deg.max() + 1e-8)


def compute_pa_desc_11d(G, Z, n, topo, zealot_set):
    """
    11D descriptor for PA-LSTM.
    Must match train_placement_aware_lstm.py EXACTLY:
      9.  hub_score    = mean_deg(zealots) / mean_deg(all)
      10. bridge_score = mean_btwn(zealots) / mean_btwn(all)
      11. fiedler_score = mean|fiedler[zealots]|
    """
    base    = compute_spectral_desc_8d(G, Z, n, topo)
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    mean_deg = degrees.mean()
    zealot_list = list(zealot_set)

    hub_score = float(degrees[zealot_list].mean() / (mean_deg + 1e-8)) \
                if zealot_list else 1.0

    try:
        btwn      = nx.betweenness_centrality(G, normalized=True, endpoints=False)
        btwn_vals = np.array(list(btwn.values()), dtype=np.float64)
        mean_btwn = btwn_vals.mean()
        bridge_score = float(np.mean([btwn[z] for z in zealot_list]) /
                             (mean_btwn + 1e-10)) if zealot_list else 1.0
    except Exception:
        bridge_score = 1.0

    fiedler = _fiedler_vector(G)
    fiedler_score = float(np.abs(fiedler[zealot_list]).mean()) \
                    if zealot_list and len(fiedler) == n else 0.0

    return np.concatenate([base, [hub_score, bridge_score, fiedler_score]]) \
               .astype(np.float32)


def compute_node_features_5d(G, zealot_set):
    """5D node features for ZealotTransformer."""
    N_g   = G.number_of_nodes()
    deg   = np.array([d for _, d in G.degree()], dtype=np.float32)
    deg_n = deg / (deg.max() + 1e-8)
    z_i   = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set:
        z_i[nd] = 1.0
    fiedler = _fiedler_vector(G)
    # PageRank proxy = degree (exact PageRank too slow at N=2048+)
    pr = deg_n.copy()
    try:
        if N_g <= 1000:
            prd = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
            pr  = np.array([prd[i] for i in range(N_g)], dtype=np.float32)
            pr /= (pr.max() + 1e-8)
    except Exception:
        pass
    try:
        cd    = nx.clustering(G)
        clust = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        clust = np.zeros(N_g, dtype=np.float32)
    return np.stack([z_i, deg_n, fiedler, pr, clust], axis=1).astype(np.float32)


def normalize_desc(desc, stats):
    if stats is None:
        return desc
    mean = np.asarray(stats[0], dtype=np.float32)
    std  = np.asarray(stats[1], dtype=np.float32)
    return (desc - mean) / (std + 1e-8)


# ═════════════════════════════════════════════════════════════
# Baselines
# ═════════════════════════════════════════════════════════════

def predict_persistence(m0, T=T_STEPS):
    return np.full(T, m0, dtype=np.float32)


def predict_meanfield(rho_Z, m0, T=T_STEPS, alpha=0.08, beta=2.5):
    def ode(m, t):
        return -alpha * m + beta * rho_Z * (1 - m)
    sol = odeint(ode, [m0], np.linspace(0, T - 1, T))
    return np.clip(sol.squeeze(), -1, 1).astype(np.float32)


# ═════════════════════════════════════════════════════════════
# Rollouts
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout_gat(model, G, zealot_set, device):
    N_g  = G.number_of_nodes()
    deg  = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn   = deg / (deg.max() + 1e-8)
    z_i  = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set:
        z_i[nd] = 1.0
    ops  = np.random.choice([-1., 1.], size=N_g).astype(np.float32)
    ops[z_i == 1] = 1.
    s_n  = (ops + 1) / 2.
    x    = torch.tensor(np.stack([s_n, z_i, dn], axis=1),
                        dtype=torch.float32).to(device)
    ei   = from_networkx(G).edge_index.to(device)
    zm   = torch.tensor(z_i, dtype=torch.float32).to(device)
    preds = []
    for _ in range(T_STEPS):
        probs  = model(x, ei)
        samp   = torch.bernoulli(probs)
        spin   = samp * 2 - 1;  spin[zm == 1] = 1.
        preds.append(spin.mean().item())
        ns = samp.clone(); ns[zm == 1] = 1.
        x  = torch.stack([ns, x[:, 1], x[:, 2]], dim=1)
    return np.array(preds, dtype=np.float32)


@torch.no_grad()
def rollout_spectral_lstm(model, norm_stats, G, Z, n, topo, device):
    desc = compute_spectral_desc_8d(G, Z, n, topo)
    desc = normalize_desc(desc[np.newaxis, :], norm_stats)
    pred = model(torch.tensor(desc, dtype=torch.float32).to(device))
    return (pred.squeeze(0).cpu().numpy() * 2 - 1).astype(np.float32)


@torch.no_grad()
def rollout_pa_lstm(model, norm_stats, G, Z, n, topo, zealot_set, device):
    desc = compute_pa_desc_11d(G, Z, n, topo, zealot_set)
    desc = normalize_desc(desc[np.newaxis, :], norm_stats)
    # Sanity check
    if np.any(np.abs(desc) > 10):
        print(f"    WARNING: PA-LSTM desc out of range "
              f"(max={np.abs(desc).max():.1f})")
    pred = model(torch.tensor(desc, dtype=torch.float32).to(device))
    return (pred.squeeze(0).cpu().numpy() * 2 - 1).astype(np.float32)


@torch.no_grad()
def rollout_zt(model, G, zealot_set, device):
    X   = compute_node_features_5d(G, zealot_set)
    z_m = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zealot_set:
        z_m[nd] = True
    pred = model(
        torch.tensor(X,   dtype=torch.float32).to(device),
        torch.tensor(z_m, dtype=torch.bool).to(device))
    return pred.cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════
# Core evaluation
# ═════════════════════════════════════════════════════════════

def evaluate_cell(loaded, topo, placement, Z, n, mc_runs, val_graphs,
                  device, seed_base=42):
    """
    Returns gt_list (list of arrays) and pred_dict (name → list of arrays).
    """
    gt_list   = []
    pred_dict = {name: [] for name in TABLE_MODELS}

    for g_idx in range(val_graphs):
        seed = seed_base + g_idx * 1000 + abs(hash((topo, placement, Z, n))) % 1000
        rng  = np.random.default_rng(seed)
        G    = make_graph(topo, n, m=max(4, 8 * n // TRAIN_N),
                          seed=int(rng.integers(0, 99999)))
        zs   = get_zealot_set(G, placement, Z, rng)
        gt   = simulate_trajectory(G, zs, T=T_STEPS, mc_runs=mc_runs, seed=seed+1)
        gt_list.append(gt)
        m0  = float(gt[0])
        rho = Z / n

        # Baselines
        pred_dict["Persistence"].append(predict_persistence(m0))
        pred_dict["Mean-Field ODE"].append(predict_meanfield(rho, m0))

        # Neural models
        for name, obj in loaded.items():
            if obj is None:
                pred_dict[name].append(predict_persistence(m0))
                continue
            try:
                if name == "Specialist-Low":
                    p = rollout_gat(obj, G, zs, device)
                elif name == "Global-GAT":
                    p = rollout_gat(obj, G, zs, device)
                elif name == "SpectralLSTM":
                    model_, ns = obj
                    p = rollout_spectral_lstm(model_, ns, G, Z, n, topo, device)
                elif name == "PA-LSTM":
                    model_, ns = obj
                    p = rollout_pa_lstm(model_, ns, G, Z, n, topo, zs, device)
                elif name == "ZealotTransformer":
                    p = rollout_zt(obj, G, zs, device)
                else:
                    p = predict_persistence(m0)
                pred_dict[name].append(p)
            except Exception as e:
                print(f"    [{name}] error: {e}")
                pred_dict[name].append(predict_persistence(m0))

    return gt_list, pred_dict


def rmse_stats(gt_list, pred_list):
    if not gt_list or not pred_list:
        return float("nan"), float("nan")
    per = [float(np.sqrt(np.mean((np.array(p) - np.array(g))**2)))
           for p, g in zip(pred_list, gt_list)]
    return float(np.mean(per)), float(np.std(per))


# ═════════════════════════════════════════════════════════════
# PLAIN TEXT TABLE GENERATION (NO LATEX)
# ═════════════════════════════════════════════════════════════

def _txt_fmt(mean, std, is_best=False):
    if np.isnan(mean):
        return "N/A"
    s = f"{mean:.3f}±{std:.3f}"
    return f"*{s}*" if is_best else s


def _txt_best_in_row(row_vals):
    valids = [v[0] for v in row_vals.values() if not np.isnan(v[0])]
    return min(valids) if valids else float("inf")


def write_text_table_1(results, out_path, models_in_table, val_graphs, mc_runs, N):
    """Plain text table for cross-topology (hub placement)"""
    with open(out_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("TABLE 1: Cross-topology RMSE (hub placement)\n")
        f.write(f"N={N}, val_graphs={val_graphs}, MC runs={mc_runs}\n")
        f.write("=" * 120 + "\n\n")
        
        # Header
        header = f"{'Z':>4}"
        for topo in ["ba", "er", "ws"]:
            for m in models_in_table:
                short = TABLE_HEADERS.get(m, m[:8])
                header += f"  {topo.upper()}_{short:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        for Z in ALL_Z:
            row = f"{Z:>4}"
            for topo in ["ba", "er", "ws"]:
                vals = {}
                for m in models_in_table:
                    mean, _ = results.get((topo, "hub", Z, m), (float("nan"), 0))
                    vals[m] = mean
                best = _txt_best_in_row(vals)
                for m in models_in_table:
                    mean, std = results.get((topo, "hub", Z, m), (float("nan"), float("nan")))
                    if np.isnan(mean):
                        cell = "      N/A      "
                    else:
                        is_best = (abs(mean - best) < 1e-6)
                        cell = _txt_fmt(mean, std, is_best)
                    row += f"  {cell:>14}"
            f.write(row + "\n")
        
        f.write("-" * 120 + "\n")
        f.write("* = best per row per topology\n")
    print(f"  Saved: {out_path}")


def write_text_table_2(results, out_path, models_in_table, val_graphs, mc_runs, N):
    """Plain text table for zealot placement"""
    placements = ["hub", "random", "bridge"]
    pl_labels = {"hub": "Hub", "random": "Random", "bridge": "Bridge"}
    
    with open(out_path, "w") as f:
        f.write("=" * 140 + "\n")
        f.write("TABLE 2: Zealot placement RMSE on Barabási–Albert networks\n")
        f.write(f"N={N}, val_graphs={val_graphs}, MC runs={mc_runs}\n")
        f.write("=" * 140 + "\n\n")
        
        # Header
        header = f"{'Z':>4}"
        for pl in placements:
            for m in models_in_table:
                short = TABLE_HEADERS.get(m, m[:8])
                header += f"  {pl_labels[pl][:6]}_{short:>12}"
        f.write(header + "\n")
        f.write("-" * 140 + "\n")
        
        for Z in ALL_Z:
            row = f"{Z:>4}"
            for pl in placements:
                vals = {}
                for m in models_in_table:
                    mean, _ = results.get(("ba", pl, Z, m), (float("nan"), 0))
                    vals[m] = mean
                best = _txt_best_in_row(vals)
                for m in models_in_table:
                    mean, std = results.get(("ba", pl, Z, m), (float("nan"), float("nan")))
                    if np.isnan(mean):
                        cell = "      N/A      "
                    else:
                        is_best = (abs(mean - best) < 1e-6)
                        cell = _txt_fmt(mean, std, is_best)
                    row += f"  {cell:>14}"
            f.write(row + "\n")
        
        f.write("-" * 140 + "\n")
        f.write("* = best per row per placement\n")
    print(f"  Saved: {out_path}")


def write_text_table_3(results, out_path, models_in_table, val_graphs, mc_runs, Z_sz=8):
    """Plain text table for size generalization"""
    with open(out_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write(f"TABLE 3: Size generalization RMSE (hub placement, Z={Z_sz})\n")
        f.write(f"val_graphs={val_graphs}, MC runs={mc_runs}\n")
        f.write("=" * 120 + "\n\n")
        
        # Header
        header = f"{'N':>6}"
        for topo in ["ba", "er", "ws"]:
            for m in models_in_table:
                short = TABLE_HEADERS.get(m, m[:8])
                header += f"  {topo.upper()}_{short:>12}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        for n_val in SIZE_LIST:
            mark = "†" if n_val == TRAIN_N else ""
            row = f"{n_val:>5}{mark}"
            for topo in ["ba", "er", "ws"]:
                vals = {}
                for m in models_in_table:
                    mean, _ = results.get((topo, n_val, m), (float("nan"), 0))
                    vals[m] = mean
                best = _txt_best_in_row(vals)
                for m in models_in_table:
                    mean, std = results.get((topo, n_val, m), (float("nan"), float("nan")))
                    if np.isnan(mean):
                        cell = "      N/A      "
                    else:
                        is_best = (abs(mean - best) < 1e-6)
                        cell = _txt_fmt(mean, std, is_best)
                    row += f"  {cell:>14}"
            f.write(row + "\n")
        
        f.write("-" * 120 + "\n")
        f.write("* = best per row per topology\n")
        f.write("† = training size (N=1024)\n")
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════
# Trajectory figure
# ═════════════════════════════════════════════════════════════

def plot_trajectories(gt_by_Z, zt_by_Z, out_dir):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 4, figsize=(FIG_W_IN, FIG_W_IN * 0.42),
        sharey=True, constrained_layout=True)
    steps = np.arange(T_STEPS)
    for ax, Z, lab in zip(axes, ALL_Z, ["a", "b", "c", "d"]):
        if gt_by_Z.get(Z):
            gt_a = np.array(gt_by_Z[Z])
            gtm, gts = gt_a.mean(0), gt_a.std(0)
            ax.fill_between(steps, gtm - gts, gtm + gts,
                            color="#1B3A5C", alpha=0.15, lw=0)
            ax.plot(steps, gtm, color="#1B3A5C", lw=1.2, label="Ground Truth")
        if zt_by_Z.get(Z):
            zt_a = np.array(zt_by_Z[Z])
            ztm, zts = zt_a.mean(0), zt_a.std(0)
            ax.fill_between(steps, ztm - zts, ztm + zts,
                            color="#0D9488", alpha=0.18, lw=0)
            ax.plot(steps, ztm, color="#0D9488", lw=1.2, ls="--",
                    label="ZealotTransformer")
        ax.set_xlim(0, T_STEPS - 1); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time step")
        if ax is axes[0]:
            ax.set_ylabel(r"Magnetization $m(t)$")
        ax.set_title(f"$Z = {Z}$", pad=3)
        ax.text(-0.12, 1.04, lab, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, pad=2)
        if ax is axes[-1]:
            ax.legend(loc="lower right", frameon=False)
    base = os.path.join(out_dir, "ba_trajectories_gt_vs_zt")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {base}.pdf / .png")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zt_checkpoint",              type=str, required=True)
    p.add_argument("--spectral_lstm_checkpoint",   type=str, default=None)
    p.add_argument("--pa_lstm_checkpoint",         type=str, default=None)
    p.add_argument("--spec_low_checkpoint",        type=str, default=None)
    p.add_argument("--global_gat_checkpoint",      type=str, default=None)
    p.add_argument("--n",          type=int,   default=1024)
    p.add_argument("--mc_runs",    type=int,   default=64)
    p.add_argument("--val_graphs", type=int,   default=10)
    p.add_argument("--out_dir",    type=str,   default="result")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--eval_sizes", action="store_true",
                   help="Run Table 3: size generalization (all topologies × N)")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tbl_dir = os.path.join(args.out_dir, "tables")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(tbl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────
    print("\nLoading models...")
    loaded = {}

    if args.spec_low_checkpoint and os.path.isfile(args.spec_low_checkpoint):
        loaded["Specialist-Low"] = load_local_gat(
            args.spec_low_checkpoint, device, "Specialist-Low")
    else:
        print("  ⚠ Specialist-Low: not found — will use persistence fallback")
        loaded["Specialist-Low"] = None

    if args.global_gat_checkpoint and os.path.isfile(args.global_gat_checkpoint):
        loaded["Global-GAT"] = load_global_gat(args.global_gat_checkpoint, device)
    else:
        print("  ⚠ Global-GAT: not found — will use persistence fallback")
        loaded["Global-GAT"] = None

    if args.spectral_lstm_checkpoint and os.path.isfile(args.spectral_lstm_checkpoint):
        loaded["SpectralLSTM"] = load_spectral_lstm(
            args.spectral_lstm_checkpoint, device)
    else:
        print("  ⚠ SpectralLSTM: not found")
        loaded["SpectralLSTM"] = None

    if args.pa_lstm_checkpoint and os.path.isfile(args.pa_lstm_checkpoint):
        loaded["PA-LSTM"] = load_pa_lstm(args.pa_lstm_checkpoint, device)
    else:
        print("  ⚠ PA-LSTM: not found")
        loaded["PA-LSTM"] = None

    loaded["ZealotTransformer"] = load_zt(args.zt_checkpoint, device)

    # Models to show in all tables
    models_in_table = [m for m in TABLE_MODELS
                       if m in ("Persistence", "Mean-Field ODE") or
                          loaded.get(m) is not None]
    print(f"\nModels in tables: {models_in_table}")

    t0_all = time.time()

    # ── Experiment 1: Cross-topology (hub placement) ──────────
    print("\n" + "="*60)
    print("EXPERIMENT 1 + TABLE 1: Cross-topology RMSE (hub placement)")
    print("="*60)
    results_topo = {}
    gt_ba_hub    = {Z: [] for Z in ALL_Z}
    zt_ba_hub    = {Z: [] for Z in ALL_Z}

    total = len(TOPOLOGIES) * len(ALL_Z)
    done  = 0
    for topo in TOPOLOGIES:
        for Z in ALL_Z:
            done += 1
            print(f"\n  [{done}/{total}]  topo={topo}  Z={Z}  "
                  f"placement=hub  N={args.n}", flush=True)
            t1 = time.time()
            gt_list, pred_dict = evaluate_cell(
                loaded, topo, "hub", Z, args.n,
                args.mc_runs, args.val_graphs, device, args.seed)
            for name in models_in_table:
                if name in ("Persistence", "Mean-Field ODE"):
                    preds = pred_dict[name]
                else:
                    preds = pred_dict.get(name, [])
                m, s = rmse_stats(gt_list, preds)
                results_topo[(topo, "hub", Z, name)] = (m, s)
                print(f"    {name:<22}  {m:.4f}±{s:.4f}")
            if topo == "ba":
                gt_ba_hub[Z].extend(gt_list)
                if "ZealotTransformer" in pred_dict:
                    zt_ba_hub[Z].extend(pred_dict["ZealotTransformer"])
            print(f"    ({time.time()-t1:.0f}s)", flush=True)

    # ── Experiment 2: Zealot placement (BA, all placements) ───
    print("\n" + "="*60)
    print("EXPERIMENT 2 + TABLE 2: Zealot Placement (BA only)")
    print("="*60)
    placements_to_run = ["hub", "bridge", "random"]
    total = len(placements_to_run) * len(ALL_Z)
    done  = 0
    for pl in placements_to_run:
        for Z in ALL_Z:
            done += 1
            print(f"\n  [{done}/{total}]  topo=ba  Z={Z}  "
                  f"placement={pl}  N={args.n}", flush=True)
            t1 = time.time()
            gt_list, pred_dict = evaluate_cell(
                loaded, "ba", pl, Z, args.n,
                args.mc_runs, args.val_graphs, device, args.seed + 100)
            for name in models_in_table:
                if name in ("Persistence", "Mean-Field ODE"):
                    preds = pred_dict[name]
                else:
                    preds = pred_dict.get(name, [])
                m, s = rmse_stats(gt_list, preds)
                results_topo[("ba", pl, Z, name)] = (m, s)
                print(f"    {name:<22}  {m:.4f}±{s:.4f}")
            print(f"    ({time.time()-t1:.0f}s)", flush=True)

    # Save Tables 1 & 2 (PLAIN TEXT)
    write_text_table_1(results_topo,
                       os.path.join(tbl_dir, "table1_cross_topology.txt"),
                       models_in_table, args.val_graphs, args.mc_runs, args.n)
    write_text_table_2(results_topo,
                       os.path.join(tbl_dir, "table2_zealot_placement.txt"),
                       models_in_table, args.val_graphs, args.mc_runs, args.n)

    # ── Experiment 3: Size generalization ─────────────────────
    results_gen = {}
    if args.eval_sizes:
        print("\n" + "="*60)
        print("EXPERIMENT 3 + TABLE 3: Size Generalization (hub, Z=8)")
        print("ALL TOPOLOGIES: BA, ER, WS")
        print("="*60)
        Z_sz   = 8
        total  = len(TOPOLOGIES) * len(SIZE_LIST)
        done   = 0
        for topo in TOPOLOGIES:
            for n_val in SIZE_LIST:
                done += 1
                mark = " ←TRAIN" if n_val == TRAIN_N else ""
                print(f"\n  [{done}/{total}]  topo={topo}  N={n_val}"
                      f"  Z={Z_sz}{mark}", flush=True)
                t1 = time.time()
                gt_list, pred_dict = evaluate_cell(
                    loaded, topo, "hub", Z_sz, n_val,
                    min(args.mc_runs, 64),
                    min(args.val_graphs, 5),
                    device, args.seed + 200)
                for name in models_in_table:
                    if name in ("Persistence", "Mean-Field ODE"):
                        preds = pred_dict[name]
                    else:
                        preds = pred_dict.get(name, [])
                    m, s = rmse_stats(gt_list, preds)
                    results_gen[(topo, n_val, name)] = (m, s)
                    print(f"    {name:<22}  {m:.4f}±{s:.4f}")
                print(f"    ({time.time()-t1:.0f}s)", flush=True)

        write_text_table_3(results_gen,
                           os.path.join(tbl_dir, "table3_size_generalization.txt"),
                           models_in_table, args.val_graphs, args.mc_runs)

    # ── Trajectory figure ──────────────────────────────────────
    print("\n[Plot] BA trajectory figure (hub placement)...")
    plot_trajectories(gt_ba_hub, zt_ba_hub, fig_dir)

    # ── Save raw JSON ──────────────────────────────────────────
    raw = {}
    for (topo, pl, Z, name), (m, s) in results_topo.items():
        raw[f"{topo}_{pl}_Z{Z}_{name}"] = {"mean": m, "std": s}
    for (topo, n_val, name), (m, s) in results_gen.items():
        raw[f"gen_{topo}_N{n_val}_Z8_{name}"] = {"mean": m, "std": s}
    raw_path = os.path.join(args.out_dir, "results_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  Raw JSON: {raw_path}")

    total_t = time.time() - t0_all
    print(f"\n✓ Done in {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"\nOutputs in {args.out_dir}/:")
    print(f"  tables/table1_cross_topology.txt")
    print(f"  tables/table2_zealot_placement.txt")
    if args.eval_sizes:
        print(f"  tables/table3_size_generalization.txt")
    print(f"  figures/ba_trajectories_gt_vs_zt.pdf")
    print(f"  results_raw.json")


if __name__ == "__main__":
    main()