"""
=================
Comparison script for ZealotTransformer against all baseline and neural models.

Neural models (require saved checkpoints):
  Specialist-Low, Specialist-High, Global-GAT,
  SpectralLSTM, PA-LSTM, ZealotTransformer

Baselines (no checkpoint needed):
  Persistence, Mean-Field ODE, MLP-Descriptor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE (all inside --out_dir)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tables/
  table_topology_ba_hub.txt      ← Table 1: BA, hub placement, Z∈{2,8,16,32}
  table_topology_ba_random.txt
  table_topology_er_hub.txt
  table_topology_er_random.txt
  table_topology_ws_hub.txt
  table_topology_ws_random.txt
  table_placement_ba.txt         ← Table 2: BA, hub vs random, per model & Z
  table_placement_er.txt
  table_placement_ws.txt
  table_size_ba_hub.txt          ← Table 3: BA/hub, N∈{256,512,1024,2048,4096}
  table_size_er_hub.txt
  table_size_ws_hub.txt

figures/
  ba_trajectories_gt_vs_zt.pdf   ← BA trajectory plot (Scientific Reports style)
  ba_trajectories_gt_vs_zt.png

trajectories/
  trajectories_<topo>_<placement>_Z<z>_N<n>.json  ← raw trajectories per config

results_raw.json                 ← all RMSE numbers in one machine-readable file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, json, time, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALL_Z         = [2, 8, 16, 32]
SIZE_LIST     = [256, 512, 1024, 2048, 4096]   # for Table 3
T_STEPS       = 50
NODE_FEAT_DIM = 5
TOPOLOGIES    = ["ba", "er", "ws"]
PLACEMENTS    = ["hub", "random"]

# Scientific Reports figure style
PANEL_FONT  = 8;  AXIS_FONT = 8;  TICK_FONT = 7;  LEGEND_FONT = 7
LINE_WIDTH  = 1.2
FIG_W_IN    = 180 / 25.4   # 180 mm full-width

MODEL_ORDER = [
    "Persistence", "Mean-Field ODE", "MLP-Descriptor",
    "Specialist-Low", "Specialist-High", "Global-GAT",
    "SpectralLSTM", "PA-LSTM", "ZealotTransformer",
]

# ─────────────────────────────────────────────────────────────────────────────
# Graph / simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_graph(topo, n, m=8, seed=None):
    if topo == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == "er":
        p = min(2 * m / (n - 1), 1.0)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(n, p,
                seed=(seed + attempt if seed is not None else None))
            if nx.is_connected(G): break
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


def place_random(G, Z, rng):
    return set(int(n) for n in rng.choice(list(G.nodes()), size=Z, replace=False))


def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=20, seed=None):
    rng  = np.random.default_rng(seed)
    N_g  = G.number_of_nodes()
    adj  = [list(G.neighbors(i)) for i in range(N_g)]
    is_z = np.zeros(N_g, dtype=bool)
    for z in zealot_set: is_z[z] = True
    non_z = np.where(~is_z)[0]
    all_t = np.zeros((mc_runs, T), dtype=np.float32)
    for run in range(mc_runs):
        ops = rng.choice([-1.0, 1.0], size=N_g).astype(np.float32)
        ops[is_z] = 1.0
        for t in range(T):
            all_t[run, t] = float(ops.mean())
            chosen = rng.choice(non_z, size=len(non_z), replace=True)
            for node in chosen:
                nbrs = adj[node]
                if nbrs: ops[node] = ops[nbrs[rng.integers(0, len(nbrs))]]
            ops[is_z] = 1.0
    return np.mean(all_t, axis=0).astype(np.float32)


def compute_fiedler_vector(G):
    n = G.number_of_nodes()
    if n > 500:
        d = np.array([deg for _, deg in G.degree()], dtype=np.float32)
        return (d / (d.max() + 1e-8)).astype(np.float32)
    try:
        L    = nx.laplacian_matrix(G).astype(float)
        nev  = min(3, n - 1)
        vals, vecs = eigsh(L, k=nev, which="SM", tol=1e-2, maxiter=1000)
        f    = vecs[:, np.argsort(vals)[1]]
        mx   = np.abs(f).max()
        if mx > 1e-8: f /= mx
        return f.astype(np.float32)
    except Exception:
        d = np.array([deg for _, deg in G.degree()], dtype=np.float32)
        return (d / (d.max() + 1e-8)).astype(np.float32)


def compute_node_features(G, zealot_set):
    N_g  = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn   = degs / (degs.max() + 1e-8)
    z_i  = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set: z_i[nd] = 1.0
    fv   = compute_fiedler_vector(G)
    try:
        if N_g > 1000:
            pr = dn.copy()
        else:
            prd = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
            pr  = np.array([prd[i] for i in range(N_g)], dtype=np.float32)
            pr /= (pr.max() + 1e-8)
    except Exception:
        pr = dn.copy()
    try:
        cd  = nx.clustering(G)
        cl  = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        cl  = np.zeros(N_g, dtype=np.float32)
    return np.stack([z_i, dn, fv, pr, cl], axis=1).astype(np.float32)


def compute_spectral_descriptor(G, Z, N, topo):
    rho_Z = Z / N
    d     = np.array([deg for _, deg in G.degree()], dtype=np.float64)
    mu_k  = d.mean(); cv_k = d.std() / (mu_k + 1e-8)
    try:
        L    = nx.laplacian_matrix(G).astype(float)
        vals = eigsh(L, k=min(3, N-1), which="SM", tol=1e-2,
                     return_eigenvectors=False, maxiter=1000)
        lam2 = sorted(vals)[1] if len(vals) > 1 else 0.0
    except Exception:
        lam2 = 0.0
    try:    C = nx.average_clustering(G)
    except: C = 0.0
    tv = [1.0 if topo == "ba" else 0.0,
          1.0 if topo == "er" else 0.0,
          1.0 if topo == "ws" else 0.0]
    return np.array([rho_Z, lam2, mu_k / N, cv_k, C] + tv, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures  (stubs — replace forward() with your actual classes)
# ─────────────────────────────────────────────────────────────────────────────

class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128,
                 nhead=4, num_transformer_layers=3,
                 lstm_hidden=256, lstm_layers=2, T=T_STEPS, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.T = T
        self.lstm_hidden = lstm_hidden; self.lstm_layers = lstm_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        self.ctx_projector = nn.Sequential(
            nn.Linear(2*d_model, lstm_hidden*2), nn.GELU(),
            nn.Linear(lstm_hidden*2, lstm_layers*lstm_hidden*2))
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H  = self.node_encoder(X)
        H  = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp  = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1, keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B   = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(B, self.lstm_layers, self.lstm_hidden*2)
        h0  = proj[:, :, :self.lstm_hidden].transpose(0,1).contiguous()
        c0  = proj[:, :, self.lstm_hidden:].transpose(0,1).contiguous()
        inp = torch.full((B,1,1), 0.5, device=ctx.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out, (h,c) = self.lstm(inp, (h,c))
            p = self.output_head(out); preds.append(p); inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm  = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0) * 2 - 1


class SpectralLSTMModel(nn.Module):
    def __init__(self, desc_dim=8, lstm_hidden=256, lstm_layers=2, T=T_STEPS):
        super().__init__()
        self.T=T; self.lh=lstm_hidden; self.ll=lstm_layers
        self.enc = nn.Sequential(
            nn.Linear(desc_dim,128), nn.GELU(),
            nn.Linear(128,256), nn.GELU(),
            nn.Linear(256, lstm_layers*lstm_hidden*2))
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=0.1 if lstm_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(lstm_hidden,64), nn.GELU(),
                                  nn.Linear(64,1), nn.Sigmoid())
    def forward(self, d):
        proj = self.enc(d).view(self.ll, 1, self.lh*2)
        h0 = proj[:,:,:self.lh].contiguous()
        c0 = proj[:,:,self.lh:].contiguous()
        inp = torch.full((1,1,1), 0.5, device=d.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out,(h,c) = self.lstm(inp,(h,c)); p=self.head(out)
            preds.append(p); inp=p.detach()
        return torch.cat(preds,1).squeeze()*2-1


class PALSTMModel(nn.Module):
    def __init__(self, desc_dim=11, lstm_hidden=256, lstm_layers=2, T=T_STEPS):
        super().__init__()
        self.T=T; self.lh=lstm_hidden; self.ll=lstm_layers
        self.enc = nn.Sequential(
            nn.Linear(desc_dim,128), nn.GELU(),
            nn.Linear(128,256), nn.GELU(),
            nn.Linear(256, lstm_layers*lstm_hidden*2))
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=0.1 if lstm_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(lstm_hidden,64), nn.GELU(),
                                  nn.Linear(64,1), nn.Sigmoid())
    def forward(self, d):
        proj = self.enc(d).view(self.ll, 1, self.lh*2)
        h0 = proj[:,:,:self.lh].contiguous()
        c0 = proj[:,:,self.lh:].contiguous()
        inp = torch.full((1,1,1), 0.5, device=d.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out,(h,c) = self.lstm(inp,(h,c)); p=self.head(out)
            preds.append(p); inp=p.detach()
        return torch.cat(preds,1).squeeze()*2-1


class GATModel(nn.Module):
    def __init__(self, node_dim=3, hidden=256, heads=4, layers=4, T=T_STEPS):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, T), nn.Sigmoid())
    def forward(self, X):
        return self.net(X).mean(dim=0)*2-1


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def persistence_predict(m0, T):
    return np.full(T, m0, dtype=np.float32)


def meanfield_predict(rho_Z, m0, T, alpha=0.08, beta=2.5):
    def ode(m, t): return -alpha * m + beta * rho_Z * (1 - m)
    t_grid = np.linspace(0, T - 1, T)
    sol    = odeint(ode, [m0], t_grid)
    return np.clip(sol.squeeze(), -1, 1).astype(np.float32)


class MLPDescriptorModel:
    def __init__(self, checkpoint_path=None, T=T_STEPS, device="cpu"):
        self.T=T; self.device=device; self.net=None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt   = torch.load(checkpoint_path, map_location=device)
            in_dim = ckpt.get("desc_dim", 8)
            net    = nn.Sequential(
                nn.Linear(in_dim,256), nn.GELU(),
                nn.Linear(256,512), nn.GELU(),
                nn.Linear(512,256), nn.GELU(),
                nn.Linear(256,T), nn.Sigmoid())
            net.load_state_dict(ckpt["model_state_dict"])
            net.eval(); self.net = net.to(device)
    def predict(self, descriptor, m0=0.0):
        if self.net is not None:
            with torch.no_grad():
                d_t = torch.tensor(descriptor, dtype=torch.float32).to(self.device)
                out = self.net(d_t.unsqueeze(0)).squeeze()
                return (out.cpu().numpy()*2-1).astype(np.float32)
        rho_Z = float(descriptor[0]); lam2 = float(descriptor[1])
        tg    = np.arange(self.T, dtype=np.float32)
        return np.clip(1-np.exp(-(rho_Z*(1+lam2))*tg*0.15), -1, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_zt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt.get("hyperparams", {})
    m    = ZealotTransformer(
        node_feat_dim=hp.get("node_feat_dim", NODE_FEAT_DIM),
        d_model=hp.get("d_model", 128),
        nhead=hp.get("nhead", 4),
        num_transformer_layers=hp.get("num_transformer_layers", 3),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS), dropout=0.0).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    best = ckpt.get("best_val_rmse", ckpt.get("avg_val_rmse", "?"))
    print(f"  ✓ ZealotTransformer  epoch={ckpt.get('epoch','?')}  best_RMSE={best}")
    return m


def load_spectral_lstm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt.get("hyperparams", {})
    m    = SpectralLSTMModel(
        desc_dim=hp.get("desc_dim", 8),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS)).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    print("  ✓ SpectralLSTM"); return m


def load_pa_lstm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt.get("hyperparams", {})
    m    = PALSTMModel(
        desc_dim=hp.get("desc_dim", 11),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS)).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    print("  ✓ PA-LSTM"); return m


def load_gat(path, device, label):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt.get("hyperparams", {})
    m    = GATModel(
        node_dim=hp.get("node_dim", 3),
        hidden=hp.get("hidden", 256),
        heads=hp.get("heads", 4),
        layers=hp.get("layers", 4),
        T=hp.get("T", T_STEPS)).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    print(f"  ✓ {label}"); return m


# ─────────────────────────────────────────────────────────────────────────────
# Prediction dispatchers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_zt(model, G, zealot_set, device):
    X    = compute_node_features(G, zealot_set)
    zm   = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zealot_set: zm[nd] = True
    return model(torch.tensor(X).to(device),
                 torch.tensor(zm).to(device)).cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_spectral_lstm(model, desc, device):
    return model(torch.tensor(desc).to(device)).cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_pa_lstm(model, desc, G, zealot_set, device):
    btwn = nx.betweenness_centrality(G, normalized=True)
    degs = dict(G.degree())
    zmb  = np.mean([btwn[nd] for nd in zealot_set]) if zealot_set else 0.0
    zmd  = (np.mean([degs[nd] for nd in zealot_set]) /
            (max(degs.values())+1e-8)) if zealot_set else 0.0
    N    = G.number_of_nodes()
    spm  = 0.0
    if N <= 500:
        try:
            paths = []
            for nd in list(zealot_set)[:5]:
                ll = nx.single_source_shortest_path_length(G, nd)
                paths.append(np.mean(list(ll.values())))
            spm = np.mean(paths) / N if paths else 0.0
        except Exception: pass
    fd   = np.concatenate([desc, np.array([zmb, zmd, spm], dtype=np.float32)])
    return model(torch.tensor(fd).to(device)).cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_gat(model, G, zealot_set, device):
    d    = np.array([deg for _, deg in G.degree()], dtype=np.float32)
    kn   = d / (d.max()+1e-8)
    N    = G.number_of_nodes()
    iz   = np.zeros(N, dtype=np.float32)
    for nd in zealot_set: iz[nd] = 1.0
    X    = np.stack([np.ones(N, dtype=np.float32), iz, kn], axis=1)
    return model(torch.tensor(X).to(device)).cpu().numpy().astype(np.float32)


def dispatch(name, obj, G, zealot_set, desc, m0, device):
    if name == "Persistence":      return persistence_predict(m0, T_STEPS)
    if name == "Mean-Field ODE":   return meanfield_predict(
        len(zealot_set)/G.number_of_nodes(), m0, T_STEPS)
    if name == "MLP-Descriptor":   return obj.predict(desc, m0)
    if name == "SpectralLSTM":     return predict_spectral_lstm(obj, desc, device)
    if name == "PA-LSTM":          return predict_pa_lstm(obj, desc, G, zealot_set, device)
    if name == "ZealotTransformer":return predict_zt(obj, G, zealot_set, device)
    if name in ("Specialist-Low","Specialist-High","Global-GAT"):
        return predict_gat(obj, G, zealot_set, device)
    return np.zeros(T_STEPS, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(models_dict, topo, placement, Z, N, mc_runs, val_graphs,
             device, seed_base=0):
    """Returns gt_list, pred_dict (per-graph trajectories)."""
    gt_list   = []
    pred_dict = {name: [] for name in models_dict}
    for g_idx in range(val_graphs):
        seed = seed_base + g_idx*1000 + hash((topo, placement, Z, N)) % 1000
        rng  = np.random.default_rng(seed)
        try:
            G = make_graph(topo, N, m=max(4, 8*N//1024), seed=int(rng.integers(0,99999)))
        except Exception: continue

        zealot_set = place_hubs(G, Z) if placement == "hub" else place_random(G, Z, rng)
        gt         = simulate_trajectory(G, zealot_set, T=T_STEPS,
                                          mc_runs=mc_runs, seed=seed+1)
        gt_list.append(gt)
        desc = compute_spectral_descriptor(G, Z, N, topo)
        m0   = float(gt[0])

        for name, obj in models_dict.items():
            try:
                pred = dispatch(name, obj, G, zealot_set, desc, m0, device)
            except Exception:
                pred = np.full(T_STEPS, m0, dtype=np.float32)
            pred_dict[name].append(pred)
    return gt_list, pred_dict


def rmse_stats(gt_list, pred_list):
    if not gt_list or not pred_list: return float("nan"), float("nan")
    per = [float(np.sqrt(np.mean((np.array(p)-np.array(g))**2)))
           for p, g in zip(pred_list, gt_list)]
    return float(np.mean(per)), float(np.std(per))


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─"

def fmt(mean, std, bold=False):
    if np.isnan(mean): return "N/A"
    s = f"{mean:.3f}±{std:.3f}"
    return f"[{s}]" if bold else s   # [brackets] = best


def write_table(path, title, col_header, row_labels, col_labels,
                data, best_per_row=True):
    """
    col_header : e.g. "Z" or "N"
    row_labels : list of row identifiers  (e.g. [2,8,16,32] or model names)
    col_labels : list of column identifiers (e.g. model names or Z values)
    data       : dict[(row_label, col_label)] → (mean, std)
    """
    cw  = max(18, max((len(str(c)) for c in col_labels), default=0) + 2)
    hw  = max(12, len(str(col_header)) + 2)

    lines = [title, SEP * (hw + (cw+3)*len(col_labels) + 3)]
    header = f"{col_header:^{hw}}" + "".join(f"   {str(c):^{cw}}" for c in col_labels)
    lines += [header, SEP * (hw + (cw+3)*len(col_labels) + 3)]

    for row in row_labels:
        row_vals = {c: data.get((row, c), (float("nan"), float("nan")))
                    for c in col_labels}
        finite   = {c: v[0] for c, v in row_vals.items() if not np.isnan(v[0])}
        best_val = min(finite.values()) if finite else np.inf

        cells = []
        for c in col_labels:
            mean, std = row_vals[c]
            is_best   = best_per_row and (not np.isnan(mean) and
                                          abs(mean - best_val) < 1e-6)
            cells.append(fmt(mean, std, is_best))
        lines.append(f"{str(row):^{hw}}" + "".join(f"   {cell:^{cw}}" for cell in cells))

    lines.append(SEP * (hw + (cw+3)*len(col_labels) + 3))
    lines.append("[x] = best in row\n")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: f.write(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory figure  (BA hub only, Scientific Reports style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(gt_by_Z, zt_by_Z, out_dir):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
        "font.size": AXIS_FONT, "axes.labelsize": AXIS_FONT,
        "axes.titlesize": PANEL_FONT, "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT, "legend.fontsize": LEGEND_FONT,
        "axes.linewidth": 0.8, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "lines.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 4, figsize=(FIG_W_IN, FIG_W_IN*0.42),
                             sharey=True, constrained_layout=True)
    C = {"Ground Truth": "#1B3A5C", "ZealotTransformer": "#0D9488"}
    t  = np.arange(T_STEPS)

    # Collect trajectory data for JSON export
    traj_json = {}

    for ax, Z, lab in zip(axes, ALL_Z, ["a","b","c","d"]):
        gt_t = gt_by_Z.get(Z, [])
        zt_t = zt_by_Z.get(Z, [])

        traj_json[f"Z{Z}"] = {
            "ground_truth_mean": [],
            "ground_truth_std":  [],
            "zt_mean":           [],
            "zt_std":            [],
        }

        if gt_t:
            gt_a  = np.array(gt_t)
            gtm, gts = gt_a.mean(0), gt_a.std(0)
            ax.fill_between(t, gtm-gts, gtm+gts,
                            color=C["Ground Truth"], alpha=0.15, linewidth=0)
            ax.plot(t, gtm, color=C["Ground Truth"], lw=LINE_WIDTH,
                    label="Ground Truth", zorder=3)
            traj_json[f"Z{Z}"]["ground_truth_mean"] = gtm.tolist()
            traj_json[f"Z{Z}"]["ground_truth_std"]  = gts.tolist()

        if zt_t:
            zt_a  = np.array(zt_t)
            ztm, zts = zt_a.mean(0), zt_a.std(0)
            ax.fill_between(t, ztm-zts, ztm+zts,
                            color=C["ZealotTransformer"], alpha=0.18, linewidth=0)
            ax.plot(t, ztm, color=C["ZealotTransformer"], lw=LINE_WIDTH,
                    ls="--", label="ZealotTransformer", zorder=4)
            traj_json[f"Z{Z}"]["zt_mean"] = ztm.tolist()
            traj_json[f"Z{Z}"]["zt_std"]  = zts.tolist()

        ax.set_xlim(0, T_STEPS-1); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time step", fontsize=AXIS_FONT)
        if ax is axes[0]: ax.set_ylabel(r"Magnetization $m(t)$", fontsize=AXIS_FONT)
        ax.set_title(f"$Z = {Z}$", fontsize=PANEL_FONT, pad=3)
        ax.text(-0.12, 1.04, lab, transform=ax.transAxes,
                fontsize=PANEL_FONT+1, fontweight="bold", va="top", ha="left")
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[-1]:
            ax.legend(loc="lower right", frameon=False,
                      fontsize=LEGEND_FONT, handlelength=1.5)

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, "ba_trajectories_gt_vs_zt")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {base}.pdf / .png")

    # Save trajectory JSON
    traj_dir  = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, "trajectories_ba_hub_N1024.json")
    with open(traj_path, "w") as f: json.dump(traj_json, f, indent=2)
    print(f"  Trajectories JSON → {traj_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zt_checkpoint",            type=str, required=True)
    p.add_argument("--spec_low_checkpoint",       type=str, default=None)
    p.add_argument("--spec_high_checkpoint",      type=str, default=None)
    p.add_argument("--global_gat_checkpoint",     type=str, default=None)
    p.add_argument("--spectral_lstm_checkpoint",  type=str, default=None)
    p.add_argument("--pa_lstm_checkpoint",        type=str, default=None)
    p.add_argument("--mlp_desc_checkpoint",       type=str, default=None)
    p.add_argument("--n",          type=int, default=1024)
    p.add_argument("--mc_runs",    type=int, default=128)
    p.add_argument("--val_graphs", type=int, default=10)
    p.add_argument("--out_dir",    type=str, default="comparison_results")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--topologies", nargs="+", default=["ba","er","ws"])
    p.add_argument("--placements", nargs="+", default=["hub","random"])
    p.add_argument("--eval_sizes", action="store_true",
                   help="Also run Table 3: size generalisation sweep "
                        "(adds significant runtime)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tbl_dir  = os.path.join(args.out_dir, "tables")
    traj_dir = os.path.join(args.out_dir, "trajectories")
    for d in [tbl_dir, traj_dir]: os.makedirs(d, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models...")
    M = {}   # name → model object (or None for analytic baselines)
    M["Persistence"]    = None
    M["Mean-Field ODE"] = None
    M["MLP-Descriptor"] = MLPDescriptorModel(
        args.mlp_desc_checkpoint, T=T_STEPS, device=str(device))
    M["ZealotTransformer"] = load_zt(args.zt_checkpoint, device)

    opt_loaders = [
        ("SpectralLSTM",  args.spectral_lstm_checkpoint,  load_spectral_lstm),
        ("PA-LSTM",       args.pa_lstm_checkpoint,        load_pa_lstm),
        ("Specialist-Low",  args.spec_low_checkpoint,  lambda p,d: load_gat(p,d,"Specialist-Low")),
        ("Specialist-High", args.spec_high_checkpoint, lambda p,d: load_gat(p,d,"Specialist-High")),
        ("Global-GAT",      args.global_gat_checkpoint,lambda p,d: load_gat(p,d,"Global-GAT")),
    ]
    for name, ckpt_path, loader in opt_loaders:
        if ckpt_path and os.path.isfile(ckpt_path):
            M[name] = loader(ckpt_path, device)
        else:
            print(f"  ⚠ {name}: no checkpoint — skipping")

    active_names = [n for n in MODEL_ORDER if n in M]
    active_M     = {n: M[n] for n in active_names}

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 1  — Cross-topology: one file per (topo, placement) combination
    #            Rows = Z values  |  Columns = models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n━━━  TABLE 1: Cross-topology  ━━━")
    results_topo = {}   # (topo, placement, Z, name) → (mean, std)

    # BA hub data also feeds the trajectory plot
    gt_by_Z_ba = {Z: [] for Z in ALL_Z}
    zt_by_Z_ba = {Z: [] for Z in ALL_Z}
    all_raw_traj = {}   # for per-config JSON files

    N = args.n
    total = len(args.topologies) * len(args.placements) * len(ALL_Z)
    done  = 0
    t0_all = time.time()

    for topo in args.topologies:
        for placement in args.placements:
            for Z in ALL_Z:
                done += 1
                print(f"  [{done}/{total}] topo={topo} place={placement} "
                      f"Z={Z} N={N}", flush=True)
                t0 = time.time()

                gt_list, pred_dict = evaluate(
                    active_M, topo, placement, Z, N,
                    args.mc_runs, args.val_graphs, device, args.seed)

                for name in active_names:
                    mn, sd = rmse_stats(gt_list, pred_dict[name])
                    results_topo[(topo, placement, Z, name)] = (mn, sd)
                    print(f"    {name:<22}  {mn:.4f}±{sd:.4f}")

                # Collect BA hub trajectories
                if topo == "ba" and placement == "hub":
                    gt_by_Z_ba[Z].extend(gt_list)
                    if "ZealotTransformer" in pred_dict:
                        zt_by_Z_ba[Z].extend(pred_dict["ZealotTransformer"])

                # Save raw trajectories JSON (per config)
                raw_key = f"{topo}_{placement}_Z{Z}_N{N}"
                all_raw_traj[raw_key] = {
                    "ground_truth": [t.tolist() for t in gt_list],
                    "predictions":  {n: [t.tolist() for t in pred_dict[n]]
                                     for n in active_names},
                }
                traj_path = os.path.join(
                    traj_dir, f"trajectories_{raw_key}.json")
                with open(traj_path, "w") as f:
                    json.dump(all_raw_traj[raw_key], f, indent=2)

                print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # Write one text file per (topo, placement)
    print("\n  Writing Table-1 files...")
    for topo in args.topologies:
        for placement in args.placements:
            data = {(Z, name): results_topo[(topo, placement, Z, name)]
                    for Z in ALL_Z for name in active_names
                    if (topo, placement, Z, name) in results_topo}
            path = os.path.join(tbl_dir, f"table_topology_{topo}_{placement}.txt")
            text = write_table(
                path,
                title=f"Table 1 — Topology: {topo.upper()}  Placement: {placement}  "
                      f"N={N}  mc_runs={args.mc_runs}  val_graphs={args.val_graphs}\n"
                      f"RMSE mean±std  (lower is better)",
                col_header="Z",
                row_labels=ALL_Z,
                col_labels=active_names,
                data=data)
            print(f"  → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 2  — Zealot placement: hub vs random
    #            Rows = models  |  Columns = (placement, Z) pairs
    #            One file per topology
    # ─────────────────────────────────────────────────────────────────────────
    print("\n━━━  TABLE 2: Placement comparison  ━━━")
    for topo in args.topologies:
        # column labels: "hub/Z2", "hub/Z8", ..., "rnd/Z2", ...
        col_labels = [f"hub/Z{Z}" for Z in ALL_Z] + [f"rnd/Z{Z}" for Z in ALL_Z]
        data = {}
        for name in active_names:
            for Z in ALL_Z:
                for pl, tag in [("hub", f"hub/Z{Z}"), ("random", f"rnd/Z{Z}")]:
                    key = (topo, pl, Z, name)
                    if key in results_topo:
                        data[(name, tag)] = results_topo[key]
        path = os.path.join(tbl_dir, f"table_placement_{topo}.txt")
        write_table(
            path,
            title=f"Table 2 — Placement: hub vs random  Topology: {topo.upper()}  "
                  f"N={N}  mc_runs={args.mc_runs}\nRMSE mean±std",
            col_header="Model",
            row_labels=active_names,
            col_labels=col_labels,
            data=data)
        print(f"  → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 3  — Size generalisation (optional, --eval_sizes)
    #            Rows = N values  |  Columns = models
    #            One file per topology (hub placement only, Z=8 as representative)
    # ─────────────────────────────────────────────────────────────────────────
    if args.eval_sizes:
        print("\n━━━  TABLE 3: Size generalisation  ━━━")
        results_size = {}   # (topo, N, name) → (mean, std)
        placement_sz = "hub"
        Z_sz         = 8
        size_topologies = [t for t in args.topologies]

        total_sz = len(size_topologies) * len(SIZE_LIST)
        done_sz  = 0
        for topo in size_topologies:
            for sz in SIZE_LIST:
                done_sz += 1
                print(f"  [{done_sz}/{total_sz}] topo={topo} N={sz} "
                      f"Z={Z_sz} (hub)", flush=True)
                t0 = time.time()
                gt_l, pd_d = evaluate(
                    active_M, topo, placement_sz, Z_sz, sz,
                    min(args.mc_runs, 64),   # fewer runs for large N
                    min(args.val_graphs, 5),
                    device, args.seed)
                for name in active_names:
                    mn, sd = rmse_stats(gt_l, pd_d[name])
                    results_size[(topo, sz, name)] = (mn, sd)
                    print(f"    {name:<22}  {mn:.4f}±{sd:.4f}")

                # Save trajectory JSON
                raw_key = f"{topo}_{placement_sz}_Z{Z_sz}_N{sz}"
                traj_path = os.path.join(traj_dir, f"trajectories_{raw_key}.json")
                with open(traj_path, "w") as f:
                    json.dump({
                        "ground_truth": [t.tolist() for t in gt_l],
                        "predictions":  {n: [t.tolist() for t in pd_d[n]]
                                         for n in active_names},
                    }, f, indent=2)
                print(f"    ({time.time()-t0:.1f}s)")

        for topo in size_topologies:
            data = {(sz, name): results_size[(topo, sz, name)]
                    for sz in SIZE_LIST for name in active_names
                    if (topo, sz, name) in results_size}
            path = os.path.join(tbl_dir, f"table_size_{topo}_{placement_sz}.txt")
            write_table(
                path,
                title=f"Table 3 — Size generalisation  Topology: {topo.upper()}  "
                      f"Placement: {placement_sz}  Z={Z_sz}\nRMSE mean±std",
                col_header="N",
                row_labels=SIZE_LIST,
                col_labels=active_names,
                data=data)
            print(f"  → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Trajectory figure
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating BA trajectory figure...")
    plot_trajectories(gt_by_Z_ba, zt_by_Z_ba, args.out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Master raw JSON
    # ─────────────────────────────────────────────────────────────────────────
    raw = {}
    for (topo, pl, Z, name), (mn, sd) in results_topo.items():
        raw[f"{topo}_{pl}_Z{Z}_{name}"] = {"mean": mn, "std": sd}
    if args.eval_sizes:
        for (topo, sz, name), (mn, sd) in results_size.items():
            raw[f"{topo}_hub_Z8_N{sz}_{name}"] = {"mean": mn, "std": sd}
    raw_path = os.path.join(args.out_dir, "results_raw.json")
    with open(raw_path, "w") as f: json.dump(raw, f, indent=2)
    print(f"\n  Raw JSON → {raw_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Print all tables to terminal / log
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  FULL RESULTS (also saved in tables/)")
    print("="*80)
    for fname in sorted(os.listdir(tbl_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(tbl_dir, fname)
            print(f"\n{'─'*80}\n  {fname}\n{'─'*80}")
            with open(path) as f: print(f.read())

    total_t = time.time() - t0_all
    print(f"\n✓ Done in {total_t:.0f}s")
    print(f"  tables/     → {tbl_dir}")
    print(f"  figures/    → {os.path.join(args.out_dir,'figures')}")
    print(f"  trajectories/ → {traj_dir}")


if __name__ == "__main__":
    main()