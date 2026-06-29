#!/usr/bin/env python3
"""
SIS evaluation — SINGLE CELL.
"""

import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.sparse.linalg import eigsh
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
T_STEPS       = 50
NODE_FEAT_DIM = 5
MC_RUNS       = 128
VAL_GRAPHS    = 10

ALL_Z         = [2, 8, 16, 32]
TOPOLOGIES    = ["ba", "er", "ws"]
N_BASE        = 1024
SIZE_LIST     = [256, 512, 1024, 2048]
OOD_N         = [4096, 8192]
OOD_Z         = [64, 128]
PLACEMENTS    = ["hub", "random", "bridge"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# SIS simulation
# ─────────────────────────────────────────────────────────────
def simulate_sis_once(G, seeds, beta, gamma, T, rng):
    N = G.number_of_nodes()
    neighbors = [np.array(list(G.neighbors(n)), dtype=np.int64) for n in range(N)]
    infected = np.zeros(N, dtype=bool)
    seed_mask = np.zeros(N, dtype=bool)
    if len(seeds) > 0:
        seed_mask[np.asarray(seeds, dtype=np.int64)] = True
    infected[seed_mask] = True
    i_traj = np.empty(T, dtype=np.float64)
    for t in range(T):
        i_traj[t] = infected.mean()
        new_state = infected.copy()
        for u in np.where(~infected)[0]:
            nb = neighbors[u]
            if nb.size == 0:
                continue
            inf_nb = np.count_nonzero(infected[nb])
            if inf_nb == 0:
                continue
            if rng.random() < 1.0 - (1.0 - beta) ** inf_nb:
                new_state[u] = True
        for v in np.where(infected & ~seed_mask)[0]:
            if rng.random() < gamma:
                new_state[v] = False
        infected = new_state
    return i_traj


def simulate_sis_mc(G, seeds, beta, gamma, T, n_runs, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.zeros(T, dtype=np.float64)
    for _ in range(n_runs):
        acc += simulate_sis_once(G, seeds, beta, gamma, T, rng)
    return acc / n_runs


def pick_seeds(G, Z, strategy, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    N = G.number_of_nodes()
    if strategy == "hub":
        deg = np.array([d for _, d in G.degree()])
        return np.argsort(deg)[-Z:]
    if strategy == "random":
        return rng.choice(N, size=Z, replace=False)
    if strategy == "bridge":
        bc = nx.betweenness_centrality(G)
        order = sorted(bc, key=bc.get)
        return np.array(order[-Z:], dtype=np.int64)
    raise ValueError(strategy)


# ─────────────────────────────────────────────────────────────
# Graph generation
# ─────────────────────────────────────────────────────────────
def make_graph(topo, n, m=8, seed=None):
    rng_np = np.random.default_rng(seed)
    if topo == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == "er":
        p = 16 / (n - 1)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(n, p, seed=(seed + attempt if seed is not None else None))
            if nx.is_connected(G):
                break
    elif topo == "ws":
        G = nx.watts_strogatz_graph(n, max(4, 2 * m), p=0.1, seed=seed)
    elif topo == "rgg":
        r = np.sqrt(2 * m / (n * np.pi))
        for attempt in range(15):
            pos = {i: (float(rng_np.random()), float(rng_np.random())) for i in range(n)}
            G = nx.random_geometric_graph(n, r, pos=pos)
            if nx.is_connected(G):
                break
            r *= 1.05
        G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(topo)
    if not nx.is_connected(G):
        G = nx.convert_node_labels_to_integers(
            G.subgraph(max(nx.connected_components(G), key=len)).copy())
    return G


def get_beta(G, gamma=0.5, ratio=1.5):
    try:
        A = nx.adjacency_matrix(G).astype(float)
        lam = float(eigsh(A, k=1, which='LM', return_eigenvectors=False, tol=1e-2)[0])
    except Exception:
        deg = np.array([d for _, d in G.degree()], dtype=float)
        lam = float(deg.max())
    return ratio * gamma / max(lam, 1e-8)


# ─────────────────────────────────────────────────────────────
# Node features (5D) — same as voter model
# ─────────────────────────────────────────────────────────────
def _fiedler_vector(G):
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    return deg / (deg.max() + 1e-8)


def compute_node_features_5d(G, zealot_set):
    N_g = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn = deg / (deg.max() + 1e-8)
    z_i = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set:
        z_i[nd] = 1.0
    fiedler = _fiedler_vector(G)
    pr = dn.copy()
    clust = np.zeros(N_g, dtype=np.float32)
    try:
        if N_g <= 2000:
            cd = nx.clustering(G)
            clust = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        pass
    return np.stack([z_i, dn, fiedler, pr, clust], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=256, nhead=4,
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
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers,
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
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out)
            preds.append(p)
            inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0)   # already [0,1] for SIS


def load_zt(path, device):
    model = ZealotTransformer(d_model=256, T=T_STEPS).to(device)
    if os.path.isfile(path):
        sd = torch.load(path, map_location=device)
        sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"  Loaded ZT from {path}")
    else:
        raise FileNotFoundError(f"ZT checkpoint not found at {path}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────
def predict_persistence(i0):
    return np.full(T_STEPS, i0, dtype=np.float32)


def predict_meanfield_sis(G, seeds, beta, gamma):
    N = G.number_of_nodes()
    kbar = 2 * G.number_of_edges() / N
    rho_seed = len(seeds) / N
    i = rho_seed
    traj = np.empty(T_STEPS)
    for t in range(T_STEPS):
        traj[t] = i
        i_free = max(i - rho_seed, 0.0)
        di = beta * kbar * (1.0 - i) * i - gamma * i_free
        i = min(max(i + di, 0.0), 1.0)
        i = max(i, rho_seed)
    return traj.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Core cell evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_cell(zt_model, topo, placement, Z, n, mc_runs, seed_base,
                  gamma=0.5, beta_ratio=1.5):
    preds_zt, preds_mf, preds_pe, gt_list = [], [], [], []
    for g_idx in range(VAL_GRAPHS):
        seed = seed_base + g_idx * 1000
        G = make_graph(topo, n, seed=seed)
        beta = get_beta(G, gamma=gamma, ratio=beta_ratio)
        seeds = pick_seeds(G, Z, placement, rng=np.random.default_rng(seed))
        gt = simulate_sis_mc(G, seeds, beta, gamma, T_STEPS, mc_runs, seed=seed + 1)
        gt_list.append(gt)
        i0 = float(gt[0])

        try:
            zset = set(int(s) for s in seeds)
            X = compute_node_features_5d(G, zset)
            z_mask = np.zeros(G.number_of_nodes(), dtype=bool)
            for nd in zset:
                z_mask[nd] = True
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            z_t = torch.tensor(z_mask, dtype=torch.bool).to(DEVICE)
            with torch.no_grad():
                pred_zt = zt_model(X_t, z_t).cpu().numpy()
            preds_zt.append(pred_zt)
        except Exception as e:
            print(f"    ZT error: {e}")
            preds_zt.append(predict_persistence(i0))

        preds_mf.append(predict_meanfield_sis(G, seeds, beta, gamma))
        preds_pe.append(predict_persistence(i0))

    results = {}
    for name, preds in [("Persistence", preds_pe),
                        ("MeanField", preds_mf),
                        ("ZealotTransformer", preds_zt)]:
        rmses = [float(np.sqrt(np.mean((p - g) ** 2))) for p, g in zip(preds, gt_list)]
        results[name] = (float(np.mean(rmses)), float(np.std(rmses)))
    return results


# ─────────────────────────────────────────────────────────────
# Experiment runners — COMPACT (2 small tables)
# ─────────────────────────────────────────────────────────────
def run_main(zt_model):
    """Table 1 — in-distribution, N=1024, Z=8: BA hub, BA random, ER random, WS random."""
    print("\n=== TABLE 1: Main (N=1024, Z=8, 128 MC) ===")
    plan = [("ba", "hub"), ("ba", "random"), ("er", "random"), ("ws", "random")]
    results = {}
    for topo, pl in plan:
        res = evaluate_cell(zt_model, topo, pl, 8, N_BASE, MC_RUNS, 1000)
        results[(topo, pl)] = res
        print(f"{topo} {pl}: ZT={res['ZealotTransformer'][0]:.4f} ±{res['ZealotTransformer'][1]:.4f}")
    return results


def run_ood(zt_model):
    """Table 2 — OOD, hub, Z=8: N=256, N=8192, RGG(N=1024). 32 MC."""
    print("\n=== TABLE 2: OOD (hub, Z=8, 32 MC) ===")
    plan = [("ba", 256), ("ba", 8192), ("rgg", 1024)]
    results = {}
    for topo, n in plan:
        res = evaluate_cell(zt_model, topo, "hub", 8, n, 32, 4000)
        results[(topo, n)] = res
        print(f"{topo} N={n}: ZT={res['ZealotTransformer'][0]:.4f} ±{res['ZealotTransformer'][1]:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# Output: text tables + Excel
# ─────────────────────────────────────────────────────────────
def _cell(res, m):
    mean, std = res.get(m, (float("nan"), float("nan")))
    return "   N/A   " if np.isnan(mean) else f"{mean:.3f}±{std:.3f}"


def write_all(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    order = ["Persistence", "MeanField", "ZealotTransformer"]
    rows_for_excel = []

    def dump(title, meta, keys, label_fn, store):
        lines = ["=" * 78, title, "=" * 78, meta, "-" * 78,
                 f"{'Config':<22}" + "".join(f"{m:>18}" for m in order), "-" * 78]
        for k in keys:
            res = store.get(k, {})
            lines.append(f"{label_fn(k):<22}" + "".join(f"{_cell(res, m):>18}" for m in order))
            rows_for_excel.append({"Table": title, "Config": label_fn(k),
                                   **{m: _cell(res, m) for m in order}})
        lines.append("-" * 78)
        print("\n".join(lines))
        return "\n".join(lines)

    txt = []
    txt.append(dump("Table 1 — SIS Main (N=1024, Z=8)",
                    "MC runs: 128   Val graphs: 10   beta=1.5*gamma/lambda_max, gamma=0.5",
                    [("ba", "hub"), ("ba", "random"), ("er", "random"), ("ws", "random")],
                    lambda k: f"{k[0].upper()} {k[1]}", results["main"]))
    txt.append(dump("Table 2 — SIS OOD (hub, Z=8)",
                    "MC runs: 32   Val graphs: 10   RGG never seen in training",
                    [("ba", 256), ("ba", 8192), ("rgg", 1024)],
                    lambda k: f"{k[0].upper()} N={k[1]}", results["ood"]))

    with open(os.path.join(out_dir, "sis_all_tables.txt"), "w") as f:
        f.write("\n\n".join(txt))
    try:
        import pandas as pd
        pd.DataFrame(rows_for_excel).to_excel(os.path.join(out_dir, "sis_results.xlsx"), index=False)
        print(f"\nSaved Excel: {os.path.join(out_dir, 'sis_results.xlsx')}")
    except Exception as e:
        print("excel save skipped:", e)
    print(f"Saved text: {os.path.join(out_dir, 'sis_all_tables.txt')}")


# ─────────────────────────────────────────────────────────────
# RUN  (edit WEIGHTS path if needed)
# ─────────────────────────────────────────────────────────────
WEIGHTS = "..."

print("Device:", DEVICE)
zt_model = load_zt(WEIGHTS, DEVICE)

results = {}
results["main"] = run_main(zt_model)
results["ood"]  = run_ood(zt_model)

write_all(results, "/kaggle/working/result_sis")
print("\nDONE")