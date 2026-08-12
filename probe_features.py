#!/usr/bin/env python3
"""

Runs on a laptop CPU. Loads your EXISTING trained ZealotTransformer checkpoint
and feeds it two different feature matrices for the same graphs:

WHAT THIS TELLS YOU
  * How far the predictions move when the feature columns stop being three
    copies of the degree. A large move means the model is highly sensitive to
    these inputs and the retrained model will behave quite differently; a small
    move means the network was mostly ignoring those columns anyway.

WHAT THIS DOES *NOT* TELL YOU
  * It does NOT give the accuracy of the corrected pipeline. The loaded model
    was trained on OLD features, so its RMSE under NEW features is expected to
    be worse -- that is a train/test mismatch, not a verdict on the fix. Only
    retraining answers that.


"""

import os, sys, time, argparse, warnings
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zt_features import compute_node_features   # corrected features

T_STEPS = 50
NODE_FEAT_DIM = 5


# ── OLD features: exactly what training used ─────────────────
def compute_node_features_OLD(G, zealot_set):
    N_g = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    deg_norm = deg / (deg.max() + 1e-8)
    z_i = np.zeros(N_g, dtype=np.float32)
    for nd in zealot_set:
        z_i[int(nd)] = 1.0
    # Fiedler: degree fallback above N=500 (this is the bug being fixed)
    if N_g > 500:
        fiedler = deg_norm.copy()
    else:
        from scipy.sparse.linalg import eigsh
        L = nx.laplacian_matrix(G).astype(float)
        vals, vecs = eigsh(L, k=min(3, N_g - 1), which="SM", tol=1e-2, maxiter=1000)
        fiedler = vecs[:, np.argsort(vals)[1]].astype(np.float32)
        mx = np.abs(fiedler).max()
        if mx > 1e-8:
            fiedler = fiedler / mx
    # PageRank: degree fallback above N=1000
    if N_g > 1000:
        pr_arr = deg_norm.copy()
    else:
        pr = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
        pr_arr = np.array([pr[i] for i in range(N_g)], dtype=np.float32)
        pr_arr /= (pr_arr.max() + 1e-8)
    cd = nx.clustering(G)
    clust = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    return np.stack([z_i, deg_norm, fiedler, pr_arr, clust], axis=1).astype(np.float32)


# ── model ────────────────────────────────────────────────────
class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128, nhead=4,
                 num_transformer_layers=3, lstm_hidden=256, lstm_layers=2,
                 T=T_STEPS, dropout=0.0):
        super().__init__()
        self.d_model = d_model; self.T = T
        self.lstm_hidden = lstm_hidden; self.lstm_layers = lstm_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=num_transformer_layers, enable_nested_tensor=False)
        self.ctx_projector = nn.Sequential(
            nn.Linear(2*d_model, lstm_hidden*2), nn.GELU(),
            nn.Linear(lstm_hidden*2, lstm_layers*lstm_hidden*2))
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H = self.node_encoder(X)
        H = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp = (H*zmk.unsqueeze(-1)).sum(1)/zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H*nzk.unsqueeze(-1)).sum(1)/nzk.sum(1, keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(B, self.lstm_layers, self.lstm_hidden*2)
        h0 = proj[:, :, :self.lstm_hidden].transpose(0, 1).contiguous()
        c0 = proj[:, :, self.lstm_hidden:].transpose(0, 1).contiguous()
        inp = torch.full((B, 1, 1), 0.5, device=ctx.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out, (h, c) = self.lstm(inp, (h, c))
            p = self.output_head(out); preds.append(p); inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0) * 2 - 1     # voter: [-1,1]


def load_zt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt.get("hyperparams", {}) if isinstance(ckpt, dict) else {}
    model = ZealotTransformer(
        node_feat_dim=hp.get("node_feat_dim", NODE_FEAT_DIM),
        d_model=hp.get("d_model", 128),
        nhead=hp.get("nhead", 4),
        num_transformer_layers=hp.get("num_transformer_layers", 3),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS)).to(device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {path}  (d_model={model.d_model})")
    return model


# ── voter MC ground truth ────────────────────────────────────
def simulate_voter(G, zealot_set, mc_runs, T, seed):
    rng = np.random.default_rng(seed)
    N_g = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N_g)]
    is_z = np.zeros(N_g, dtype=bool)
    for z in zealot_set:
        is_z[int(z)] = True
    non_z = np.where(~is_z)[0]
    acc = np.zeros(T, dtype=np.float64)
    for _ in range(mc_runs):
        ops = rng.choice([-1., 1.], size=N_g).astype(np.float32)
        ops[is_z] = 1.
        for t in range(T):
            acc[t] += ops.mean()
            chosen = rng.choice(non_z, size=len(non_z), replace=True)
            for nd in chosen:
                nb = adj[nd]
                if nb:
                    ops[nd] = ops[nb[rng.integers(0, len(nb))]]
            ops[is_z] = 1.
    return (acc / mc_runs).astype(np.float32)


def make_graph(topo, n, seed):
    if topo == "ba":
        return nx.barabasi_albert_graph(n, 8, seed=seed)
    if topo == "er":
        for a in range(10):
            G = nx.erdos_renyi_graph(n, 16/(n-1), seed=seed+a)
            if nx.is_connected(G):
                return G
        return G
    if topo == "ws":
        return nx.watts_strogatz_graph(n, 16, 0.1, seed=seed)
    raise ValueError(topo)


@torch.no_grad()
def predict(model, G, zs, X, device):
    zm = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zs:
        zm[int(nd)] = True
    return model(torch.tensor(X, dtype=torch.float32).to(device),
                 torch.tensor(zm, dtype=torch.bool).to(device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zt_checkpoint", type=str, required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--Z", type=int, default=8)
    ap.add_argument("--graphs", type=int, default=3)
    ap.add_argument("--mc_runs", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = load_zt(args.zt_checkpoint, device)

    print(f"\nN={args.n}  Z={args.Z}  graphs={args.graphs}  MC runs={args.mc_runs}")
    print("(model was TRAINED on OLD features -- see header note)\n")
    print(f"{'topo':>5} {'RMSE(old)':>10} {'RMSE(new)':>10} "
          f"{'|pred diff| mean':>17} {'max':>8}")

    rows = []
    for topo in ("ba", "er", "ws"):
        for g in range(args.graphs):
            seed = 1000 + g*97
            G = make_graph(topo, args.n, seed)
            deg = np.array([d for _, d in G.degree()])
            zs = np.argsort(deg)[-args.Z:]           # hub placement

            gt = simulate_voter(G, zs, args.mc_runs, T_STEPS, seed+1)

            X_old = compute_node_features_OLD(G, zs)
            X_new = compute_node_features(G, zs)

            p_old = predict(model, G, zs, X_old, device)
            p_new = predict(model, G, zs, X_new, device)

            r_old = float(np.sqrt(np.mean((p_old - gt)**2)))
            r_new = float(np.sqrt(np.mean((p_new - gt)**2)))
            d = np.abs(p_old - p_new)
            rows.append((topo, r_old, r_new, d.mean(), d.max()))
            print(f"{topo:>5} {r_old:10.4f} {r_new:10.4f} "
                  f"{d.mean():17.4f} {d.max():8.4f}")

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print("\n" + "-"*56)
    print(f"mean RMSE with OLD features : {arr[:,0].mean():.4f}")
    print(f"mean RMSE with NEW features : {arr[:,1].mean():.4f}")
    print(f"mean |prediction difference|: {arr[:,2].mean():.4f}")
    print(f"max  |prediction difference|: {arr[:,3].max():.4f}")
    print("-"*56)
    print("\nHow to read this:")
    print(" * small prediction difference -> the network was largely ignoring")
    print("   the duplicated columns; retraining should land near your")
    print("   published numbers.")
    print(" * large prediction difference -> those columns carry real weight,")
    print("   so expect the retrained model's tables to shift noticeably.")
    print(" * RMSE(new) being worse is EXPECTED here and is not evidence")
    print("   against the fix: this model never saw the corrected features.")


if __name__ == "__main__":
    main()
