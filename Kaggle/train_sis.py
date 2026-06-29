#!/usr/bin/env python3
"""
Train ZealotTransformer on the SIS dataset — PAPER RECIPE.
"""

import os, numpy as np, torch, torch.nn as nn, networkx as nx
from torch.utils.data import Dataset, DataLoader

T_STEPS       = 50
NODE_FEAT_DIM = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH     = "/kaggle/working/sis_dataset.npz"
OUT_PATH      = "/kaggle/working/zt_sis.pt"


def compute_node_features_5d(G, seed_set):
    N_g = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn = deg / (deg.max() + 1e-8)
    z_i = np.zeros(N_g, dtype=np.float32)
    for nd in seed_set:
        z_i[nd] = 1.0
    fiedler = dn.copy()
    pr = dn.copy()
    clust = np.zeros(N_g, dtype=np.float32)
    try:
        if N_g <= 2000:
            cd = nx.clustering(G)
            clust = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception:
        pass
    return np.stack([z_i, dn, fiedler, pr, clust], axis=1).astype(np.float32)


def graph_from_record(r):
    N = int(r["N"]); G = nx.Graph(); G.add_nodes_from(range(N))
    G.add_edges_from([tuple(e) for e in r["edges"]])
    return G


class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=256, nhead=4,
                 num_transformer_layers=3, lstm_hidden=256, lstm_layers=2,
                 T=T_STEPS, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.T = T
        self.lstm_hidden = lstm_hidden; self.lstm_layers = lstm_layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_transformer_layers,
                                                 enable_nested_tensor=False)
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
        zp = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1, keepdim=True).clamp(1)
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


def load_samples(npz_path):
    d = np.load(npz_path, allow_pickle=True); recs = d["records"]
    samples = []
    for r in recs:
        r = r.item() if isinstance(r, np.ndarray) else r
        G = graph_from_record(r)
        seeds = set(int(x) for x in r["seeds"])
        X = compute_node_features_5d(G, seeds)
        N = int(r["N"]); zmask = np.zeros(N, dtype=bool)
        for nd in seeds: zmask[nd] = True
        samples.append({"X": X, "z_mask": zmask,
                        "traj": r["traj"].astype(np.float32), "N": N})
    print(f"loaded {len(samples)} SIS samples from {npz_path}")
    return samples


class SISData(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]


def make_size_bucketed_batches(samples, max_nodes_per_batch=32768, shuffle=True):
    by_size = {}
    for i, s in enumerate(samples):
        by_size.setdefault(s["N"], []).append(i)
    batches = []
    for N, idxs in by_size.items():
        if shuffle:
            np.random.shuffle(idxs)
        bs = max(1, max_nodes_per_batch // N)      # adapt batch size to N
        for k in range(0, len(idxs), bs):
            batches.append(idxs[k:k + bs])
    if shuffle:
        np.random.shuffle(batches)
    return batches


class BatchedLoader:
    """Yields collated batches from precomputed index-lists (size-bucketed)."""
    def __init__(self, samples, max_nodes_per_batch=32768, shuffle=True):
        self.samples = samples
        self.cap = max_nodes_per_batch
        self.shuffle = shuffle
    def __iter__(self):
        batches = make_size_bucketed_batches(self.samples, self.cap, self.shuffle)
        for idxs in batches:
            yield collate([self.samples[i] for i in idxs])

def collate(batch):
    maxN = max(s["N"] for s in batch); B = len(batch)
    X = torch.zeros(B, maxN, NODE_FEAT_DIM)
    zmask = torch.zeros(B, maxN, dtype=torch.bool)
    pad = torch.zeros(B, maxN, dtype=torch.bool)
    traj = torch.zeros(B, T_STEPS)
    for i, s in enumerate(batch):
        N = s["N"]
        X[i, :N] = torch.from_numpy(s["X"])
        zmask[i, :N] = torch.from_numpy(s["z_mask"])
        pad[i, :N] = True
        traj[i] = torch.from_numpy(s["traj"])
    return {"X": X, "z_mask": zmask, "padding_mask": pad, "traj": traj}


def traj_loss(pred, target, T):
    w = torch.linspace(1.0, 2.0, T, device=pred.device); w = w / w.sum()
    return torch.sqrt(((pred - target) ** 2 * w.unsqueeze(0)).sum(dim=1).mean() + 1e-8)

def train(npz_path=DATA_PATH, epochs=300, batch_size=64, lr=1e-4, wd=1e-4, warmup=20):
    samples = load_samples(npz_path)
    np.random.shuffle(samples)
    split = int(0.8 * len(samples))
    tr, va = samples[:split], samples[split:]
    model = ZealotTransformer(d_model=256).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    print("params:", sum(p.numel() for p in model.parameters()))

    def lr_at(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)

    tdl = BatchedLoader(tr, max_nodes_per_batch=32768, shuffle=True)
    vdl = BatchedLoader(va, max_nodes_per_batch=32768, shuffle=False)
    best = float("inf")
    for ep in range(1, epochs + 1):                 # full 300, NO early stop
        model.train(); tot = nb = 0
        for b in tdl:
            X = b["X"].to(DEVICE); zm = b["z_mask"].to(DEVICE)
            pm = b["padding_mask"].to(DEVICE); tg = b["traj"].to(DEVICE)
            pred = model.decode_batch(model.encode_graph_batch(X, zm, pm))
            loss = traj_loss(pred, tg, T_STEPS)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for b in vdl:
                X = b["X"].to(DEVICE); zm = b["z_mask"].to(DEVICE)
                pm = b["padding_mask"].to(DEVICE); tg = b["traj"].to(DEVICE)
                pred = model.decode_batch(model.encode_graph_batch(X, zm, pm))
                vt += traj_loss(pred, tg, T_STEPS).item(); vn += 1
        vr = vt / max(1, vn)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d}  train {tot/nb:.4f}  val {vr:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
        if vr < best:
            best = vr; torch.save(model.state_dict(), OUT_PATH)
    print(f"best val RMSE: {best:.4f}  (ran full {epochs} epochs)")
    print(f"saved -> {OUT_PATH}")
    return model


if __name__ == "__main__":
    train()