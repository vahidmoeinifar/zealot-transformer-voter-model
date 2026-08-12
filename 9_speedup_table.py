"""
speedup.py
==========
Wall-clock speedup of ZealotTransformer inference vs Monte Carlo simulation.

Matches the methodology of the paper:
  - MC runs on single-threaded CPU  (OMP_NUM_THREADS=1)
  - ZealotTransformer measured on both GPU and CPU
  - Median of 5 timing repeats per configuration
  - Configurations: N ∈ {256, 512, 1024, 2048, 4096}  ×  Z ∈ {8, 32}
                    topology ∈ {ba, er, ws}  (N=1024, Z=8)
  - T=50 time steps, 20 MC runs per ground-truth estimate

Outputs
-------
  speedup_results.json       raw timing + speedup numbers
  speedup_chart.pdf / .png   bar chart (Scientific Reports style, no title)
  speedup_table.txt          plain-text table for paper

"""

import os, json, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

# ── constants (must match training) ──────────────────────────────────────────
T_STEPS       = 50
MC_RUNS       = 20
NODE_FEAT_DIM = 5
N_REPEATS     = 5          # timing repeats → take median

AXIS_FONT = 8; TICK_FONT = 7; LEGEND_FONT = 7; LINE_WIDTH = 1.2
FIG_W_IN  = 180 / 25.4

# ── graph / simulation helpers ────────────────────────────────────────────────

def make_graph(topo, n, m=8, seed=None):
    if topo == "ba":
        return nx.barabasi_albert_graph(n, m, seed=seed)
    elif topo == "er":
        p = min(2 * m / (n - 1), 1.0)
        for attempt in range(10):
            G = nx.erdos_renyi_graph(n, p,
                seed=(seed + attempt if seed is not None else None))
            if nx.is_connected(G): return G
    elif topo == "ws":
        return nx.watts_strogatz_graph(n, max(4, 2*m), p=0.1, seed=seed)
    raise ValueError(topo)


def place_hubs(G, Z):
    return set(n for n, _ in
               sorted(G.degree(), key=lambda x: x[1], reverse=True)[:Z])


def compute_fiedler_vector(G):
    n = G.number_of_nodes()
    if n > 500:
        d = np.array([deg for _, deg in G.degree()], dtype=np.float32)
        return (d / (d.max() + 1e-8)).astype(np.float32)
    try:
        L = nx.laplacian_matrix(G).astype(float)
        nev = min(3, n - 1)
        vals, vecs = eigsh(L, k=nev, which="SM", tol=1e-2, maxiter=1000)
        f = vecs[:, np.argsort(vals)[1]]
        mx = np.abs(f).max()
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


# ── ZealotTransformer (exact copy from training script) ──────────────────────

class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128,
                 nhead=4, num_transformer_layers=3,
                 lstm_hidden=256, lstm_layers=2, T=T_STEPS, dropout=0.0):
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
                            dropout=0.0)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H  = self.node_encoder(X)
        H  = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp  = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1, keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1, keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B    = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(B, self.lstm_layers, self.lstm_hidden*2)
        h0   = proj[:, :, :self.lstm_hidden].transpose(0,1).contiguous()
        c0   = proj[:, :, self.lstm_hidden:].transpose(0,1).contiguous()
        inp  = torch.full((B,1,1), 0.5, device=ctx.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out, (h,c) = self.lstm(inp, (h,c))
            p = self.output_head(out); preds.append(p); inp = p.detach()
        return torch.cat(preds, dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm  = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    hp   = ckpt.get("hyperparams", {})
    m    = ZealotTransformer(
        node_feat_dim=hp.get("node_feat_dim", NODE_FEAT_DIM),
        d_model=hp.get("d_model", 128),
        nhead=hp.get("nhead", 4),
        num_transformer_layers=hp.get("num_transformer_layers", 3),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS), dropout=0.0).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


# ── timing helpers ────────────────────────────────────────────────────────────

def time_mc(G, zealot_set, T, mc_runs, n_repeats):
    """Time Monte Carlo simulation (CPU, single-threaded)."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    N_g  = G.number_of_nodes()
    adj  = [list(G.neighbors(i)) for i in range(N_g)]
    is_z = np.zeros(N_g, dtype=bool)
    for z in zealot_set: is_z[z] = True
    non_z = np.where(~is_z)[0]

    def _run():
        rng = np.random.default_rng(0)
        all_t = np.zeros((mc_runs, T), dtype=np.float32)
        for run in range(mc_runs):
            ops = rng.choice([-1.0,1.0], size=N_g).astype(np.float32)
            ops[is_z] = 1.0
            for t in range(T):
                all_t[run,t] = float(ops.mean())
                chosen = rng.choice(non_z, size=len(non_z), replace=True)
                for node in chosen:
                    nbrs = adj[node]
                    if nbrs: ops[node] = ops[nbrs[rng.integers(0,len(nbrs))]]
                ops[is_z] = 1.0
        return np.mean(all_t, axis=0)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _run()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


@torch.no_grad()
def time_zt(model, G, zealot_set, device, n_repeats):
    """Time ZealotTransformer inference on the given device."""
    X    = compute_node_features(G, zealot_set)
    zm   = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zealot_set: zm[nd] = True
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)
    z_t  = torch.tensor(zm, dtype=torch.bool).to(device)

    # Warm-up (important for GPU / JIT)
    for _ in range(3):
        model(X_t, z_t)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(X_t, z_t)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_speedup(records, out_dir):

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
        "font.size": AXIS_FONT, "axes.labelsize": AXIS_FONT,
        "xtick.labelsize": TICK_FONT, "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT, "axes.linewidth": 0.8,
        "lines.linewidth": LINE_WIDTH, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    labels       = [r["label"] for r in records]
    cpu_speedup  = [r["cpu_speedup"]  for r in records]
    gpu_speedup  = [r.get("gpu_speedup", None) for r in records]
    has_gpu      = any(g is not None for g in gpu_speedup)

    x   = np.arange(len(labels))
    w   = 0.35 if has_gpu else 0.6
    fig_h = FIG_W_IN * 0.45
    fig, ax = plt.subplots(figsize=(FIG_W_IN, fig_h), constrained_layout=True)

    bars_cpu = ax.bar(x - (w/2 if has_gpu else 0), cpu_speedup, w,
                      color="#1B3A5C", label="CPU inference")
    if has_gpu:
        gpu_vals = [g if g is not None else 0 for g in gpu_speedup]
        bars_gpu = ax.bar(x + w/2, gpu_vals, w,
                          color="#0D9488", label="GPU inference")

    # Value labels on bars
    for bar in bars_cpu:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.0f}×", ha="center", va="bottom",
                fontsize=TICK_FONT, color="#1B3A5C")
    if has_gpu:
        for bar in bars_gpu:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.0f}×", ha="center", va="bottom",
                        fontsize=TICK_FONT, color="#0D9488")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FONT)
    ax.set_ylabel("Speedup over MC (×)", fontsize=AXIS_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="#E2E8F0")
    ax.set_axisbelow(True)
    if has_gpu:
        ax.legend(frameon=False, fontsize=LEGEND_FONT, loc="upper left")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "speedup_chart")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {base}.pdf / .png")


# ── table ─────────────────────────────────────────────────────────────────────

def write_speedup_table(records, out_dir):
    SEP = "─"
    has_gpu = any(r.get("gpu_speedup") is not None for r in records)

    cols = ["Configuration", "MC CPU (s)", "ZT CPU (ms)", "CPU Speedup"]
    if has_gpu:
        cols += ["ZT GPU (ms)", "GPU Speedup"]

    cw = 18
    lw = 26
    total_w = lw + cw * len(cols[1:]) + 3 * len(cols[1:])

    lines = [
        "Speedup: ZealotTransformer vs Monte Carlo",
        f"MC: single-threaded CPU  |  median of {N_REPEATS} repeats  |  "
        f"T={T_STEPS}  mc_runs={MC_RUNS}",
        SEP * total_w,
        f"{'Configuration':<{lw}}"
        + "".join(f"   {c:^{cw}}" for c in cols[1:]),
        SEP * total_w,
    ]
    for r in records:
        gpu_str = (f"   {r['zt_gpu_ms']:^{cw}.2f}   {r['gpu_speedup']:^{cw}.0f}×"
                   if has_gpu and r.get("gpu_speedup") is not None else "")
        lines.append(
            f"{r['label']:<{lw}}"
            f"   {r['mc_cpu_s']:^{cw}.3f}"
            f"   {r['zt_cpu_ms']:^{cw}.2f}"
            f"   {r['cpu_speedup']:^{cw}.0f}×"
            + gpu_str
        )
    lines.append(SEP * total_w)
    text = "\n".join(lines)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "speedup_table.txt")
    with open(path, "w") as f: f.write(text + "\n")
    print(f"  Table  → {path}")
    return text


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zt_checkpoint", type=str, required=True)
    p.add_argument("--out_dir",       type=str, default="speedup_results")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--n_repeats",     type=int, default=N_REPEATS)
    p.add_argument("--mc_runs",       type=int, default=MC_RUNS)
    p.add_argument("--T",             type=int, default=T_STEPS)
    return p.parse_args()


def main():
    args   = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    has_gpu    = torch.cuda.is_available()

    print(f"\nLoading ZealotTransformer...")
    model_gpu = load_model(args.zt_checkpoint, gpu_device)
    model_cpu = load_model(args.zt_checkpoint, cpu_device)

    if has_gpu:
        gp = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gp.name}  VRAM={gp.total_memory/1e9:.1f} GB")
    print()

    # ── Benchmark configurations ──────────────────────────────────────────────
    # Table rows match the paper: size sweep + topology sweep
    size_configs = [
        {"label": f"N={n}, Z=8, BA",  "topo": "ba", "n": n, "Z": 8}
        for n in [256, 512, 1024, 2048, 4096]
    ] + [
        {"label": "N=1024, Z=32, BA", "topo": "ba", "n": 1024, "Z": 32},
        {"label": "N=1024, Z=8, ER",  "topo": "er", "n": 1024, "Z": 8},
        {"label": "N=1024, Z=8, WS",  "topo": "ws", "n": 1024, "Z": 8},
    ]

    records = []
    for cfg in size_configs:
        n, Z, topo = cfg["n"], cfg["Z"], cfg["topo"]
        print(f"  {cfg['label']}", flush=True)

        m  = max(4, 8 * n // 1024)
        G  = make_graph(topo, n, m=m, seed=args.seed)
        zs = place_hubs(G, Z)

        # MC timing
        t_mc = time_mc(G, zs, args.T, args.mc_runs, args.n_repeats)

        # ZT CPU timing
        t_cpu = time_zt(model_cpu, G, zs, cpu_device, args.n_repeats)

        rec = {
            "label":       cfg["label"],
            "topo":        topo,
            "n":           n,
            "Z":           Z,
            "mc_cpu_s":    t_mc,
            "zt_cpu_ms":   t_cpu * 1000,
            "cpu_speedup": t_mc / t_cpu,
        }

        if has_gpu:
            t_gpu = time_zt(model_gpu, G, zs, gpu_device, args.n_repeats)
            rec["zt_gpu_ms"]  = t_gpu * 1000
            rec["gpu_speedup"] = t_mc / t_gpu
            print(f"    MC CPU={t_mc:.3f}s  ZT CPU={t_cpu*1e3:.1f}ms "
                  f"({t_mc/t_cpu:.0f}×)  ZT GPU={t_gpu*1e3:.1f}ms "
                  f"({t_mc/t_gpu:.0f}×)")
        else:
            print(f"    MC CPU={t_mc:.3f}s  ZT CPU={t_cpu*1e3:.1f}ms "
                  f"({t_mc/t_cpu:.0f}×)")

        records.append(rec)

    # Outputs
    os.makedirs(args.out_dir, exist_ok=True)

    plot_speedup(records, args.out_dir)
    print(write_speedup_table(records, args.out_dir))

    json_path = os.path.join(args.out_dir, "speedup_results.json")
    with open(json_path, "w") as f: json.dump(records, f, indent=2)
    print(f"  JSON   → {json_path}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()