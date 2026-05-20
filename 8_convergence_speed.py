"""
convergence_speed.py
====================
Plots and analyses convergence speed of ZealotTransformer across topologies
and zealot counts.

Two complementary analyses:

  1. Training convergence  (from training_log.json)
     - Validation RMSE vs epoch for all tracked configurations
     - Saved as: convergence_training.pdf / .png

  2. Opinion-dynamics convergence  (MC simulation + ZealotTransformer prediction)
     - First time step t where predicted/true m(t) >= threshold (default 0.80)
     - Plotted as t_consensus vs Z for BA / ER / WS
     - Saved as: convergence_speed.pdf / .png  (matches paper Fig style)
     - Also saves: convergence_speed.json

Usage
-----
  python convergence_speed.py --zt_checkpoint saved_models/zealot_transformer.pt
  python convergence_speed.py --zt_checkpoint ... --log_path saved_models/training_log.json
"""

import os, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
T_STEPS       = 50
MC_RUNS       = 128
NODE_FEAT_DIM = 5
ALL_Z         = [2, 8, 16, 32]
TOPOLOGIES    = ["ba", "er", "ws"]
THRESHOLD     = 0.80      # consensus threshold for convergence-time plot
VAL_GRAPHS    = 20        # graphs per (topo, Z) cell

AXIS_FONT = 8; TICK_FONT = 7; LEGEND_FONT = 7; LINE_WIDTH = 1.2
FIG_W_IN  = 180 / 25.4

TOPO_STYLE = {
    "ba": {"color": "#1B3A5C", "marker": "o", "ls": "-",  "label": "BA"},
    "er": {"color": "#0D9488", "marker": "s", "ls": "--", "label": "ER"},
    "ws": {"color": "#F59E0B", "marker": "^", "ls": ":",  "label": "WS"},
}

# ── graph / feature helpers ───────────────────────────────────────────────────

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
        if N_g > 1000: pr = dn.copy()
        else:
            prd = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
            pr  = np.array([prd[i] for i in range(N_g)], dtype=np.float32)
            pr /= (pr.max() + 1e-8)
    except Exception: pr = dn.copy()
    try:
        cd = nx.clustering(G)
        cl = np.array([cd[i] for i in range(N_g)], dtype=np.float32)
    except Exception: cl = np.zeros(N_g, dtype=np.float32)
    return np.stack([z_i, dn, fv, pr, cl], axis=1).astype(np.float32)


def simulate_trajectory(G, zealot_set, T=T_STEPS, mc_runs=MC_RUNS, seed=None):
    rng  = np.random.default_rng(seed)
    N_g  = G.number_of_nodes()
    adj  = [list(G.neighbors(i)) for i in range(N_g)]
    is_z = np.zeros(N_g, dtype=bool)
    for z in zealot_set: is_z[z] = True
    non_z = np.where(~is_z)[0]
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
    return np.mean(all_t, axis=0).astype(np.float32)


# ── ZealotTransformer ─────────────────────────────────────────────────────────

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
            nn.Linear(lstm_hidden,64), nn.GELU(), nn.Linear(64,1), nn.Sigmoid())

    def encode_graph_batch(self, X, zm, pm):
        H   = self.node_encoder(X)
        H   = self.transformer(H, src_key_padding_mask=~pm)
        zmk = zm & pm
        zp  = (H * zmk.unsqueeze(-1)).sum(1) / zmk.sum(1,keepdim=True).clamp(1)
        nzk = (~zm) & pm
        np_ = (H * nzk.unsqueeze(-1)).sum(1) / nzk.sum(1,keepdim=True).clamp(1)
        return torch.cat([zp, np_], dim=-1)

    def decode_batch(self, ctx):
        B    = ctx.shape[0]
        proj = self.ctx_projector(ctx).view(B, self.lstm_layers, self.lstm_hidden*2)
        h0   = proj[:,:,:self.lstm_hidden].transpose(0,1).contiguous()
        c0   = proj[:,:,self.lstm_hidden:].transpose(0,1).contiguous()
        inp  = torch.full((B,1,1), 0.5, device=ctx.device)
        preds, h, c = [], h0, c0
        for _ in range(self.T):
            out,(h,c) = self.lstm(inp,(h,c)); p=self.output_head(out)
            preds.append(p); inp=p.detach()
        return torch.cat(preds,dim=1).squeeze(-1)

    def forward(self, X, zm):
        pm  = torch.ones(1, X.shape[0], dtype=torch.bool, device=X.device)
        ctx = self.encode_graph_batch(X.unsqueeze(0), zm.unsqueeze(0), pm)
        return self.decode_batch(ctx).squeeze(0) * 2 - 1


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp   = ckpt.get("hyperparams", {})
    m    = ZealotTransformer(
        node_feat_dim=hp.get("node_feat_dim", NODE_FEAT_DIM),
        d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
        num_transformer_layers=hp.get("num_transformer_layers", 3),
        lstm_hidden=hp.get("lstm_hidden", 256),
        lstm_layers=hp.get("lstm_layers", 2),
        T=hp.get("T", T_STEPS), dropout=0.0).to(device)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def predict_zt(model, G, zealot_set, device):
    X  = compute_node_features(G, zealot_set)
    zm = np.zeros(G.number_of_nodes(), dtype=bool)
    for nd in zealot_set: zm[nd] = True
    return model(torch.tensor(X).to(device),
                 torch.tensor(zm).to(device)).cpu().numpy().astype(np.float32)


# ── convergence-time helper ───────────────────────────────────────────────────

def first_crossing(traj, threshold):
    """First t where traj[t] >= threshold; returns T if never reached."""
    hits = np.where(np.array(traj) >= threshold)[0]
    return int(hits[0]) if len(hits) > 0 else len(traj)


# ── Plot 1: training convergence from log ─────────────────────────────────────

def plot_training_convergence(log_path, out_dir):
    if not os.path.isfile(log_path):
        print(f"  ⚠ training_log.json not found at {log_path} — skipping panel 1")
        return

    with open(log_path) as f:
        log = json.load(f)

    # log entries: [{epoch, avg_val_rmse, best_val_rmse, val_results:{key:{mean,std}}}]
    epochs   = [e["epoch"]        for e in log if "avg_val_rmse" in e]
    avg_rmse = [e["avg_val_rmse"] for e in log if "avg_val_rmse" in e]
    best     = [e["best_val_rmse"]for e in log if "best_val_rmse" in e]

    # Per-config curves (from val_results dict)
    config_curves = {}
    for entry in log:
        if "val_results" not in entry or not entry["val_results"]: continue
        ep = entry["epoch"]
        for key, vals in entry["val_results"].items():
            config_curves.setdefault(key, {"epochs":[], "mean":[], "std":[]})
            config_curves[key]["epochs"].append(ep)
            config_curves[key]["mean"].append(vals["mean"])
            config_curves[key]["std"].append(vals["std"])

    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":AXIS_FONT,"axes.labelsize":AXIS_FONT,
        "xtick.labelsize":TICK_FONT,"ytick.labelsize":TICK_FONT,
        "legend.fontsize":LEGEND_FONT,"axes.linewidth":0.8,
        "pdf.fonttype":42,"ps.fonttype":42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_W_IN * 0.42),
                             constrained_layout=True)

    # Left: overall avg + best
    ax = axes[0]
    ax.plot(epochs, avg_rmse, color="#1B3A5C", lw=LINE_WIDTH, label="Avg RMSE")
    ax.plot(epochs, best,     color="#0D9488", lw=LINE_WIDTH,
            ls="--", label="Best RMSE")
    ax.set_xlabel("Epoch", fontsize=AXIS_FONT)
    ax.set_ylabel("Validation RMSE", fontsize=AXIS_FONT)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes,
            fontsize=AXIS_FONT+1, fontweight="bold", va="top", ha="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=LEGEND_FONT)
    ax.tick_params(length=3, pad=2)

    # Right: per-config curves (group by topology)
    ax2 = axes[1]
    palette = {"ba":"#1B3A5C","er":"#0D9488","ws":"#F59E0B",
               "hub":"solid","random":"dashed"}
    for key, vals in sorted(config_curves.items()):
        topo = key.split("_")[0]
        col  = palette.get(topo, "#64748B")
        ls   = "dashed" if "random" in key else "solid"
        ax2.plot(vals["epochs"], vals["mean"], color=col, lw=0.9,
                 ls=ls, alpha=0.85, label=key)
    ax2.set_xlabel("Epoch", fontsize=AXIS_FONT)
    ax2.set_ylabel("Per-config RMSE", fontsize=AXIS_FONT)
    ax2.text(-0.12, 1.04, "b", transform=ax2.transAxes,
             fontsize=AXIS_FONT+1, fontweight="bold", va="top", ha="left")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=max(5, LEGEND_FONT-1),
               loc="upper right", ncol=2)
    ax2.tick_params(length=3, pad=2)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "convergence_training")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training convergence → {base}.pdf / .png")


# ── Plot 2: opinion-dynamics convergence time vs Z ───────────────────────────

def compute_convergence_times(model, device, N, mc_runs, val_graphs,
                               threshold, seed):
    """
    Returns dict:
        gt_times  [topo][Z]  → list of t_consensus values (MC)
        zt_times  [topo][Z]  → list of t_consensus values (ZealotTransformer)
    """
    gt_times = {t: {Z: [] for Z in ALL_Z} for t in TOPOLOGIES}
    zt_times = {t: {Z: [] for Z in ALL_Z} for t in TOPOLOGIES}

    total = len(TOPOLOGIES) * len(ALL_Z) * val_graphs
    done  = 0

    for topo in TOPOLOGIES:
        for Z in ALL_Z:
            for g_idx in range(val_graphs):
                done += 1
                s  = seed + g_idx*1000 + hash((topo, Z)) % 1000
                rng = np.random.default_rng(s)
                try:
                    G  = make_graph(topo, N, m=8,
                                    seed=int(rng.integers(0, 99999)))
                    zs = place_hubs(G, Z)
                    gt = simulate_trajectory(G, zs, T=T_STEPS,
                                             mc_runs=mc_runs, seed=s+1)
                    pred = predict_zt(model, G, zs, device)
                    gt_times[topo][Z].append(first_crossing(gt,   threshold))
                    zt_times[topo][Z].append(first_crossing(pred, threshold))
                except Exception:
                    continue
                if done % 20 == 0:
                    print(f"    {done}/{total}", flush=True)

    return gt_times, zt_times


def plot_convergence_speed(gt_times, zt_times, out_dir, threshold, N):
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":AXIS_FONT,"axes.labelsize":AXIS_FONT,
        "xtick.labelsize":TICK_FONT,"ytick.labelsize":TICK_FONT,
        "legend.fontsize":LEGEND_FONT,"axes.linewidth":0.8,
        "pdf.fonttype":42,"ps.fonttype":42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_W_IN * 0.42),
                             sharey=False, constrained_layout=True)

    for ax, topo, lab in zip(axes, TOPOLOGIES, ["a","b","c"]):
        st = TOPO_STYLE[topo]
        gt_means, gt_stds, zt_means, zt_stds = [], [], [], []
        for Z in ALL_Z:
            gtv = gt_times[topo][Z]
            ztv = zt_times[topo][Z]
            gt_means.append(np.mean(gtv) if gtv else np.nan)
            gt_stds .append(np.std(gtv)  if gtv else np.nan)
            zt_means.append(np.mean(ztv) if ztv else np.nan)
            zt_stds .append(np.std(ztv)  if ztv else np.nan)

        ax.errorbar(ALL_Z, gt_means, yerr=gt_stds,
                    color=st["color"], marker=st["marker"], ms=4,
                    lw=LINE_WIDTH, ls=st["ls"],
                    capsize=2, label="Ground Truth", zorder=3)
        ax.errorbar(ALL_Z, zt_means, yerr=zt_stds,
                    color="#CBD5E1", marker="D", ms=4,
                    lw=LINE_WIDTH, ls="--",
                    capsize=2, label="ZealotTransformer", zorder=4,
                    markerfacecolor=st["color"], markeredgecolor=st["color"])

        ax.set_xlabel("Zealot count $Z$", fontsize=AXIS_FONT)
        ax.set_ylabel(f"$t_{{\\mathrm{{consensus}}}}$ ($m \\geq {threshold}$)",
                      fontsize=AXIS_FONT)
        ax.set_title(st["label"], fontsize=AXIS_FONT, pad=3)
        ax.set_xticks(ALL_Z)
        ax.text(-0.14, 1.04, lab, transform=ax.transAxes,
                fontsize=AXIS_FONT+1, fontweight="bold", va="top", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, pad=2)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=LEGEND_FONT, loc="upper right")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "convergence_speed")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence speed  → {base}.pdf / .png")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zt_checkpoint", type=str, required=True)
    p.add_argument("--log_path",      type=str, default=None,
                   help="Path to training_log.json (enables training curve panel)")
    p.add_argument("--out_dir",       type=str, default="convergence_results")
    p.add_argument("--n",             type=int, default=1024)
    p.add_argument("--mc_runs",       type=int, default=128)
    p.add_argument("--val_graphs",    type=int, default=20)
    p.add_argument("--threshold",     type=float, default=THRESHOLD)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nLoading ZealotTransformer...")
    model = load_model(args.zt_checkpoint, device)
    print("  ✓")

    # Panel 1: training convergence (if log available)
    log_path = args.log_path or os.path.join(
        os.path.dirname(args.zt_checkpoint), "training_log.json")
    plot_training_convergence(log_path, args.out_dir)

    # Panel 2: opinion-dynamics convergence time
    print(f"\nComputing convergence times  "
          f"(threshold={args.threshold}, N={args.n}, "
          f"mc_runs={args.mc_runs}, val_graphs={args.val_graphs})...")
    gt_times, zt_times = compute_convergence_times(
        model, device, args.n, args.mc_runs,
        args.val_graphs, args.threshold, args.seed)

    plot_convergence_speed(gt_times, zt_times, args.out_dir,
                           args.threshold, args.n)

    # JSON
    data = {}
    for topo in TOPOLOGIES:
        for Z in ALL_Z:
            k = f"{topo}_Z{Z}"
            data[k] = {
                "gt_times":  gt_times[topo][Z],
                "zt_times":  zt_times[topo][Z],
                "gt_mean":   float(np.mean(gt_times[topo][Z])) if gt_times[topo][Z] else None,
                "gt_std":    float(np.std(gt_times[topo][Z]))  if gt_times[topo][Z] else None,
                "zt_mean":   float(np.mean(zt_times[topo][Z])) if zt_times[topo][Z] else None,
                "zt_std":    float(np.std(zt_times[topo][Z]))  if zt_times[topo][Z] else None,
            }
    jp = os.path.join(args.out_dir, "convergence_speed.json")
    with open(jp, "w") as f: json.dump(data, f, indent=2)
    print(f"  JSON → {jp}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()