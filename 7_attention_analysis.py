"""
attention_analysis.py
=====================
Extracts and analyses the self-attention patterns of ZealotTransformer's
Transformer encoder across BA network densities.

What it measures
----------------
  For each graph instance the script:
    1. Runs a forward pass with attention-weight hooks registered on every
       TransformerEncoderLayer.
    2. Averages attention weights across heads → per-node "received attention"
       (how much other nodes attend to node i).
    3. Correlates received attention with:
         - Node degree
         - Betweenness centrality
    4. Aggregates Pearson r and Spearman ρ across graphs.

This directly replicates the bridge-to-hub transition analysis from the
SpectralLSTM paper, now applied to ZealotTransformer.

Outputs
-------
  attention_vs_degree.pdf / .png    heatmap + scatter: attention vs degree
  attention_correlations.pdf / .png Pearson/Spearman vs m (density) per layer
  attention_analysis.json           raw correlation numbers

Usage
-----
  python attention_analysis.py --zt_checkpoint saved_models/zealot_transformer.pt
"""

import os, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
T_STEPS       = 50
NODE_FEAT_DIM = 5
ALL_Z         = [2, 8, 16, 32]
M_VALUES      = [4, 8, 16]    # BA attachment parameter → controls mean degree
N_GRAPHS      = 10            # graphs per (m, Z) cell

AXIS_FONT = 8; TICK_FONT = 7; LEGEND_FONT = 7; LINE_WIDTH = 1.2
FIG_W_IN  = 180 / 25.4

# ── graph / feature helpers ───────────────────────────────────────────────────

def make_ba_graph(n, m, seed=None):
    return nx.barabasi_albert_graph(n, m, seed=seed)


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


# ── ZealotTransformer with attention hooks ───────────────────────────────────

class ZealotTransformer(nn.Module):
    def __init__(self, node_feat_dim=NODE_FEAT_DIM, d_model=128,
                 nhead=4, num_transformer_layers=3,
                 lstm_hidden=256, lstm_layers=2, T=T_STEPS, dropout=0.0):
        super().__init__()
        self.d_model = d_model; self.T = T; self.nhead = nhead
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
        proj = self.ctx_projector(ctx).view(B,self.lstm_layers,self.lstm_hidden*2)
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


# ── attention extraction ──────────────────────────────────────────────────────

def extract_attention_weights(model, X_t, zm_t, device):
    """
    Registers forward hooks on every TransformerEncoderLayer's self-attention
    module to capture the raw attention weight matrices.

    Returns list of np.arrays, one per layer, shape (N, N) after head-averaging.
    PyTorch's nn.MultiheadAttention stores weights in attn_output_weights when
    need_weights=True (the default in TransformerEncoderLayer).

    Because TransformerEncoderLayer calls F.multi_head_attention_forward internally
    we hook the self_attn submodule's forward output.
    """
    captured = []
    hooks    = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output of MultiheadAttention is (attn_output, attn_weights)
            # attn_weights shape: (B, N, N)  — already averaged over heads
            # when average_attn_weights=True (default in PyTorch ≥ 1.12)
            if isinstance(output, tuple) and len(output) == 2:
                w = output[1]   # (B, N, N)
                if w is not None:
                    captured.append(w.squeeze(0).detach().cpu().numpy())
        return hook

    for i, layer in enumerate(model.transformer.layers):
        # Patch: enable weight return
        layer.self_attn.need_weights = True
        try:
            layer.self_attn.average_attn_weights = True
        except AttributeError:
            pass
        h = layer.self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        pm  = torch.ones(1, X_t.shape[0], dtype=torch.bool, device=device)
        H   = model.node_encoder(X_t.unsqueeze(0))
        model.transformer(H, src_key_padding_mask=~pm)   # triggers hooks

    for h in hooks: h.remove()
    return captured   # list[ (N,N) ] per layer


def received_attention(attn_matrix):
    """
    Column-sum of attention matrix → how much each node is attended to.
    attn_matrix: (N, N) where [i,j] = attention from node i to node j.
    Returns (N,) vector.
    """
    return attn_matrix.sum(axis=0)   # sum over query dimension


# ── correlation analysis ──────────────────────────────────────────────────────

def analyse_graph(model, G, zealot_set, device, n=1024):
    """
    Returns dict with arrays:
      degree, betweenness, received_attn_per_layer
    """
    X    = compute_node_features(G, zealot_set)
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)
    zm_t = torch.tensor(
        [1.0 if nd in zealot_set else 0.0 for nd in range(n)],
        dtype=torch.bool).to(device)

    attn_list = extract_attention_weights(model, X_t, zm_t, device)

    deg  = np.array([d for _, d in G.degree()], dtype=np.float64)
    try:
        btwn_d = nx.betweenness_centrality(G, normalized=True)
        btwn   = np.array([btwn_d[i] for i in range(n)], dtype=np.float64)
    except Exception:
        btwn = np.zeros(n, dtype=np.float64)

    recv = [received_attention(a) for a in attn_list]   # one per layer

    return {"degree": deg, "betweenness": btwn, "received_attn": recv}


def correlate(x, y):
    """Returns (pearson_r, spearman_rho) ignoring NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5: return float("nan"), float("nan")
    pr, _ = pearsonr(x[mask],  y[mask])
    sr, _ = spearmanr(x[mask], y[mask])
    return float(pr), float(sr)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_attention_vs_degree(records, m_val, out_dir):
    """
    Scatter of received attention vs degree for one m value,
    coloured by node type (zealot / hub / peripheral).
    """
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":AXIS_FONT,"axes.labelsize":AXIS_FONT,
        "xtick.labelsize":TICK_FONT,"ytick.labelsize":TICK_FONT,
        "legend.fontsize":LEGEND_FONT,"axes.linewidth":0.8,
        "pdf.fonttype":42,"ps.fonttype":42,
    })

    n_layers = len(records[0]["received_attn"]) if records else 0
    n_panels = min(n_layers, 3)
    if n_panels == 0:
        print("  ⚠ No attention layers captured — skipping scatter plot")
        return

    fig, axes = plt.subplots(1, n_panels,
                              figsize=(FIG_W_IN, FIG_W_IN * 0.38),
                              constrained_layout=True)
    if n_panels == 1: axes = [axes]

    for li in range(n_panels):
        ax = axes[li]
        all_deg, all_attn = [], []
        for r in records:
            all_deg.extend(r["degree"].tolist())
            all_attn.extend(r["received_attn"][li].tolist())
        deg_arr  = np.array(all_deg)
        attn_arr = np.array(all_attn)

        # Scatter with degree-based colour gradient
        sc = ax.scatter(deg_arr, attn_arr, c=deg_arr, cmap="YlOrRd",
                        s=4, alpha=0.5, linewidths=0, rasterized=True)
        pr, sr = correlate(deg_arr, attn_arr)

        ax.set_xlabel("Node degree $k$", fontsize=AXIS_FONT)
        ax.set_ylabel("Received attention", fontsize=AXIS_FONT) if li == 0 else None
        ax.set_title(f"Layer {li+1}  "
                     r"$\rho$" + f"={sr:.3f}",
                     fontsize=AXIS_FONT, pad=3)
        ax.text(-0.14, 1.04, chr(ord("a")+li), transform=ax.transAxes,
                fontsize=AXIS_FONT+1, fontweight="bold", va="top", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, pad=2)

    plt.colorbar(sc, ax=axes[-1], label="Degree", shrink=0.8,
                 fraction=0.03, pad=0.02)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"attention_vs_degree_m{m_val}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter (m={m_val}) → {base}.pdf / .png")


def plot_correlation_vs_density(corr_table, out_dir):
    """
    Line plot: Pearson r and Spearman ρ vs m (network density proxy)
    for degree and betweenness centrality, averaged over last layer.
    Reproduces the bridge-to-hub transition figure.
    """
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":AXIS_FONT,"axes.labelsize":AXIS_FONT,
        "xtick.labelsize":TICK_FONT,"ytick.labelsize":TICK_FONT,
        "legend.fontsize":LEGEND_FONT,"axes.linewidth":0.8,
        "pdf.fonttype":42,"ps.fonttype":42,
    })

    m_vals    = sorted(corr_table.keys())
    pr_deg    = [corr_table[m]["pearson_deg"][-1]   for m in m_vals]
    sr_deg    = [corr_table[m]["spearman_deg"][-1]  for m in m_vals]
    pr_btwn   = [corr_table[m]["pearson_btwn"][-1]  for m in m_vals]
    sr_btwn   = [corr_table[m]["spearman_btwn"][-1] for m in m_vals]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN * 0.6, FIG_W_IN * 0.38),
                              constrained_layout=True)

    for ax, (pr_vals, sr_vals, ylabel, lab) in zip(axes, [
        (pr_deg,  sr_deg,  "Correlation with degree",      "a"),
        (pr_btwn, sr_btwn, "Correlation with betweenness", "b"),
    ]):
        ax.plot(m_vals, pr_vals, color="#1B3A5C", marker="o", ms=4,
                lw=LINE_WIDTH, label="Pearson $r$")
        ax.plot(m_vals, sr_vals, color="#0D9488", marker="s", ms=4,
                lw=LINE_WIDTH, ls="--", label=r"Spearman $\rho$")
        ax.axhline(0, color="#CBD5E1", lw=0.6, ls=":")
        ax.set_xlabel("BA attachment parameter $m$  ($\\langle k\\rangle = 2m$)",
                      fontsize=AXIS_FONT)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONT)
        ax.set_xticks(m_vals)
        ax.set_xticklabels([f"$m={m}$\n$\\langle k\\rangle={2*m}$"
                            for m in m_vals], fontsize=TICK_FONT)
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes,
                fontsize=AXIS_FONT+1, fontweight="bold", va="top", ha="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, pad=2)
        ax.legend(frameon=False, fontsize=LEGEND_FONT)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "attention_correlations")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Correlation plot → {base}.pdf / .png")


def print_correlation_table(corr_table, out_dir):
    """Print Table matching the paper's Table 4 format."""
    SEP = "─"
    cols = ["Pearson r (deg)", "Pearson r (btwn)",
            "Spearman ρ (deg)", "Spearman ρ (btwn)"]
    cw = 18; hw = 6
    total_w = hw + (cw+3)*len(cols) + 3

    lines = [
        "Global-GAT → ZealotTransformer Attention vs Centrality",
        "(Z=8, N=1024, averaged over graphs, last Transformer layer)",
        SEP * total_w,
        f"{'m':^{hw}}" + "".join(f"   {c:^{cw}}" for c in cols),
        SEP * total_w,
    ]
    for m in sorted(corr_table.keys()):
        d = corr_table[m]
        row = (f"{m:^{hw}}"
               f"   {d['pearson_deg'][-1]:^{cw}.3f}"
               f"   {d['pearson_btwn'][-1]:^{cw}.3f}"
               f"   {d['spearman_deg'][-1]:^{cw}.3f}"
               f"   {d['spearman_btwn'][-1]:^{cw}.3f}")
        lines.append(row)
    lines.append(SEP * total_w)
    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "attention_correlations_table.txt")
    with open(path, "w") as f: f.write(text + "\n")
    print(f"\n  Table → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zt_checkpoint", type=str, required=True)
    p.add_argument("--out_dir",       type=str, default="attention_results")
    p.add_argument("--n",             type=int, default=1024)
    p.add_argument("--n_graphs",      type=int, default=N_GRAPHS,
                   help="Graphs per (m, Z) cell")
    p.add_argument("--z_fixed",       type=int, default=8,
                   help="Zealot count to use for attention analysis")
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
    n_layers = len(model.transformer.layers)
    print(f"  ✓  ({n_layers} Transformer layers, nhead={model.nhead})")

    corr_table = {}   # m → {pearson_deg:[per layer], ...}
    all_json   = {}

    for m_val in M_VALUES:
        mean_deg = 2 * m_val
        print(f"\n  m={m_val}  ⟨k⟩={mean_deg}  Z={args.z_fixed}", flush=True)

        records      = []
        pr_deg_buf   = [[] for _ in range(n_layers)]
        sr_deg_buf   = [[] for _ in range(n_layers)]
        pr_btwn_buf  = [[] for _ in range(n_layers)]
        sr_btwn_buf  = [[] for _ in range(n_layers)]

        for g_idx in range(args.n_graphs):
            seed = args.seed + g_idx * 100 + m_val
            try:
                G  = make_ba_graph(args.n, m_val, seed=seed)
                zs = place_hubs(G, args.z_fixed)
                r  = analyse_graph(model, G, zs, device, args.n)
                records.append(r)

                for li in range(n_layers):
                    recv = r["received_attn"][li]
                    pr, sr = correlate(r["degree"], recv)
                    pr_deg_buf[li].append(pr); sr_deg_buf[li].append(sr)
                    pr, sr = correlate(r["betweenness"], recv)
                    pr_btwn_buf[li].append(pr); sr_btwn_buf[li].append(sr)
            except Exception as e:
                print(f"    graph {g_idx} failed: {e}")
                continue

        # Layer-wise means
        corr_table[m_val] = {
            "mean_degree":   mean_deg,
            "pearson_deg":   [float(np.nanmean(b)) for b in pr_deg_buf],
            "spearman_deg":  [float(np.nanmean(b)) for b in sr_deg_buf],
            "pearson_btwn":  [float(np.nanmean(b)) for b in pr_btwn_buf],
            "spearman_btwn": [float(np.nanmean(b)) for b in sr_btwn_buf],
        }
        all_json[f"m{m_val}"] = corr_table[m_val]

        # Print per-layer table for this m
        print(f"  {'Layer':>6}  {'Pearson_deg':>12}  "
              f"{'Spearman_deg':>13}  {'Pearson_btwn':>13}  {'Spearman_btwn':>14}")
        for li in range(n_layers):
            print(f"  {li+1:>6}  "
                  f"{corr_table[m_val]['pearson_deg'][li]:>12.3f}  "
                  f"{corr_table[m_val]['spearman_deg'][li]:>13.3f}  "
                  f"{corr_table[m_val]['pearson_btwn'][li]:>13.3f}  "
                  f"{corr_table[m_val]['spearman_btwn'][li]:>14.3f}")

        # Scatter plot for this m
        plot_attention_vs_degree(records, m_val, args.out_dir)

    # Correlation vs density figure + table
    plot_correlation_vs_density(corr_table, args.out_dir)
    print_correlation_table(corr_table, args.out_dir)

    jp = os.path.join(args.out_dir, "attention_analysis.json")
    with open(jp, "w") as f: json.dump(all_json, f, indent=2)
    print(f"\n  JSON → {jp}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()