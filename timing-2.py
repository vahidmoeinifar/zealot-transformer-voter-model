import time, warnings
import numpy as np, networkx as nx
warnings.filterwarnings("ignore")

def features_as_used(G, zealot_set):
    """The pipeline actually used for the reported results (degree proxy above thresholds)."""
    N = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    dn = deg / (deg.max() + 1e-8)
    z = np.zeros(N, dtype=np.float32)
    for nd in zealot_set: z[int(nd)] = 1.0
    fied = dn.copy() if N > 500 else dn.copy()      # proxy at all sizes >500
    pr = dn.copy() if N > 1000 else dn.copy()
    clust = np.zeros(N, dtype=np.float32)
    if N <= 2000:
        cd = nx.clustering(G)
        clust = np.array([cd[i] for i in range(N)], dtype=np.float32)
    return np.stack([z, dn, fied, pr, clust], axis=1)

for n in (1024, 8192):
    G = nx.barabasi_albert_graph(n, 8, seed=1)
    s = np.argsort([d for _, d in G.degree()])[-8:]
    ts = []
    for _ in range(3):
        t = time.time(); features_as_used(G, s); ts.append(time.time() - t)
    print(f"N={n}: preprocessing (as used) = {np.median(ts):.3f} s")