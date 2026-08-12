import time, warnings
import numpy as np, networkx as nx
warnings.filterwarnings("ignore")
from zt_features import compute_node_features

for n in (1024, 8192):
    G = nx.barabasi_albert_graph(n, 8, seed=1)
    s = np.argsort([d for _, d in G.degree()])[-8:]
    ts = []
    for _ in range(3):
        t = time.time()
        compute_node_features(G, s)
        ts.append(time.time() - t)
    print(f"N={n}: preprocessing = {np.median(ts):.3f} s")