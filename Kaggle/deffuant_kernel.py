"""
Deffuant bounded-confidence kernel with stubborn agents (the 'zealot' analogue).
================================================================================

Pure numpy + networkx. Runs on Kaggle CPU.
"""

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Cluster counting (the observable)
# ---------------------------------------------------------------------------
def count_clusters(opinions, tol=0.02):

    x = np.sort(np.asarray(opinions, dtype=np.float64))
    if x.size == 0:
        return 0
    n = 1
    for a, b in zip(x[:-1], x[1:]):
        if (b - a) > tol:
            n += 1
    return n


# ---------------------------------------------------------------------------
# One Deffuant run -> n_clusters trajectory
# ---------------------------------------------------------------------------
def simulate_deffuant_once(G, stubborn, eps, mu, T, rng, steps_per_t=None):

    N = G.number_of_nodes()
    edges = np.array(G.edges(), dtype=np.int64)
    if edges.size == 0:
        return np.ones(T, dtype=np.float64)
    if steps_per_t is None:
        steps_per_t = len(edges)

    x = rng.random(N).astype(np.float64)        # initial opinions in [0,1)
    stub_mask = np.zeros(N, dtype=bool)
    if len(stubborn) > 0:
        stub_mask[np.asarray(stubborn, dtype=np.int64)] = True
    x[stub_mask] = 1.0                          # stubborn agents fixed at 1.0

    nclust = np.empty(T, dtype=np.float64)
    for t in range(T):
        nclust[t] = count_clusters(x)
        # one "sweep" = steps_per_t random pairwise interactions
        idx = rng.integers(0, len(edges), size=steps_per_t)
        for e in idx:
            i, j = edges[e]
            if abs(x[i] - x[j]) < eps:
                xi, xj = x[i], x[j]
                if not stub_mask[i]:
                    x[i] = xi + mu * (xj - xi)
                if not stub_mask[j]:
                    x[j] = xj + mu * (xi - xj)
    return nclust


def simulate_deffuant_mc(G, stubborn, eps, mu, T, n_runs, seed=0):
    """MC-average the n_clusters(t) trajectory over n_runs runs."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(T, dtype=np.float64)
    for _ in range(n_runs):
        acc += simulate_deffuant_once(G, stubborn, eps, mu, T, rng)
    return acc / n_runs


# ---------------------------------------------------------------------------
# Stubborn-agent placement (mirrors hub / random / bridge)
# ---------------------------------------------------------------------------
def pick_stubborn(G, Z, strategy="hub", rng=None):
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


# ---------------------------------------------------------------------------
# 1) CORRECTNESS CHECK -- run FIRST
# ---------------------------------------------------------------------------
def validate_confidence(N=500, m=8, T=40, n_runs=10, mu=0.5):

    G = nx.barabasi_albert_graph(N, m, seed=1)
    print(f"{'eps':>6} | {'final n_clusters':>16} | rough 1/(2eps)")
    for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        traj = simulate_deffuant_mc(G, stubborn=[], eps=eps, mu=mu,
                                    T=T, n_runs=n_runs, seed=2)
        approx = 1.0 / (2 * eps)
        print(f"{eps:>6.2f} | {traj[-1]:>16.2f} | {approx:>10.1f}")
    print("\nExpect: clusters DECREASE as eps increases "
          "(large eps -> ~1 consensus). If so, the kernel is correct.")


# ---------------------------------------------------------------------------
# 2) DATASET GENERATION
# ---------------------------------------------------------------------------
def _make_graph(topo, N, rng):
    s = int(rng.integers(1e9))
    if topo == "ba":
        return nx.barabasi_albert_graph(N, 8, seed=s)
    if topo == "er":
        return nx.gnp_random_graph(N, 16 / (N - 1), seed=s)
    if topo == "ws":
        return nx.watts_strogatz_graph(N, 16, 0.1, seed=s)
    if topo == "rgg":
        return nx.random_geometric_graph(N, np.sqrt(16 / (np.pi * N)), seed=s)
    raise ValueError(topo)

# in the real paper we just used BA/ER/WS graphs at N∈{256,512,1024}, Z=8, both hub and random placements
def generate_dataset(topologies=("ba", "er", "ws"),
                     N_list=(256, 512, 1024, 2048),
                     Z_list=(2, 8, 16, 32),
                     placements=("hub", "random"),
                     eps=0.15, mu=0.5,
                     T=50, n_runs=128, n_graphs=10, out="deffuant_dataset.npz"):

    rng = np.random.default_rng(0)
    records = []
    for topo in topologies:
        for N in N_list:
            for g in range(n_graphs):
                G = _make_graph(topo, N, rng)
                edges = np.array(G.edges(), dtype=np.int64)
                for placement in placements:
                    for Z in Z_list:
                        stub = pick_stubborn(G, Z, placement, rng)
                        traj = simulate_deffuant_mc(G, stub, eps, mu, T, n_runs, seed=g)
                        records.append(dict(topo=topo, N=N, Z=Z, strategy=placement,
                                            eps=eps, mu=mu, graph_id=g,
                                            traj=traj, seeds=stub, edges=edges))
                print(f"{topo} N={N} graph {g}: done")
    np.savez_compressed(out, records=records)
    print(f"saved {len(records)} trajectories -> {out}")


if __name__ == "__main__":
    # STEP 1: always run this first and read the output.
    validate_confidence()
    # STEP 2: uncomment once the check looks right.
    # generate_dataset()
