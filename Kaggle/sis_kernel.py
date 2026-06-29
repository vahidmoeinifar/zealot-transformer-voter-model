"""
SIS epidemic kernel with fixed infected seeds (the 'zealot' analogue).
======================================================================
"""

import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count


# Core SIS run
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


def _mc_worker(args):
    G, seeds, beta, gamma, T, run_seed = args
    rng = np.random.default_rng(run_seed)
    return simulate_sis_once(G, seeds, beta, gamma, T, rng)


def simulate_sis_mc(G, seeds, beta, gamma, T, n_runs, seed=0, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count())
    if n_runs < 8 or n_workers == 1:
        rng = np.random.default_rng(seed)
        acc = np.zeros(T, dtype=np.float64)
        for _ in range(n_runs):
            acc += simulate_sis_once(G, seeds, beta, gamma, T, rng)
        return acc / n_runs
    args = [(G, seeds, beta, gamma, T, seed * 100000 + r) for r in range(n_runs)]
    with Pool(processes=n_workers) as pool:
        trajs = pool.map(_mc_worker, args)
    return np.mean(np.stack(trajs, axis=0), axis=0)


def pick_seeds(G, Z, strategy="hub", rng=None):
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


def _steady_no_seed(G, init_infected, beta, gamma, T, n_runs):
    N = G.number_of_nodes()
    neighbors = [np.array(list(G.neighbors(n)), dtype=np.int64) for n in range(N)]
    rng = np.random.default_rng(7)
    tail = []
    for _ in range(n_runs):
        infected = np.zeros(N, dtype=bool)
        infected[init_infected] = True
        for t in range(T):
            new = infected.copy()
            for u in np.where(~infected)[0]:
                nb = neighbors[u]
                if nb.size and infected[nb].any():
                    if rng.random() < 1 - (1 - beta) ** int(infected[nb].sum()):
                        new[u] = True
            for v in np.where(infected)[0]:
                if rng.random() < gamma:
                    new[v] = False
            infected = new
        tail.append(infected.mean())
    return float(np.mean(tail))


def validate_threshold(N=1000, m=8, T=80, n_runs=40):
    rng = np.random.default_rng(1)
    G = nx.barabasi_albert_graph(N, m, seed=1)
    A = nx.to_numpy_array(G)
    lam = np.max(np.linalg.eigvalsh(A))
    gamma = 0.5
    beta_c = gamma / lam
    print(f"lambda_max = {lam:.3f},  gamma = {gamma},  predicted beta_c = {beta_c:.4f}")
    print(f"{'beta/beta_c':>11} | {'beta':>7} | {'steady i':>9}")
    for ratio in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        beta = ratio * beta_c
        init = rng.choice(N, size=max(1, N // 100), replace=False)
        steady = _steady_no_seed(G, init, beta, gamma, T, n_runs)
        flag = "endemic" if steady > 0.01 else "die-out"
        print(f"{ratio:>11.2f} | {beta:>7.4f} | {steady:>9.4f}  {flag}")
    print("\nExpect: die-out below ratio 1.0, endemic above. If so, kernel is correct.")


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
                     gamma=0.5, beta_ratio=1.5,
                     T=50, n_runs=128, n_graphs=5, out="sis_dataset.npz"):
    rng = np.random.default_rng(0)
    records = []
    for topo in topologies:
        for N in N_list:
            for g in range(n_graphs):
                G = _make_graph(topo, N, rng)
                A = nx.to_numpy_array(G)
                lam = np.max(np.linalg.eigvalsh(A))
                beta = beta_ratio * gamma / lam
                edges = np.array(G.edges(), dtype=np.int64)
                for placement in placements:
                    for Z in Z_list:
                        seeds = pick_seeds(G, Z, placement, rng)
                        traj = simulate_sis_mc(G, seeds, beta, gamma, T, n_runs, seed=g)
                        records.append(dict(topo=topo, N=N, Z=Z, strategy=placement,
                                            beta=beta, gamma=gamma, graph_id=g,
                                            traj=traj, seeds=seeds, edges=edges))
                print(f"{topo} N={N} graph {g}: done")
    np.savez_compressed(out, records=records)
    print(f"saved {len(records)} trajectories -> {out}")


if __name__ == "__main__":
    validate_threshold()
    # generate_dataset(out="/kaggle/working/sis_dataset.npz")
