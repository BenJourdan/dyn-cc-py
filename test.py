import dyn_cc_py
from dyn_cc_py import DynamicClustering, new_persistent_graph
from typing import Tuple, List
from scipy.sparse import csr_array, csr_matrix
from sknetwork.clustering import Leiden, Louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import warnings

try:
    import kaleido  # noqa: F401
    HAVE_KALEIDO = True
except ImportError:
    HAVE_KALEIDO = False

import time

def apply_sbm_edges(
    graph,
    k_clusters,
    n_per_cluster,
    p_internal,
    q_external,
    n_multiplier,
    rng,
    nodes
    ):
    # Expected edge count of an SBM(k, n_per_cluster, p_internal, q_external).
    expected_internal = k_clusters * n_per_cluster * (n_per_cluster - 1) * 0.5 * p_internal
    expected_external = k_clusters * (k_clusters - 1) * 0.5 * (n_per_cluster * n_per_cluster) * q_external
    expected_edges = expected_internal + expected_external
    num_updates = int(np.ceil(n_multiplier * expected_edges))
    internal_prob = expected_internal / expected_edges if expected_edges > 0 else 0.5
    t = 0
    weights = {}
    for _ in range(num_updates):
        if rng.random() < internal_prob:
            c = rng.integers(0, k_clusters)
            choices = list(range(c * n_per_cluster, (c + 1) * n_per_cluster))
            u, v = rng.choice(choices, size=2, replace=False)
        else:
            c1, c2 = rng.choice(k_clusters, size=2, replace=False)
            u = rng.integers(c1 * n_per_cluster, (c1 + 1) * n_per_cluster)
            v = rng.integers(c2 * n_per_cluster, (c2 + 1) * n_per_cluster)

        if u == v:
            continue

        key = tuple(sorted((u, v)))
        w = weights.get(key, 0.0) + 1.0
        weights[key] = w

        graph.add_edge(
            timestamp=int(t),
            src=nodes[u],
            dst=nodes[v],
            properties={"w": w},
            layer="sbm",
        )
        graph.add_edge(
            timestamp=int(t),
            src=nodes[v],
            dst=nodes[u],
            properties={"w": w},
            layer="sbm",
        )
        t += 1


    end = t -1
    start = 0
    return start,end

def leiden(graph: csr_array, clusters: int) -> Tuple[List[int], int]:
    """Leiden clustering (sknetwork) on a precomputed adjacency (CSR)."""

    nrows, _ = graph.shape
    if nrows == 0:
        return [], 0

    # Zero the diagonal
    adj = csr_matrix(graph, dtype=float)

    adj.setdiag(0.0)
    adj.eliminate_zeros()

    model = Leiden(random_state=0, resolution=1.0)
    labels = model.fit_predict(adj)
    used = int(labels.max() + 1) if labels.size else 0
    return labels.tolist(), used

def spectral(graph: csr_array, clusters: int) -> Tuple[List[int], int]:
    """Spectral clustering using scikit-learn on a precomputed adjacency (CSR)."""

    nrows, _ = graph.shape
    if nrows == 0:
        return [], 0
    adj = csr_matrix(graph, dtype=float)
    model = SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(adj)
    return labels.tolist(), clusters


def louvain(graph: csr_array, clusters: int) -> Tuple[List[int], int]:
    """Louvain clustering using scikit-network on a precomputed adjacency (CSR)."""

    nrows, _ = graph.shape
    if nrows == 0:
        return [], 0

    adj = csr_matrix(graph, dtype=float)
    adj.setdiag(0.0)
    adj.eliminate_zeros()

    model = Louvain(random_state=0, resolution=1.0)
    labels = model.fit_predict(adj)
    used = int(labels.max() + 1) if labels.size else 0
    return labels.tolist(), used


def raphtory_louvain_baseline(graph, start: int, end: int, step: int, subset: List[str], subset_labels: List[int], weight_prop: str = "w"):
    """Run Raphtory's Louvain on the same snapshot times as the dynamic algorithms."""
    times = list(range(start, end, step))
    records = []

    t0 = time.time()
    for t in times:
        view = graph.at(t)
        clustering = dyn_cc_py.algorithms.louvain(view, weight_prop=weight_prop)
        labels = []
        for node in subset:
            val = clustering.get(node)
            labels.append(val if val is not None else -1)
        clusters = int(max((v for v in labels if v >= 0), default=-1) + 1) if labels else 0
        ari = adjusted_rand_score(labels, subset_labels)
        records.append({"time": t, "ari": ari, "clusters": clusters})

    duration = time.time()-t0
    print(f"baseline took {duration:.3f} seconds")
    return records, duration


def run_with_alg(graph,start,end,step,alg, num_clusters, subset, subset_labels, coreset_size=1024):
    dyn_cc = DynamicClustering(
    alg,
    sigma=1000,
    coreset_size=coreset_size,
    sampling_seeds=num_clusters*4,
    num_clusters=num_clusters,
    prop_name="w"
    )

    t0 = time.time()

    times, snapshot_predicted_labels,cluster_sizes= dyn_cc.cluster_subset_on_snapshots(
    graph,        # PersistentGraph from dyn_cc_py
    start=start,
    end=end,
    step=step,
    subset=subset,  # cluster all nodes
    )

    records = []
    for (t, pred_labels, k) in zip(times,snapshot_predicted_labels, cluster_sizes):
        ari = adjusted_rand_score(pred_labels,subset_labels)
        records.append({"time": t, "ari": ari, "clusters": k})

    duration = time.time() -t0
    print(f"{alg.__name__}Took {duration:.3f} seconds")
    return records, duration

def main():
    # Mimic the generate_sbm_commands helper from dyn-cc tests (simplified: only inserts).
    seed = 4242
    
    n_per_cluster = 1024
    k_clusters = 10
    p_internal = 0.33
    q_external = 1.0/(n_per_cluster*k_clusters)
    n_multiplier = 3

    total_nodes = n_per_cluster * k_clusters
    coreset_size = int(total_nodes/ 20.0)

    rng = np.random.default_rng(seed)
    memberships = []
    nodes = []
    labels = []
    for c in range(k_clusters):
        for i in range(n_per_cluster):
            memberships.append(c)
            nodes.append(f"C{c}_{i}")
            labels.append(c)


    subset = nodes
    subset_labels = labels

    graph = new_persistent_graph()
    start,end = apply_sbm_edges(
        graph,
        k_clusters,
        n_per_cluster,
        p_internal,
        q_external,
        n_multiplier,
        rng,
        nodes
    )

    start = int(end/16)
    step = int(end/20)

    # Run all algs and collect metrics
    baseline_louvain, baseline_time = raphtory_louvain_baseline(graph, start, end, step, subset, subset_labels, weight_prop="w")
    results = {
        "Leiden": run_with_alg(graph, start,end,step, leiden, k_clusters, subset, subset_labels, coreset_size),
        "Spectral": run_with_alg(graph, start,end,step, spectral, k_clusters, subset, subset_labels, coreset_size),
        "Louvain": run_with_alg(graph, start,end,step, louvain, k_clusters, subset, subset_labels, coreset_size),
        "Raphtory Louvain": (baseline_louvain, baseline_time),
    }

    # Plot ARI and clusters used over time on separate subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.1,
        subplot_titles=("Adjusted Rand Index", "Clusters used"),
    )
    palette = {
        "Leiden": "blue",
        "Spectral": "green",
        "Louvain": "red",
        "Raphtory Louvain": "orange",
    }

    ari_title_added = False
    cluster_title_added = False

    for name, (recs, duration) in results.items():
        times = [r["time"] for r in recs]
        aris = [r["ari"] for r in recs]
        clusters = [r["clusters"] for r in recs]
        label = f"{name} ({duration:.1f}s)"

        fig.add_trace(
            go.Scatter(
                x=times,
                y=aris,
                mode="lines+markers",
                name=f"{label} ARI",
                line=dict(color=palette.get(name, None)),
                legendgroup="ARI",
                legendgrouptitle_text=None if ari_title_added else "ARI",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=clusters,
                mode="lines+markers",
                name=f"{label} clusters",
                line=dict(dash="dot", color=palette.get(name, None)),
                legendgroup="Clusters",
                legendgrouptitle_text=None if cluster_title_added else "Clusters",
            ),
            row=2, col=1,
        )
        ari_title_added = True
        cluster_title_added = True

    fig.update_layout(
        title="Clustering metrics over time",
        xaxis_title="Time",
        legend=dict(tracegroupgap=10),
    )
    fig.update_yaxes(title_text="ARI", range=[0,1], row=1, col=1)
    fig.update_yaxes(title_text="Clusters used", row=2, col=1)
    fig.write_image("cluster_metrics.png")
    fig.write_image("cluster_metrics.svg")
    fig.show()





    

if __name__ == "__main__":
    main()
