import dyn_cc_py
from dyn_cc_py import DynamicClustering, test_dummy_callback
from typing import Tuple, List
from scipy.sparse import csr_array
import numpy as np
from raphtory import PersistentGraph


def cluster_alg(graph: csr_array, clusters: int) -> Tuple[List[int], int]:
    """Dummy clustering callback: labels 0..nrows-1, clusters_used=clusters."""
    nrows, _ = graph.shape
    labels = list(range(nrows))
    return labels, clusters


def main():

    graph = PersistentGraph() 

    graph.add_edge(
        timestamp=1,
        src="A",
        dst="B",
        properties={"w": 10},
        layer="Friends",
    )

    dyn_cc = DynamicClustering(
        cluster_alg,
        sigma=1000,
        coreset_size=1024,
        sampling_seeds=200,
        num_clusters=10,
        prop_name="w"
    )

    dyn_cc.cluster_subset_on_snapshots(
        graph,        # PersistentGraph
        start=0,
        end=10,
        step=1,
        subset=["A", "B"],  # GIDs (str/u64)
    )


    # This runner is intended for Rust-side invocation; we keep a simple smoke test of instantiation.



if __name__ == "__main__":
    main()
