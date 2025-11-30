"""
Python frontend for the dyn-cc Rust extension.

The compiled extension module (built with maturin) is named ``dyn_cc_py``.
This package re-exports the native functions and is the place to add
any pure-Python helpers or ergonomics.
"""

from importlib import metadata
from typing import List, Sequence, Tuple
import numpy as np

# Import everything from the native extension.
from .dyn_cc_py import *  # noqa: F401,F403


def __getattr__(name: str):
    """Expose the package version via dyn_cc_py.__version__."""
    if name == "__version__":
        try:
            return metadata.version("dyn-cc-py")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_sbm(
    block_sizes: Sequence[int],
    p_in: float,
    p_out: float,
    *,
    seed: int | None = None,
    layer: str = "sbm",
    timestamp: int = 0,
    weight: float = 1.0,
) -> Tuple["PersistentGraph", List[int]]:
    """
    Generate a simple undirected stochastic block model into a Raphtory persistent graph.

    Args:
        block_sizes: sizes of each community.
        p_in: edge probability within a block.
        p_out: edge probability between blocks.
        seed: RNG seed.
        layer: optional layer name for edges.
        timestamp: timestamp used for all edges.
        weight: edge weight stored in property ``w``.

    Returns:
        (graph, memberships) where ``memberships[i]`` is the block id of node ``i``.
    """
    rng = np.random.default_rng(seed)
    memberships: List[int] = []
    for block_id, size in enumerate(block_sizes):
        memberships.extend([block_id] * size)

    g = new_persistent_graph()
    n = len(memberships)

    for u in range(n):
        for v in range(u + 1, n):
            same_block = memberships[u] == memberships[v]
            p = p_in if same_block else p_out
            if rng.random() < p:
                g.add_edge(
                    timestamp=timestamp,
                    src=u,
                    dst=v,
                    properties={"w": weight},
                    layer=layer,
                )

    return g, memberships


__all__ = [name for name in globals() if not name.startswith("_")]
