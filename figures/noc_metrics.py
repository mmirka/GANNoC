"""Vendored-minimal NoC graph/validity helpers for the paper-figure scripts.

Self-contained by design: this module mirrors the topology-validity logic in
``gannoc.data`` but does **not** import it (or TensorFlow, or anything from the
``gannoc`` package). The figure scripts only need to turn a stack of binary
adjacency matrices into connection counts and validity flags, so a bare
``pip install numpy matplotlib networkx`` environment is enough to run
``figures/``.

A NoC topology here is a ``(N_ROUTERS, N_ROUTERS)`` binary, symmetric,
zero-diagonal adjacency matrix (RAPIDO 2021 / PhD thesis Chapter 5). "Number
of connections" always means the number of undirected physical links, i.e. the
number of 1s in the strict upper triangle.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

import numpy as np

# Fixed 9-router problem size of the GANNoC paper, and the physical limits a
# generated topology must respect to be buildable (RAPIDO 2021 / Chapter 5):
# every router has at most 4 ports, a connected 9-node graph needs at least 8
# links (a spanning tree), and 9 routers at degree 4 cap out at 18 links.
N_ROUTERS = 9
MAX_DEGREE = 4
MIN_CONNECTIONS = 8
MAX_CONNECTIONS = 18


def _as_adjacency(matrix: "np.ndarray | Iterable") -> np.ndarray:
    """Return ``matrix`` as a float ndarray with its diagonal forced to zero.

    Self-loops are meaningless for a router graph; zeroing the diagonal here
    keeps every downstream degree/connection count honest regardless of what
    the caller passed in.
    """
    adj = np.array(matrix, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"expected a square adjacency matrix, got shape {adj.shape}")
    np.fill_diagonal(adj, 0.0)
    return adj


def degrees(matrix: "np.ndarray | Iterable") -> np.ndarray:
    """Per-router degree (number of incident links) as an int ndarray."""
    return _as_adjacency(matrix).sum(axis=1).astype(int)


def n_connections_of(matrix: "np.ndarray | Iterable") -> int:
    """Number of undirected physical links: the count of 1s above the diagonal."""
    return int(np.triu(_as_adjacency(matrix), k=1).sum())


def is_connected(matrix: "np.ndarray | Iterable", n_routers: int = N_ROUTERS) -> bool:
    """True iff every router is reachable from router 0 (plain BFS).

    An isolated router, or two disjoint clusters, make a topology unbuildable
    as a single NoC, so this is a hard validity gate.
    """
    adj = _as_adjacency(matrix)
    if adj.shape[0] != n_routers:
        return False

    neighbours = [np.flatnonzero(adj[i] > 0).tolist() for i in range(n_routers)]
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nxt in neighbours[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == n_routers


def is_valid_topology(
    matrix: "np.ndarray | Iterable",
    n_routers: int = N_ROUTERS,
    max_degree: int = MAX_DEGREE,
) -> bool:
    """True iff ``matrix`` is a buildable NoC topology.

    Checks, in order: right shape, binary entries, symmetric, no self-loops,
    no router over ``max_degree`` ports, and the whole graph connected. A
    connected 9-node graph automatically has at least ``MIN_CONNECTIONS``
    links, and the degree cap bounds it by ``MAX_CONNECTIONS``.
    """
    adj = np.array(matrix, dtype=float)
    if adj.ndim != 2 or adj.shape != (n_routers, n_routers):
        return False
    if not np.array_equal(adj, adj.astype(bool)):
        return False
    if not np.array_equal(adj, adj.T):
        return False
    if np.any(np.diag(adj) != 0):
        return False
    if degrees(adj).max(initial=0) > max_degree:
        return False
    return is_connected(adj, n_routers=n_routers)


def connection_count_histogram(
    matrices: Iterable,
    lo: int = MIN_CONNECTIONS,
    hi: int = MAX_CONNECTIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of link counts over the integer bins ``lo..hi`` (both inclusive).

    Returns ``(bins, counts)`` where ``bins == np.arange(lo, hi + 1)`` and
    ``counts[i]`` is how many input matrices have exactly ``bins[i]`` links.
    Matrices whose link count falls outside ``[lo, hi]`` are not counted; the
    bin range spans every count a valid 9-router topology can have.
    """
    bins = np.arange(lo, hi + 1)
    counts = np.zeros(bins.shape, dtype=int)
    for matrix in matrices:
        n = n_connections_of(matrix)
        if lo <= n <= hi:
            counts[n - lo] += 1
    return bins, counts
