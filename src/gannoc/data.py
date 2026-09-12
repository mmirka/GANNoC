"""Data representation and synthetic dataset generation for GANNoC.

A Network-on-Chip topology over a fixed set of routers is represented as a
binary, symmetric adjacency matrix with a zero diagonal and treated as a
single-channel image. A topology is *valid* when every router carries at least
one link, no router exceeds its external-port budget (degree <= 4), and the
graph is connected. The synthetic training set is built purely by random
construction (RAPIDO 2021 / PhD Thesis Chapter 5, "Algorithm 1"): no NoC
simulator is involved and the dataset carries graph structure only.

This module provides the random topology builder, a connection-count-stratified
dataset generator with on-disk (de)serialization, an in-memory ``NoCDataset``
wrapper that exposes the model-facing image / normalized-target views, and a
handful of numpy-only graph-validity helpers (connectivity is checked with a
plain BFS rather than a graph library).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Router count of the topologies this repo generates. The paper fixes the
# problem to a 9-router NoC, so the adjacency "image" is always 9x9.
N_ROUTERS = 9

# Maximum node degree. A router exposes at most 4 external ports, so it can be
# wired to at most 4 other routers.
MAX_DEGREE = 4

# Fewest links in a valid topology: a connected graph on N_ROUTERS nodes needs
# at least a spanning tree, i.e. N_ROUTERS - 1 = 8 edges (a spanning tree of 9
# nodes always fits within degree <= 4, e.g. a simple path).
MIN_CONNECTIONS = 8

# Most links in a valid topology: every router saturated at degree 4 gives
# 9 * 4 / 2 = 18 undirected edges.
MAX_CONNECTIONS = 18


def compute_adjacency(n_connections: int, n_routers: int = N_ROUTERS,
                      max_degree: int = MAX_DEGREE, rng: "np.random.Generator | None" = None,
                      max_tries: int = 200) -> "np.ndarray":
    """Build one random valid adjacency matrix (paper Algorithm 1).

    Returns an (n_routers, n_routers) int8 symmetric matrix, zero diagonal, with
    exactly `n_connections` undirected edges, every node degree <= max_degree, and the
    graph connected. Raises RuntimeError if no valid matrix is found within `max_tries`
    restarts (e.g. an infeasible (n_connections, max_degree) combination).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Edges to add on top of the spanning tree. Negative => fewer links than a
    # connected graph can have; every attempt below then fails and we raise.
    n_extra = n_connections - (n_routers - 1)

    for _ in range(max_tries):
        adj = np.zeros((n_routers, n_routers), dtype=np.int8)
        deg = np.zeros(n_routers, dtype=np.int64)

        # --- Phase 1: grow a random spanning tree so the graph is connected. ---
        # Repeatedly join a not-yet-connected router to a random connected
        # router that still has a spare port. If every connected router is
        # already saturated while unconnected ones remain, this attempt is a
        # dead end and we restart.
        order = [int(x) for x in rng.permutation(n_routers)]
        connected = [order[0]]
        unconnected = order[1:]
        dead_end = False
        while unconnected:
            candidates = [u for u in connected if deg[u] < max_degree]
            if not candidates:
                dead_end = True
                break
            u = int(rng.choice(candidates))
            v = unconnected.pop(int(rng.integers(len(unconnected))))
            adj[u, v] = adj[v, u] = 1
            deg[u] += 1
            deg[v] += 1
            connected.append(v)
        if dead_end or n_extra < 0:
            continue

        # --- Phase 2: add the remaining edges among currently-legal pairs. ---
        # A pair is legal when both endpoints are below max_degree and not yet
        # connected. If we run out of legal pairs before placing them all, the
        # attempt is stuck: restart.
        failed = False
        for _ in range(n_extra):
            avail = [int(i) for i in np.flatnonzero(deg < max_degree)]
            legal = [(i, j)
                     for pos, i in enumerate(avail)
                     for j in avail[pos + 1:]
                     if adj[i, j] == 0]
            if not legal:
                failed = True
                break
            i, j = legal[int(rng.integers(len(legal)))]
            adj[i, j] = adj[j, i] = 1
            deg[i] += 1
            deg[j] += 1
        if failed:
            continue

        return adj

    raise RuntimeError(
        f"No valid adjacency for n_connections={n_connections}, n_routers={n_routers}, "
        f"max_degree={max_degree} within {max_tries} restarts "
        "(the requested combination may be infeasible)."
    )


def generate_dataset(connection_counts: "Iterable[int]" = range(MIN_CONNECTIONS, MAX_CONNECTIONS + 1),
                     samples_per_class: int = 10000, n_routers: int = N_ROUTERS,
                     max_degree: int = MAX_DEGREE, seed: int | None = 0) -> dict:
    """Generate a connection-count-stratified dataset of unique topologies.

    For each target connection count, generates up to `samples_per_class` DISTINCT matrices
    (dedup by bytes across the whole dataset). Returns a dict with numpy arrays:
      "matrices"      int8  (N, n_routers, n_routers)
      "n_connections" int32 (N,)
      "n_routers"     int   (scalar, stored as 0-d array or python int)
    Logs a one-line-per-class progress summary (print) with the count actually produced
    (may be < samples_per_class if the class saturates its distinct-graph space).
    """
    rng = np.random.default_rng(seed)
    counts = list(connection_counts)

    seen: set[bytes] = set()          # global dedup across every class, keyed on raw bytes
    matrices: list[np.ndarray] = []
    labels: list[int] = []

    # Stop a class once this many consecutive draws add nothing new (a
    # duplicate, or an infeasible request): the distinct-graph space for that
    # connection count is effectively exhausted.
    max_consecutive_misses = max(1000, samples_per_class // 2)

    for c in counts:
        produced = 0
        misses = 0
        while produced < samples_per_class and misses < max_consecutive_misses:
            try:
                adj = compute_adjacency(c, n_routers=n_routers, max_degree=max_degree, rng=rng)
            except RuntimeError:
                # Infeasible (n_connections, max_degree) combination for this class.
                break
            key = adj.tobytes()
            if key in seen:
                misses += 1
                continue
            seen.add(key)
            matrices.append(adj)
            labels.append(c)
            produced += 1
            misses = 0
        print(f"  n_connections={c:2d}: {produced:6d} / {samples_per_class} unique topologies")

    stacked = (np.stack(matrices).astype(np.int8) if matrices
               else np.empty((0, n_routers, n_routers), dtype=np.int8))
    return {
        "matrices": stacked,
        "n_connections": np.asarray(labels, dtype=np.int32),
        "n_routers": int(n_routers),
    }


def save_dataset(data: dict, path: "str | Path") -> None:
    """Write a dataset dict to `path` as a compressed .npz (np.savez_compressed)."""
    path = Path(path)
    # Be forgiving about a not-yet-existing target directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)


@dataclass
class NoCDataset:
    """In-memory NoC topology dataset.

    matrices:       float32 (N, R, R), values in {0.0, 1.0}, symmetric, zero diagonal
    n_connections:  int32   (N,)
    n_routers:      int
    latency:        float32 (N,) or None   (never populated by this repo; simulator-only)
    """

    matrices: "np.ndarray"
    n_connections: "np.ndarray"
    n_routers: int
    latency: "np.ndarray | None" = None

    def __len__(self) -> int:
        """Number of topologies in the dataset."""
        return int(self.matrices.shape[0])

    def images(self) -> "np.ndarray":
        """(N, R, R, 1) float32 in {0,1} — the adjacency matrices as single-channel images."""
        return self.matrices[..., np.newaxis].astype(np.float32, copy=False)

    def scaled_images(self) -> "np.ndarray":
        """(N, R, R, 1) float32 in [-1, 1] — `images()` mapped x*2 - 1 to match the
        generator's tanh output range."""
        return self.images() * 2.0 - 1.0

    def connections_norm(self) -> "np.ndarray":
        """(N, 1) float32 in [-1, 1] — the number of connections min-max normalized with
        the fixed constants MIN_CONNECTIONS..MAX_CONNECTIONS:
        2*(n_connections - MIN_CONNECTIONS)/(MAX_CONNECTIONS - MIN_CONNECTIONS) - 1.
        This is the regression target of the reward network; the RWGAN generator's
        reward target (default 0.5) lives in this same normalized space."""
        span = float(MAX_CONNECTIONS - MIN_CONNECTIONS)
        norm = 2.0 * (self.n_connections.astype(np.float32) - MIN_CONNECTIONS) / span - 1.0
        return norm.reshape(-1, 1).astype(np.float32)


def load_dataset(path: "str | Path") -> NoCDataset:
    """Load a .npz written by `save_dataset` into a NoCDataset (matrices cast to float32)."""
    with np.load(str(path)) as npz:
        matrices = npz["matrices"].astype(np.float32)
        n_connections = npz["n_connections"].astype(np.int32)
        n_routers = int(npz["n_routers"])
        # `save_dataset` never writes a latency field; honour one only if some
        # other tool added it (this repo leaves it None).
        latency = npz["latency"].astype(np.float32) if "latency" in npz.files else None
    return NoCDataset(matrices=matrices, n_connections=n_connections,
                      n_routers=n_routers, latency=latency)


# ---- graph / validity helpers (numpy + stdlib only) ----

def adjacency_to_graph(matrix: "np.ndarray") -> "dict[int, list[int]]":
    """Adjacency matrix -> {node: [neighbours]} for nodes that have at least one edge."""
    graph: dict[int, list[int]] = {}
    for node in range(matrix.shape[0]):
        neighbours = [int(x) for x in np.flatnonzero(matrix[node])]
        if neighbours:
            graph[node] = neighbours
    return graph


def degrees(matrix: "np.ndarray") -> "np.ndarray":
    """Row sums (int) = per-router degree."""
    return matrix.sum(axis=1).astype(int)


def is_connected(matrix: "np.ndarray", n_routers: int = N_ROUTERS) -> bool:
    """True iff the `n_routers` nodes form a single connected component (BFS/DFS).
    An all-zero matrix is not connected."""
    if not np.any(matrix):
        return False

    # Iterative BFS from node 0 over the first `n_routers` nodes.
    visited = {0}
    queue = [0]
    while queue:
        node = queue.pop()
        for neighbour in np.flatnonzero(matrix[node]):
            neighbour = int(neighbour)
            if neighbour < n_routers and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return len(visited) == n_routers


def n_connections_of(matrix: "np.ndarray") -> int:
    """Number of undirected edges = matrix.sum() // 2 (assumes symmetric, zero diagonal)."""
    return int(matrix.sum() // 2)


def is_valid_topology(matrix: "np.ndarray", n_routers: int = N_ROUTERS,
                      max_degree: int = MAX_DEGREE) -> bool:
    """True iff: exactly `n_routers` nodes carry >=1 edge, every degree <= max_degree,
    and the graph is connected. (Symmetry is assumed; callers symmetrize first.)"""
    deg = degrees(matrix)
    if int(np.count_nonzero(deg)) != n_routers:
        return False
    if int(deg.max()) > max_degree:
        return False
    return is_connected(matrix, n_routers)


def binarize_symmetric(real_matrix: "np.ndarray", threshold: float) -> "np.ndarray":
    """Turn a real-valued generator output (R,R) into a symmetric binary matrix:
    edge (m,n) present iff real_matrix[m,n] + real_matrix[n,m] > threshold. Sets both
    [m,n] and [n,m]. Zero diagonal. Returns int8. `threshold` is typically 0.0, 0.5 or 1.0."""
    # Summing with the transpose both symmetrizes and applies the same rule to
    # (m,n) and (n,m) in one shot.
    combined = np.asarray(real_matrix, dtype=np.float64)
    combined = combined + combined.T
    out = (combined > threshold).astype(np.int8)
    np.fill_diagonal(out, 0)
    return out


def contains_matrix(matrix: "np.ndarray", dataset_matrices: "np.ndarray") -> bool:
    """True iff `matrix` is byte-equal to any matrix in `dataset_matrices` (N,R,R).
    Used as the novelty check ("is this generated topology already in the training set")."""
    if dataset_matrices.size == 0:
        return False
    if tuple(dataset_matrices.shape[1:]) != tuple(matrix.shape):
        return False
    # Compare per-sample; numpy promotes dtypes so an int8 query still matches a
    # float32 stored copy of the same 0/1 topology.
    axes = tuple(range(1, dataset_matrices.ndim))
    return bool(np.any(np.all(dataset_matrices == matrix, axis=axes)))
