#!/usr/bin/env python3
"""Generate a synthetic, connection-count-stratified NoC-topology dataset.

Builds valid 9-router topologies by random construction (RAPIDO 2021 / PhD
Thesis Chapter 5, "Algorithm 1"): each sample is a connected 9x9 binary
symmetric adjacency matrix with a zero diagonal and every router degree <= 4.
Samples are stratified by physical connection count, deduplicated across the
whole dataset, and written as a compressed .npz. The output is reloaded and
its structural invariants are re-checked before the script returns.

No NoC simulator is involved: this dataset carries graph structure only.
Latency labels would require an external NoC simulator and are out of scope.

Edit the settings below, then run:

    python scripts/generate_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gannoc.data import (  # noqa: E402
    MAX_CONNECTIONS,
    MAX_DEGREE,
    MIN_CONNECTIONS,
    N_ROUTERS,
    degrees,
    generate_dataset,
    is_connected,
    load_dataset,
    n_connections_of,
    save_dataset,
)

# --------------------------------------------------------------------------- #
# Settings                                                                      #
# --------------------------------------------------------------------------- #
# Inclusive range of physical connection counts to stratify over. 8..18 spans a
# bare spanning tree up to every router saturated at degree 4.
CONNECTION_RANGE = (MIN_CONNECTIONS, MAX_CONNECTIONS)
SAMPLES_PER_CLASS = 10000       # distinct topologies per connection count
SEED = 0
OUTPUT = "data/raw/nocs.npz"    # relative to the repo root


def _verify(path: "str | Path") -> bool:
    """Reload the saved dataset and re-check every structural invariant.

    Prints an N-samples line, a per-connection-count histogram, and a PASS/FAIL
    summary. Returns True iff every sample is symmetric, zero-diagonal, within
    the degree cap, connected, and carries its recorded connection count.
    """
    ds = load_dataset(path)
    print(f"Reloaded {len(ds)} samples, n_routers={ds.n_routers}")

    values, hist = np.unique(ds.n_connections, return_counts=True)
    print("Histogram (n_connections: count):")
    for value, count in zip(values, hist):
        print(f"  {int(value):2d}: {int(count)}")

    issues = {
        "asymmetric": 0,
        "nonzero_diagonal": 0,
        "degree_over_cap": 0,
        "disconnected": 0,
        "wrong_n_connections": 0,
    }
    for idx in range(len(ds)):
        matrix = ds.matrices[idx]
        if not np.array_equal(matrix, matrix.T):
            issues["asymmetric"] += 1
        if np.any(np.diagonal(matrix) != 0):
            issues["nonzero_diagonal"] += 1
        if int(degrees(matrix).max()) > MAX_DEGREE:
            issues["degree_over_cap"] += 1
        if not is_connected(matrix, N_ROUTERS):
            issues["disconnected"] += 1
        if n_connections_of(matrix) != int(ds.n_connections[idx]):
            issues["wrong_n_connections"] += 1

    passed = all(count == 0 for count in issues.values())
    print("Structural checks:", "PASS" if passed else "FAIL")
    if not passed:
        for name, count in issues.items():
            if count:
                print(f"  {name}: {count} sample(s)")
    return passed


def main() -> int:
    """Generate the dataset, save it, reload it, and print a verification summary."""
    low, high = CONNECTION_RANGE
    connection_counts = list(range(low, high + 1))

    print(
        f"Generating up to {SAMPLES_PER_CLASS} topologies per connection count "
        f"in {low}..{high} ({N_ROUTERS} routers, degree <= {MAX_DEGREE}, "
        f"seed={SEED}):"
    )
    data = generate_dataset(
        connection_counts=connection_counts,
        samples_per_class=SAMPLES_PER_CLASS,
        n_routers=N_ROUTERS,
        max_degree=MAX_DEGREE,
        seed=SEED,
    )

    out_path = Path(OUTPUT)
    save_dataset(data, out_path)
    print(f"\nWrote {out_path} ({data['matrices'].shape[0]} samples)\n")

    return 0 if _verify(out_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
