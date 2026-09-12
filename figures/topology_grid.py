#!/usr/bin/env python3
"""Draw a gallery grid of generated NoC topologies.

RAPIDO 2021 / PhD thesis Chapter 5. Samples a handful of generated 9-router
topologies and lays them out on a square grid, either as ``networkx``
spring-layout graph drawings, as adjacency-matrix heatmaps, or both (two
separate figures). Each cell is titled with that topology's physical
connection count, so a denser RWGAN set reads at a glance next to a WGAN one.

Self-contained: ``numpy`` + ``matplotlib`` + ``networkx`` only, no ``gannoc``
import.

Edit the settings below (which topology set, style, how many), then run:

    cd figures
    python topology_grid.py
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from noc_metrics import N_ROUTERS, n_connections_of  # noqa: E402

STYLES = ("graph", "matrix", "both")

# ------------------------------- settings ---------------------------------- #
# Swap for wgan_valid_topologies.pkl to draw the WGAN-GP baseline set instead.
INPUT = "../data/reference/generated_topologies/rwgan_valid_topologies.pkl"
STYLE = "both"      # one of STYLES; "both" writes two figures
N_SAMPLES = 25      # grid side = ceil(sqrt(N_SAMPLES))
SEED = 0            # sample selection and spring layout
OUTPUT_DIR = "output"


def load_topologies(path: str) -> np.ndarray:
    """Load a pickled ``np.ndarray`` of shape ``(N, 9, 9)``; fail loudly if absent."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Topology pickle not found: {p}\n"
            "Pass --input explicitly, or run this from inside figures/ so the "
            "default ../data/reference/generated_topologies/ path resolves."
        )
    with open(p, "rb") as f:
        matrices = np.asarray(pickle.load(f))
    if matrices.ndim != 3 or matrices.shape[1:] != (N_ROUTERS, N_ROUTERS):
        raise ValueError(
            f"{p}: expected an (N, {N_ROUTERS}, {N_ROUTERS}) array, got {matrices.shape}"
        )
    return matrices


def pick_samples(matrices: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Choose up to ``n`` topologies without replacement, deterministically."""
    n = min(n, len(matrices))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(matrices), size=n, replace=False))
    return matrices[idx]


def _blank_axes(axes, used: int) -> None:
    """Hide the trailing axes of a grid that hold no topology."""
    for ax in axes.flat[used:]:
        ax.axis("off")


def plot_graph_grid(samples: np.ndarray, seed: int, output_dir: Path) -> Path:
    side = math.ceil(math.sqrt(len(samples)))
    fig, axes = plt.subplots(side, side, figsize=(2.2 * side, 2.2 * side), squeeze=False)
    for ax, matrix in zip(axes.flat, samples):
        graph = nx.from_numpy_array(np.triu(np.asarray(matrix), k=1))
        # A fixed seed keeps the spring layout stable run-to-run.
        pos = nx.spring_layout(graph, seed=seed)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#888888")
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=90, node_color="#4c72b0")
        ax.set_title(f"{n_connections_of(matrix)} links", fontsize=9)
        ax.axis("off")
    _blank_axes(axes, len(samples))
    fig.suptitle("Generated NoC topologies (graph view)")
    fig.tight_layout()
    out_path = output_dir / "topology_grid_graph.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_matrix_grid(samples: np.ndarray, output_dir: Path) -> Path:
    side = math.ceil(math.sqrt(len(samples)))
    fig, axes = plt.subplots(side, side, figsize=(2.0 * side, 2.0 * side), squeeze=False)
    for ax, matrix in zip(axes.flat, samples):
        ax.imshow(np.asarray(matrix), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"{n_connections_of(matrix)} links", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    _blank_axes(axes, len(samples))
    fig.suptitle("Generated NoC topologies (adjacency matrix)")
    fig.tight_layout()
    out_path = output_dir / "topology_grid_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    """Draw the configured sample of topologies as graph and/or matrix grids."""
    if STYLE not in STYLES:
        print(f"ERROR: STYLE must be one of {STYLES}, got {STYLE!r}", file=sys.stderr)
        return 1
    if N_SAMPLES < 1:
        print("ERROR: N_SAMPLES must be >= 1", file=sys.stderr)
        return 1

    try:
        matrices = load_topologies(INPUT)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pick_samples(matrices, N_SAMPLES, SEED)
    print(f"drawing {len(samples)} of {len(matrices)} topologies from {INPUT}")

    if STYLE in ("graph", "both"):
        print(f"wrote {plot_graph_grid(samples, SEED, output_dir)}")
    if STYLE in ("matrix", "both"):
        print(f"wrote {plot_matrix_grid(samples, output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
