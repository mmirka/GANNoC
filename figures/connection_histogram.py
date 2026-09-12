#!/usr/bin/env python3
"""Reproduce the WGAN vs. RWGAN connection-count distribution (Chapter 5).

RAPIDO 2021 / PhD thesis Chapter 5. The plain WGAN-GP baseline and the
reward-guided RWGAN are each sampled from the *same* noise seeds; the reward
network steers RWGAN toward denser (more-connected, lower-latency) 9-router
topologies. This script overlays the two link-count histograms and annotates
each with its mean. The paper's headline result is that the RWGAN mean sits
clearly higher (~14 links vs. ~12 for the baseline).

This is the guaranteed-reproducible figure of the repo: it needs only the two
bundled pickles of already-generated valid topologies, ``numpy`` and
``matplotlib``, no training run and no ``gannoc`` import.

Edit the settings below if needed, then run:

    cd figures
    python connection_histogram.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from noc_metrics import MAX_CONNECTIONS, MIN_CONNECTIONS, n_connections_of  # noqa: E402

HERE = Path(__file__).resolve().parent

# ------------------------------- settings ---------------------------------- #
WGAN_PKL = "../data/reference/generated_topologies/wgan_valid_topologies.pkl"
RWGAN_PKL = "../data/reference/generated_topologies/rwgan_valid_topologies.pkl"
OUTPUT_DIR = "output"


def load_topologies(path: str) -> np.ndarray:
    """Load a pickled ``np.ndarray`` of shape ``(N, 9, 9)``; fail loudly if absent."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Topology pickle not found: {p}\n"
            "Pass --wgan / --rwgan explicitly, or run this from inside figures/ so "
            "the default ../data/reference/generated_topologies/ paths resolve."
        )
    with open(p, "rb") as f:
        matrices = np.asarray(pickle.load(f))
    if matrices.ndim != 3 or matrices.shape[1:] != (9, 9):
        raise ValueError(f"{p}: expected an (N, 9, 9) array, got shape {matrices.shape}")
    return matrices


def connection_counts(matrices: np.ndarray) -> np.ndarray:
    """Per-sample number of physical links."""
    return np.array([n_connections_of(m) for m in matrices], dtype=int)


def plot_histogram(
    wgan_counts: np.ndarray, rwgan_counts: np.ndarray, output_dir: Path
) -> Path:
    """Side-by-side bar histogram over the integer bins ``MIN..MAX_CONNECTIONS``."""
    bins = np.arange(MIN_CONNECTIONS, MAX_CONNECTIONS + 1)
    # bincount then slice to the fixed bin range so both series share an x-axis
    # even when one of them never hits the extreme counts.
    wgan_hist = np.bincount(wgan_counts, minlength=MAX_CONNECTIONS + 1)[bins]
    rwgan_hist = np.bincount(rwgan_counts, minlength=MAX_CONNECTIONS + 1)[bins]

    wgan_mean = float(wgan_counts.mean())
    rwgan_mean = float(rwgan_counts.mean())

    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(bins - width / 2, wgan_hist, width, label=f"WGAN (mean {wgan_mean:.2f})",
           color="#4c72b0")
    ax.bar(bins + width / 2, rwgan_hist, width, label=f"RWGAN (mean {rwgan_mean:.2f})",
           color="#c44e52")
    ax.axvline(wgan_mean, color="#4c72b0", linestyle="--", linewidth=1.5)
    ax.axvline(rwgan_mean, color="#c44e52", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Number of physical connections")
    ax.set_ylabel("Number of generated topologies")
    ax.set_title("WGAN vs. RWGAN connection-count distribution (Chapter 5)")
    ax.set_xticks(bins)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "connection_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    """Load both bundled topology sets, print their means, write the overlay."""
    try:
        wgan = load_topologies(WGAN_PKL)
        rwgan = load_topologies(RWGAN_PKL)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    wgan_counts = connection_counts(wgan)
    rwgan_counts = connection_counts(rwgan)
    print(f"WGAN : {len(wgan_counts):4d} topologies, mean {wgan_counts.mean():.3f} connections")
    print(f"RWGAN: {len(rwgan_counts):4d} topologies, mean {rwgan_counts.mean():.3f} connections")

    out_path = plot_histogram(wgan_counts, rwgan_counts, output_dir)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
