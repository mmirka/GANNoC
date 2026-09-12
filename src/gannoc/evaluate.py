"""Quality metrics for a trained GANNoC generator.

Samples a batch of latent vectors, runs them through a generator, binarizes
each candidate 9x9 adjacency matrix at one or more thresholds, and aggregates
how many samples are structurally valid NoC topologies (nine active routers,
router degree <= 4, connected), how many of those are novel with respect to the
training set, and what their physical connection-count distribution looks like.

Everything here is pure NumPy apart from the forward pass through the passed-in
``tensorflow.keras`` generator model; this module never imports TensorFlow
itself, so it stays cheap to import from plotting/demo code.

RAPIDO 2021 / PhD Thesis Chapter 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gannoc.data import (
    MAX_DEGREE,
    N_ROUTERS,
    binarize_symmetric,
    contains_matrix,
    degrees,
    is_connected,
    n_connections_of,
)


@dataclass
class GeneratorMetrics:
    """Aggregated quality of a batch of generated topologies, per binarization threshold.

    thresholds are evaluated on real_matrix[m,n] + real_matrix[n,m] (typically 0.0/0.5/1.0).
    For each threshold t the following are stored keyed by t:
      n_nine_router[t]   int    # samples whose binarized matrix has exactly n_routers active nodes
      n_feasible[t]      int    # samples: all degrees <= max_degree AND connected
      n_valid[t]         int    # samples: nine-router AND feasible
      n_novel[t]         int    # of the valid ones NOT byte-equal to any training matrix
      connection_counts[t]  list[int]  # connection count of each VALID sample
      asymmetry_mean[t]  float  # mean over samples of mean(|real[m,n]-real[n,m]|)
      asymmetry_std[t]   float
    """

    n_samples: int
    thresholds: tuple
    n_nine_router: dict = field(default_factory=dict)
    n_feasible: dict = field(default_factory=dict)
    n_valid: dict = field(default_factory=dict)
    n_novel: dict = field(default_factory=dict)
    connection_counts: dict = field(default_factory=dict)
    asymmetry_mean: dict = field(default_factory=dict)
    asymmetry_std: dict = field(default_factory=dict)

    def valid_rate(self, threshold: float) -> float:
        """Fraction of all sampled topologies that are valid at ``threshold``."""
        return self.n_valid[threshold] / self.n_samples

    def novelty_rate(self, threshold: float) -> float:
        """Fraction of the *valid* topologies absent from the training set.

        Denominator is floored at 1 so a run that produced no valid samples
        reports 0.0 rather than dividing by zero.
        """
        return self.n_novel[threshold] / max(self.n_valid[threshold], 1)

    def mean_connections(self, threshold: float) -> float:
        """Mean physical connection count over the valid topologies (nan if none)."""
        counts = self.connection_counts[threshold]
        return float(np.mean(counts)) if counts else float("nan")

    def summary(self) -> str:
        """A few readable lines, one block per binarization threshold."""
        lines = [f"GeneratorMetrics over {self.n_samples} samples:"]
        for threshold in self.thresholds:
            lines.append(f"  threshold {threshold:g}:")
            lines.append(
                f"    nine-router={self.n_nine_router[threshold]}  "
                f"feasible={self.n_feasible[threshold]}  "
                f"valid={self.n_valid[threshold]} ({self.valid_rate(threshold):.1%})"
            )
            lines.append(
                f"    novel={self.n_novel[threshold]} "
                f"({self.novelty_rate(threshold):.1%} of valid)  "
                f"mean_connections={self.mean_connections(threshold):.2f}"
            )
            lines.append(
                f"    asymmetry={self.asymmetry_mean[threshold]:.4f} "
                f"+/- {self.asymmetry_std[threshold]:.4f}"
            )
        return "\n".join(lines)


def _as_matrix_batch(raw) -> np.ndarray:
    """Coerce a generator output to an ``(N, R, R)`` float array.

    The generator emits ``(N, R, R, 1)`` single-channel images; the trailing
    channel axis is always singleton, so we simply drop it. A generator that
    already emits ``(N, R, R)`` is passed through untouched.
    """
    batch = np.asarray(raw, dtype=np.float32)
    if batch.ndim == 4:
        batch = batch[..., 0]
    return batch


def evaluate_generator(generator, training_matrices: "np.ndarray", n_samples: int = 1000,
                       latent_dim: int = 100, thresholds: "tuple[float, ...]" = (0.0, 0.5, 1.0),
                       seed: "int | None" = 0) -> GeneratorMetrics:
    """Sample `n_samples` noise vectors (np.random.default_rng(seed).random, UNIFORM [0,1);
    the paper uses uniform latent noise, not Gaussian), run generator.predict, and compute a
    GeneratorMetrics. `training_matrices` is (M,R,R) {0,1} for the novelty check.
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((n_samples, latent_dim), dtype=np.float32)

    generated = _as_matrix_batch(generator.predict(noise))
    training_matrices = np.asarray(training_matrices)

    metrics = GeneratorMetrics(n_samples=n_samples, thresholds=tuple(thresholds))

    # Asymmetry of the raw, pre-binarization generator output: how far the
    # network's matrix is from being symmetric before we force symmetry in
    # binarize_symmetric. This does not depend on the threshold, but it is
    # stored per-threshold so every column of the summary table has the same
    # shape.
    per_sample_asymmetry = np.array(
        [float(np.mean(np.abs(matrix - matrix.T))) for matrix in generated],
        dtype=np.float64,
    )
    asymmetry_mean = float(np.mean(per_sample_asymmetry)) if n_samples else float("nan")
    asymmetry_std = float(np.std(per_sample_asymmetry)) if n_samples else float("nan")

    for threshold in metrics.thresholds:
        n_nine_router = 0
        n_feasible = 0
        n_valid = 0
        n_novel = 0
        connection_counts = []

        for raw_matrix in generated:
            # binarize_symmetric thresholds real[m,n] + real[n,m], so the
            # result is symmetric with a zero diagonal by construction.
            binary = binarize_symmetric(raw_matrix, threshold)
            node_degrees = degrees(binary)

            nine_router = int(np.count_nonzero(node_degrees)) == N_ROUTERS
            feasible = bool(np.all(node_degrees <= MAX_DEGREE)) and is_connected(
                binary, N_ROUTERS
            )

            if nine_router:
                n_nine_router += 1
            if feasible:
                n_feasible += 1
            if nine_router and feasible:
                n_valid += 1
                connection_counts.append(int(n_connections_of(binary)))
                if not contains_matrix(binary, training_matrices):
                    n_novel += 1

        metrics.n_nine_router[threshold] = n_nine_router
        metrics.n_feasible[threshold] = n_feasible
        metrics.n_valid[threshold] = n_valid
        metrics.n_novel[threshold] = n_novel
        metrics.connection_counts[threshold] = connection_counts
        metrics.asymmetry_mean[threshold] = asymmetry_mean
        metrics.asymmetry_std[threshold] = asymmetry_std

    return metrics


def sample_topologies(generator, n_samples: int = 100, latent_dim: int = 100,
                      threshold: float = 0.5, seed: "int | None" = 0) -> "np.ndarray":
    """Generate and binarize `n_samples` topologies -> int8 (n_samples, R, R) symmetric.
    Convenience for figures/demos.
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((n_samples, latent_dim), dtype=np.float32)

    generated = _as_matrix_batch(generator.predict(noise))
    return np.stack(
        [binarize_symmetric(matrix, threshold) for matrix in generated]
    ).astype(np.int8)
