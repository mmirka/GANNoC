#!/usr/bin/env python3
"""Plot loss and reward diagnostics from a GANNoC training run.

RAPIDO 2021 / PhD thesis Chapter 5. ``gannoc.training.train_wgangp`` and
``train_rwgan`` write a per-step history pickle to
``results/logs/<run>/history.pkl``, a dict of
equal-length lists keyed by:

    step, epoch, lambda, critic_loss, critic_real, critic_fake, critic_gp,
    generator_wasserstein, reward_mse, mean_reward_output, mean_critic_output

This script turns one such pickle into two PNGs:

- ``<name>_losses.png``:  critic_loss, critic_gp, generator_wasserstein and
  reward_mse against training step.
- ``<name>_reward.png``:  lambda, mean_reward_output and mean_critic_output
  against training step (the reward-guidance channels).

Missing optional keys are skipped with a note; a missing file or a pickle
with no recognizable keys is a hard error.

Self-contained: ``numpy`` + ``matplotlib`` only, no ``gannoc`` import. It reads
runs *you* produced under ``results/logs/``, nothing pre-generated is bundled.

Point ``HISTORY_PATH`` below at one of your runs, then:

    cd figures
    python training_curves.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ------------------------------- settings ---------------------------------- #
# A history.pkl from one of *your* runs; nothing is bundled.
HISTORY_PATH = "../results/logs/gannoc_rwgan/history.pkl"
OUTPUT_DIR = "output"
NAME = None     # output basename; None -> the history file's parent dir name

# Every key the training loop may log, so an unrecognizable pickle can be rejected.
KNOWN_KEYS = (
    "step", "epoch", "lambda", "critic_loss", "critic_real", "critic_fake",
    "critic_gp", "generator_wasserstein", "reward_mse", "mean_reward_output",
    "mean_critic_output",
)
LOSS_KEYS = ("critic_loss", "critic_gp", "generator_wasserstein", "reward_mse")
REWARD_KEYS = ("lambda", "mean_reward_output", "mean_critic_output")


def require_history(history_path: str) -> dict:
    """Load and sanity-check the history pickle; raise loudly on any problem."""
    p = Path(history_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"History pickle not found: {p}\n"
            "gannoc.training.train writes it to results/logs/<run>/history.pkl. "
            "Point --history at one of your own runs."
        )
    with open(p, "rb") as f:
        history = pickle.load(f)
    if not isinstance(history, dict) or not (set(history) & set(KNOWN_KEYS)):
        raise ValueError(
            f"{p}: not a recognizable GANNoC history dict (expected keys such as "
            f"{', '.join(LOSS_KEYS)})."
        )
    return history


def _x_axis(history: dict) -> np.ndarray:
    """Training step for the x-axis, falling back to a plain sample index."""
    if history.get("step"):
        return np.asarray(history["step"])
    any_series = next(v for v in history.values() if v is not None)
    return np.arange(len(any_series))


def _plot_series(history: dict, keys, x, title: str, ylabel: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for key in keys:
        series = history.get(key)
        if series is None or len(series) == 0:
            print(f"note: skipping {key!r} (missing from history)")
            continue
        ax.plot(x[: len(series)], np.asarray(series)[: len(x)], label=key)
        plotted += 1
    if not plotted:
        plt.close(fig)
        raise ValueError(f"none of {list(keys)} present in history, nothing to plot for {title}")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    """Plot the configured history pickle into the two diagnostic figures."""
    try:
        history = require_history(HISTORY_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = NAME or Path(HISTORY_PATH).resolve().parent.name

    x = _x_axis(history)
    losses_path = _plot_series(
        history, LOSS_KEYS, x, f"Training losses ({name})", "loss value",
        output_dir / f"{name}_losses.png",
    )
    print(f"wrote {losses_path}")
    reward_path = _plot_series(
        history, REWARD_KEYS, x, f"Reward-guidance channels ({name})", "value",
        output_dir / f"{name}_reward.png",
    )
    print(f"wrote {reward_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
