"""Training loop for the GANNoC reward network.

The reward network is a CNN regressor that maps a NoC adjacency-matrix image
(rescaled to ``[-1, 1]``) to its normalized physical connection count. Once
trained it is frozen and re-used inside :mod:`gannoc.training`, where its MSE
against a target connection count is annealed into the generator objective (the
"RWGAN" variant). The plain WGAN-GP baseline does not use it.

The architecture is :func:`gannoc.reward_network.build_reward_net`, i.e. the
critic's, as specified in RAPIDO 2021 section 4.3.2 / PhD Thesis table
``tab:NN:RWGAN``.

Training data requirement
-------------------------
Provide a dataset built by ``scripts/generate_dataset.py`` -- a compressed
``.npz`` loadable via :func:`gannoc.data.load_dataset`. The network regresses
``NoCDataset.connections_norm()`` (a scalar in ``[-1, 1]``) from
``NoCDataset.scaled_images()`` (the adjacency image in ``[-1, 1]``).
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Unlike gannoc.training there is no WGAN-GP gradient-penalty functional model
# here, so eager execution is left untouched: compiled ``train_on_batch`` /
# ``test_on_batch`` behave identically in eager and graph mode. (The RWGAN
# process disables eager before importing this module; that is harmless too.)

from .data import load_dataset
from .gpu import configure_gpu
from .reward_network import build_reward_net

# --------------------------------------------------------------------------- #
# Hyper-parameters. Edit here to experiment.                                    #
# --------------------------------------------------------------------------- #
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-4        # Adam
TRAIN_FRACTION = 0.8        # rest is the held-out split
SEED = None                 # set an int for a reproducible run


def train(
    dataset_path: "str | Path",
    output_dir: "str | Path",
    epochs: int = EPOCHS,
    force_cpu: bool = False,
) -> dict:
    """Train the reward network to regress ``connections_norm()`` from ``scaled_images()``.

    MSE with ``Adam(LEARNING_RATE)``, a manual epoch loop over shuffled batches
    using ``train_on_batch``, and after each epoch a ``test_on_batch`` over the
    held-out split. Saves the model to
    ``<output_dir>/checkpoints/reward/reward.h5`` and a pickled history dict
    ``{"train_mse": [...], "test_mse": [...]}`` to
    ``<output_dir>/logs/reward/history.pkl``. Returns
    ``{"checkpoint", "history", "final_test_mse"}``.
    """
    print(configure_gpu(force_cpu=force_cpu))

    if SEED is not None:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}\n"
            "Build one with scripts/generate_dataset.py (RAPIDO 2021 / PhD "
            "Thesis Chapter 5)."
        )

    data = load_dataset(dataset_path)
    images = data.scaled_images().astype(np.float32)       # (N, R, R, 1) in [-1, 1]
    targets = data.connections_norm().astype(np.float32)   # (N, 1) in [-1, 1]

    # One deterministic shuffle, then a fixed train / held-out split.
    order = np.random.permutation(len(data))
    images, targets = images[order], targets[order]
    split = int(TRAIN_FRACTION * len(data))
    x_train, x_test = images[:split], images[split:]
    y_train, y_test = targets[:split], targets[split:]

    model = build_reward_net(n_routers=data.n_routers)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    history = {"train_mse": [], "test_mse": []}
    n_batches = max(len(x_train) // BATCH_SIZE, 1)

    start = time.time()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

        batch_losses = []
        for b in range(n_batches):
            lo, hi = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
            batch_losses.append(float(model.train_on_batch(x_train[lo:hi], y_train[lo:hi])))

        train_mse = float(np.mean(batch_losses))
        # The reward net is small enough to score the whole held-out split in
        # one call; loss-only compile means test_on_batch returns a scalar.
        test_mse = float(model.test_on_batch(x_test, y_test))
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        print(f"epoch {epoch}/{epochs}  train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")

    print(f"Reward-network training complete in {time.time() - start:.1f}s")

    return _save_reward(output_dir, model, history, history["test_mse"][-1])


def _save_reward(output_dir, model, history: dict, final_test_mse: float) -> dict:
    """Persist the trained reward net and its loss history; return their paths."""
    checkpoint_dir = Path(output_dir) / "checkpoints" / "reward"
    log_dir = Path(output_dir) / "logs" / "reward"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "reward.h5"
    history_path = log_dir / "history.pkl"

    model.save(checkpoint_path)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "final_test_mse": float(final_test_mse),
    }
