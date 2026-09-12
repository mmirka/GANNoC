#!/usr/bin/env python3
"""Train the GANNoC reward network.

The reward network is a CNN regressor that scores a NoC adjacency-matrix image
by its normalized physical connection count. Once trained it is frozen and
steers the RWGAN generator (see ``scripts/train_rwgan.py``); the plain WGAN-GP
baseline does not need it.

Its architecture is the critic's, per RAPIDO 2021 section 4.3.2 / PhD Thesis
table ``tab:NN:RWGAN``. See ``src/gannoc/reward_training.py`` for the loop.

Requires a dataset built by ``scripts/generate_dataset.py``. Edit the settings
below, then run:

    python scripts/train_reward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gannoc import reward_training  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings                                                                      #
# --------------------------------------------------------------------------- #
DATASET = "data/raw/nocs.npz"   # built by scripts/generate_dataset.py
OUTPUT_DIR = "results"          # -> results/checkpoints/reward/reward.h5
EPOCHS = reward_training.EPOCHS  # 15; the remaining hyper-parameters live in
                                 # src/gannoc/reward_training.py
FORCE_CPU = False


def main() -> int:
    """Train the reward network and print where it was saved."""
    result = reward_training.train(
        dataset_path=DATASET,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        force_cpu=FORCE_CPU,
    )
    print("Saved reward network:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
