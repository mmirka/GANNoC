#!/usr/bin/env python3
"""Train the reward-guided RWGAN on 9x9 binary symmetric NoC adjacency matrices.

The same WGAN-GP core as ``scripts/train_wgangp.py`` plus a pretrained, frozen
reward network whose MSE against ``training.REWARD_TARGET`` is blended into the
generator loss. LAMBDA anneals from 1.0 (purely adversarial) down to
``LAMBDA_FLOOR``, handing a share of the generator gradient to the reward --
the paper's Phase 2 -> Phase 3 transition (RAPIDO 2021 section 4.3.4).

A pretrained reward pair ships in ``data/reference/reward_networks/``, so this
can be run without ``scripts/train_reward.py`` first.

Checkpoints land in ``<OUTPUT_DIR>/checkpoints/<RUN_NAME>/`` and the per-step
history in ``<OUTPUT_DIR>/logs/<RUN_NAME>/history.pkl``.

Edit the settings below, then run:

    python scripts/train_rwgan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gannoc import training  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings                                                                      #
# --------------------------------------------------------------------------- #
DATASET = "data/raw/nocs.npz"   # built by scripts/generate_dataset.py
# The checkpoint the reference implementation (RWGANgp_pretrainedR.py) loaded.
# Point this at results/checkpoints/reward/reward.h5 to use your own instead.
REWARD_CHECKPOINT = "data/reference/reward_networks/r_cnn_214.h5"
OUTPUT_DIR = "results"
RUN_NAME = "gannoc_rwgan"
EPOCHS = training.EPOCHS            # 250
LAMBDA_FLOOR = training.LAMBDA_FLOOR  # 0.9 => 10% reward / 90% critic at the end
FORCE_CPU = False


def main() -> int:
    """Run the RWGAN and print the saved artifacts."""
    artifacts = training.train_rwgan(
        dataset_path=DATASET,
        reward_checkpoint=REWARD_CHECKPOINT,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        epochs=EPOCHS,
        lambda_floor=LAMBDA_FLOOR,
        force_cpu=FORCE_CPU,
    )
    print("Saved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
