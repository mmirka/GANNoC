#!/usr/bin/env python3
"""Train the plain WGAN-GP baseline on 9x9 binary symmetric NoC adjacency matrices.

No reward network is involved: LAMBDA stays pinned at 1.0, so the generator is
trained purely adversarially. This is the "WGAN" of RAPIDO 2021 section 4.3.3 --
the reference point the RWGAN (``scripts/train_rwgan.py``) is compared against.

Checkpoints land in ``<OUTPUT_DIR>/checkpoints/<RUN_NAME>/`` and the per-step
history in ``<OUTPUT_DIR>/logs/<RUN_NAME>/history.pkl``.

Edit the settings below, then run:

    python scripts/train_wgangp.py
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
OUTPUT_DIR = "results"
RUN_NAME = "gannoc_wgangp"
EPOCHS = training.EPOCHS        # 250; the architecture and every other
                                # hyper-parameter live in src/gannoc/training.py
FORCE_CPU = False


def main() -> int:
    """Run the WGAN-GP baseline and print the saved artifacts."""
    artifacts = training.train_wgangp(
        dataset_path=DATASET,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        epochs=EPOCHS,
        force_cpu=FORCE_CPU,
    )
    print("Saved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
