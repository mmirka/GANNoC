# GANNoC demo notebooks

Two short, self-contained Jupyter notebooks that run the 9-router problem
end to end on a tiny in-memory dataset — fast enough for CPU (a few minutes
each).

| Notebook | Demonstrates |
|---|---|
| `01_wgan_gp_adjacency.ipynb` | Plain **WGAN-GP** — MLP generator + CNN critic learn to emit 9×9 binary symmetric adjacency matrices. `train_wgangp` keeps `LAMBDA` pinned at 1.0, so the reward term is never used. |
| `02_rwgan_reward_connections.ipynb` | **RWGAN** — the same WGAN-GP core plus a frozen connection-count reward CNN, with `LAMBDA` annealed down so the reward pulls the generator toward denser topologies. The generated connection-count distribution shifts above the WGAN-GP reference. |

Both notebooks **use the repo's own `gannoc` package** — cell 1 does
`sys.path.insert(0, "../src")` and imports `gannoc.data`, `gannoc.model`,
`gannoc.training`, `gannoc.evaluate` (and, in notebook 02,
`gannoc.reward_network`). Run them from this `demos/` directory in the
`gannoc` conda env:

```bash
conda activate gannoc
cd demos
jupyter lab        # or: jupyter notebook
```

Headless:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_wgan_gp_adjacency.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_rwgan_reward_connections.ipynb
```

The notebooks are shipped **unexecuted** (no output cells).

## Fast smoke-run knobs

Both notebooks are already sized for a smoke run and expose the knobs near the
top:

- **Dataset** — `samples_per_class=200` over connection counts 8..18
  (~2.2k topologies), written to `output/_demo_nocs.npz`.
- **Epochs** — `epochs=5` (notebook 01) / `epochs=8` (notebook 02), passed to
  `gannoc.training.train_wgangp` / `train_rwgan`. Raise for a less noisy result.
- **Notebook 02 anneal** — the paper's schedule (module constants in
  `gannoc/training.py`) starts the `LAMBDA` anneal at epoch 100 of 250; the
  notebook assigns `training.LAMBDA_START_EPOCH = 1`,
  `training.LAMBDA_DELTA = -0.1`, `training.LAMBDA_EVERY_N_EPOCHS = 1` so
  `LAMBDA` actually descends from 1.0 to the floor within 8 epochs. Every other
  hyper-parameter stays at its paper value.

## Expected outputs (`output/`)

| File | Written by | Expectation |
|---|---|---|
| `output/_demo_nocs.npz` | both | the shared tiny training set |
| `output/01_topology_samples.png` | 01 | a handful of generated 9-node graphs drawn with `networkx`; after only 5 epochs some are already connected and degree ≤ 4, others are not |
| `output/02_connection_histogram.png` | 02 | connection-count histogram of the RWGAN-generated topologies sitting to the **right** of the bundled WGAN-GP reference set (`../data/reference/generated_topologies/wgan_valid_topologies.pkl`, mean ≈ 12) — the reward term biased generation denser |

`evaluate_generator(...).summary()` is printed inline in both notebooks
(structural validity, novelty vs the training set, mean connection count). With
this deliberately short training the validity rate is modest and noisy — the
demos show the *mechanism*, not the paper's numbers. `output/` is git-ignored
(kept by `.gitkeep`).
