# Paper figures

Self-contained reproduction of the GANNoC figures from RAPIDO 2021 / the PhD
thesis Chapter 5. The scripts use only `numpy` + `matplotlib` (+ `networkx`
for `topology_grid.py`) and a small vendored `noc_metrics.py` for the
graph/validity helpers: **no `gannoc` import**, no TensorFlow, no external
download. Any environment with those packages runs them; the repo's `gannoc`
conda env works out of the box.

Run everything from this directory:

```
cd figures
```

Each script has a short settings block at the top (input paths, `OUTPUT_DIR`,
and for `topology_grid.py` the style / sample count). Figures are written to
`figures/output/` by default.

## Figure → script → data

| Figure | Script (from `figures/`) | Data consumed |
|---|---|---|
| **Ch. 5**: WGAN vs. RWGAN connection-count distribution | `connection_histogram.py` | `../data/reference/generated_topologies/wgan_valid_topologies.pkl` + `../data/reference/generated_topologies/rwgan_valid_topologies.pkl` |
| **Topology gallery**: grid of generated 9-router topologies, titled by link count | `topology_grid.py` (`STYLE` = graph / matrix / both) | `../data/reference/generated_topologies/rwgan_valid_topologies.pkl` (set `INPUT` to `wgan_valid_topologies.pkl` for the baseline) |
| **Training diagnostics**: critic/generator/reward loss + reward-guidance channels | `training_curves.py` (set `HISTORY_PATH`) | a `results/logs/<run>/history.pkl` from *your own* training run |

`connection_histogram.py` is the guaranteed-reproducible figure: it needs
only the two bundled pickles. `training_curves.py` has no bundled input by
design: it documents a run you produced yourself.

## Regenerate

```
cd figures

# Ch. 5: WGAN vs. RWGAN connection-count distribution (mean ~12 vs. ~14)
python connection_histogram.py

# Topology gallery: 25 RWGAN topologies, both graph and matrix views
python topology_grid.py

# ...only the graph view, or the WGAN baseline set: edit STYLE / N_SAMPLES /
# INPUT at the top of topology_grid.py

# Training diagnostics: point HISTORY_PATH at one of your own runs
python training_curves.py
```

## What the bundled data is

- **`../data/reference/generated_topologies/wgan_valid_topologies.pkl`** and
  **`../data/reference/generated_topologies/rwgan_valid_topologies.pkl`**:
  the paper's WGAN- and RWGAN-generated *valid* 9-router topologies (446
  each), drawn from identical noise seeds so the two sets are directly
  comparable. Each is a pickled `np.ndarray` of shape `(446, 9, 9)`,
  entries in `{0, 1}`, symmetric, zero diagonal, every router degree ≤ 4,
  every topology connected. `connection_histogram.py` counts the physical
  links per topology (the 1s above the diagonal) and shows that the
  reward-guided RWGAN shifts the distribution toward denser, lower-latency
  topologies. `topology_grid.py` draws a sample of them.
- **Training histories**: `training_curves.py` reads a `history.pkl`
  written by `gannoc.training.train_wgangp` / `train_rwgan` under `results/logs/<run>/`: a dict of
  equal-length per-step lists (`step`, `epoch`, `lambda`, `critic_loss`,
  `critic_real`, `critic_fake`, `critic_gp`, `generator_wasserstein`,
  `reward_mse`, `mean_reward_output`, `mean_critic_output`). None is
  bundled, run a training first, then pass its `history.pkl`.

The thesis saturation-curve figures (5.8 / 5.9) plot simulated NoC latency
against injection rate and need an external, cycle-accurate NoC simulator to
produce their data. That is out of scope for this clean-room repo, which
carries graph structure only, so those figures are not reproduced here.
