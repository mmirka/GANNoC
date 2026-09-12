# GANNoC: Reward-guided generation of Network-on-Chip topologies

**GANNoC** casts Network-on-Chip (NoC) topology design as image generation. A
topology over 9 routers is a 9×9 binary, symmetric adjacency matrix (a link
between router *i* and *j*), handled as a single-channel image. A router exposes
4 external ports (N/E/S/W) plus one local port, so node degree is capped at 4; a
usable topology is a connected graph within that cap. A **WGAN-GP** (MLP
generator + CNN critic) learns to generate such matrices.

**RWGAN** ("Reward-Wasserstein GAN") adds one signal on top of that WGAN-GP: a
separately pretrained, **frozen** CNN that regresses a topology's normalized
number of physical connections. Its error against a target connection count is
folded into the generator loss with a weight `LAMBDA` that is annealed from
"critic only" toward "reward matters", biasing generation toward denser
topologies, which, under uniform traffic, correlate with lower packet latency.
Setting the `LAMBDA` floor to `1.0` removes the reward term and recovers the
plain WGAN-GP baseline. RWGAN is the single-objective precursor to **M-RWGAN**
(PhD thesis Chapter 6), which generalizes the same frozen-reward mechanism to
several simultaneous objectives.

This repository is a clean-room reproduction of the method in **RAPIDO 2021**
("GANNoC: A Framework for Automatic Generation of NoC Topologies using GANs")
and **PhD thesis, Chapter 5**. It rebuilds the architecture, the synthetic
dataset generator, and the training and evaluation code; it does **not**
regenerate the paper's headline numbers (see below).

## What this repository reproduces

- **Data representation**: a NoC topology as a 9×9 binary symmetric adjacency
  matrix / single-channel image; validity = every router wired, degree ≤ 4,
  graph connected (`gannoc.data`).
- **Synthetic dataset generator**: connection-count-stratified valid topologies
  built by random construction (paper "Algorithm 1"), pure Python, no simulator
  (`gannoc.data.generate_dataset`, `scripts/generate_dataset.py`).
- **WGAN-GP architecture**: MLP generator + CNN critic, Wasserstein loss,
  two-sided gradient penalty on random real/fake interpolates (`gannoc.model`,
  `gannoc.losses`).
- **Connection-count reward + annealed blending**: a frozen CNN regressor
  (`gannoc.reward_network`) whose MSE against a target is mixed into the
  generator loss through the `LAMBDA` schedule (`gannoc.losses.LambdaSchedule`);
  `LAMBDA` floor `1.0` ⇒ WGAN-GP, lower ⇒ stronger reward.
- **RWGAN training loop**: graph-mode WGAN-GP with the reward branch attached
  (`gannoc.training`, `scripts/train_rwgan.py`; the baseline is
  `scripts/train_wgangp.py`).
- **Generator evaluation**: structural validity, novelty vs the training set,
  and the connection-count distribution of generated topologies
  (`gannoc.evaluate`), plus comparison figures (`figures/`).

### Not included / out of scope

- **Latency and saturation-curve results.** The paper's latency improvement
  (quoted below) is measured with an external cycle-accurate NoC simulator
  (e.g. Ratatoskr, an optional, non-bundled dependency). Nothing here calls a
  simulator, and no dataset produced here carries latency labels.
- **The paper's exact trained weights.** Retrain locally with
  `scripts/train_wgangp.py` / `scripts/train_rwgan.py`. Pretrained *reward*
  networks are shipped
  (`data/reference/reward_networks/`); trained *generator* weights are not.
- **Any headline-number reproduction.** Validity / novelty / connection-count
  behaviour is reproducible; the specific percentages and nanosecond figures in
  the paper are not re-derived here.

> **Paper-reported (not regenerated here).** WGAN-GP: up to ~82 % structurally
> valid, 100 % novel vs the training set. RWGAN: mean connection count up ~36 %
> (validity falls to ~54 % as the degree ≤ 4 constraint bites) and mean
> generated-topology packet latency improved ~37 % (45.4 ns → 43.3 ns in the
> paper's simulator).

## Architecture & provenance

The published dimensioning (RAPIDO 2021 §4.3.2; PhD thesis table
`tab:NN:RWGAN`), which this repo follows:

| Block | Layers |
|---|---|
| Generator (`gannoc.model.build_generator`) | Input 100 → Dense 162 → Dense 162 → Dense 81, `tanh`, reshaped 9×9 |
| Critic (`build_critic`) **and** reward (`reward_network.build_reward_net`): the paper gives the reward the critic's architecture | Input 9×9 → Conv 64 (9×9, stride 1) → Conv 128 (3×3, stride 2) → Dense 512 → Dense 1, LeakyReLU α = 0.2 |
| Training | RMSprop 5e-5, `n_critic` 5, gradient-penalty weight 10, batch 64, 250 epochs |
| Reward blending | `LAMBDA` 1.0 for epochs 0–100 (purely adversarial), then annealed by −0.005 every 5 epochs down to 0.9, the paper's 10 % reward / 90 % critic split of Phase 3 |

The reference implementation is `GANs/RewardGAN/RWGAN/RWGANgp_pretrainedR.py` in
the original PhD workspace (`build_generator_2` + `build_critic_3` + a frozen
pretrained reward behind `binary_tanh`). Where it and the published tables
disagree (the reference critic's dense head is 1024 units, not 512), this repo
follows the tables. The earlier `RWGANgp_base/RWGANgp_9x9mat.py` (5-layer
generator, 3-conv critic, jointly trained reward, 100 epochs) is a
pre-publication exploratory variant and is **not** reproduced here.

Every hyper-parameter is a named module constant in
[`src/gannoc/training.py`](src/gannoc/training.py), see its "Parameter
provenance" docstring, which maps each value to its source. One caveat is
recorded there: the paper's β (Eq. 3) is stated as 3, but the reference
implementation ran β = 1, which is what `REWARD_BETA` defaults to.

## Repository layout

```
src/gannoc/       importable package: data representation + dataset generator, generator/critic/reward models, losses, reward and RWGAN training loops, evaluation
scripts/          runnable entry points, one settings block each: build the models, generate a dataset, train a reward net, train the WGAN-GP baseline, train the RWGAN
figures/          self-contained: figure scripts + bundled generated-topology sets to reproduce the paper comparisons (no training)
demos/            two standalone Jupyter notebooks: WGAN-GP then RWGAN on the 9-router problem (uses the gannoc package + conda env)
data/reference/   small, git-tracked: pretrained frozen reward-network checkpoints + the paper's generated-topology comparison sets
data/raw/         locally generated NoC datasets (gitignored; regenerate with scripts/generate_dataset.py)
results/          default output location for the training scripts (contents gitignored)
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate gannoc
```

The stack is pinned to **TensorFlow 2.2.0 / Keras 2.3.1** (Python 3.8): the
bundled `.h5` reward checkpoints in `data/reference/reward_networks/` are HDF5
models saved with that Keras and load without conversion here. `protobuf` must
stay below 3.20 or TF 2.2 will not import at all. See the header comments in
`environment.yml` for the per-pin rationale. The package code targets `tf.keras`,
so it also runs unchanged on the newer GPU stack below.

### GPU training

The TF 2.2 / CUDA 10.1 stack has **no kernels for Ampere-or-newer GPUs**
(compute capability ≥ 8.0, e.g. an RTX 30-series card). For GPU training use the
separate environment, which retargets to TensorFlow 2.10 + CUDA 11.2 / cuDNN 8.1
(its `tf.keras` still reads the legacy `.h5` reward checkpoints):

```bash
conda env create -f environment-gpu.yml
conda activate gannoc-gpu
# let TF find the conda-provided CUDA libraries:
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' \
  > "$CONDA_PREFIX/etc/conda/activate.d/cuda_ld.sh"
conda deactivate && conda activate gannoc-gpu
```

The scripts in `scripts/` print which device TensorFlow selected on startup and
enable GPU memory growth; set `FORCE_CPU = True` in a script's settings block to
force CPU even when a GPU is present.

## Build the architecture

```bash
python scripts/build_gannoc.py
```

Builds the generator, the critic, a reward network, and the composite RWGAN
training model, then prints each `model.summary()` and a combined
parameter-count table. Nothing is trained.

Settings at the top of the script: `REWARD_CHECKPOINT` (`None` ⇒ a fresh
untrained net; a path ⇒ load that frozen checkpoint), `PLOT_DIR` (writes Keras
`plot_model` PNGs, needs `pydot` + Graphviz `dot`), `SAVE_DIR` (writes the
untrained models as `.h5`), `FORCE_CPU`.

## Generate a dataset

```bash
python scripts/generate_dataset.py
```

Builds valid **9-router** topologies by random construction (paper
"Algorithm 1"), stratified by physical connection count over
`CONNECTION_RANGE` (8..18 spans a bare spanning tree up to every router
saturated at degree 4), deduplicated across the whole dataset. The paper uses
~10k unique topologies per count (~110k total). Pure Python, **no simulator**.
The output `.npz` is reloaded and structurally re-checked before the script
exits.

Settings: `CONNECTION_RANGE`, `SAMPLES_PER_CLASS` (default 10000), `SEED`,
`OUTPUT` (default `data/raw/nocs.npz`).

`.npz` schema:

| key | dtype / shape | meaning |
|---|---|---|
| `matrices` | int8 `(N, 9, 9)` | binary symmetric adjacency matrices, zero diagonal |
| `n_connections` | int32 `(N,)` | physical connection count (undirected edges) per sample |
| `n_routers` | int scalar | routers per topology (9) |

Latency labels are never written: they would require an external NoC simulator.

## Train

Each script carries a `# settings` block at the top: dataset path, output
directory, epochs, device. The model architecture and the training
hyper-parameters are module constants in `src/gannoc/{training,reward_training}.py`
(see [Architecture & provenance](#architecture--provenance)).

### Reward network

```bash
python scripts/train_reward.py
```

Trains a CNN (`gannoc.reward_network.build_reward_net`, the critic's
architecture) to regress the min-max **normalized** connection count into
`[-1, 1]`, then saves it to `results/checkpoints/reward/reward.h5`. Inside the
RWGAN this network is loaded with `trainable = False`, frozen.

### WGAN-GP baseline

```bash
python scripts/train_wgangp.py
```

No reward network: `LAMBDA` stays pinned at 1.0, so the generator is trained
purely adversarially. This is the reference point the RWGAN is compared against.

### RWGAN

```bash
python scripts/train_rwgan.py
```

The same WGAN-GP core plus a frozen reward net, with `LAMBDA` annealed down so
the reward gains influence. `REWARD_CHECKPOINT` defaults to the shipped
`data/reference/reward_networks/r_cnn_214.h5`, so this runs without
`train_reward.py` first; point it at `results/checkpoints/reward/reward.h5` to
use your own. `LAMBDA_FLOOR` (default `0.9`) is the floor the anneal decays
toward: lower ⇒ stronger reward influence.

Checkpoints and per-epoch history land under `results/{checkpoints,logs}/<run>/`.

### The bundled reward checkpoints

Two pretrained reward nets ship in `data/reference/reward_networks/`, each
rebuildable from `gannoc.reward_network`:

| Checkpoint | Builder | Original source |
|---|---|---|
| `r_cnn_214.h5` (~2.19M params) | `build_reward_net_cnn2` | `build_reward_CNN_2` in `Reward/workspace/reward_NbC_test-all.py` |
| `r_model14.h5` (~148k params) | `build_reward_net_small` | `build_reward` in `Reward/Reward_CNN_NbC/reward.py` |

Neither matches `build_reward_net` (the critic-shaped default the paper
describes). Both predate that choice. `r_model14.h5` was trained on the
topology's connection count; `r_cnn_214.h5` was trained against a separate
`sorted/score_network/` target that is not bundled and cannot be re-verified
here. It stays the default because it is the checkpoint the reference RWGAN
loaded; for a reward whose connection-count semantics is guaranteed, retrain
with `scripts/train_reward.py`.

## Evaluate & figures

`gannoc.evaluate.evaluate_generator` samples a trained generator, binarizes and
symmetrizes each matrix, and reports structural validity, novelty against the
training set, and the connection-count distribution (`metrics.summary()` prints
the lot).

`figures/` reproduces the paper's comparison plots, in particular
`connection_histogram.py`, which draws the WGAN-GP vs RWGAN connection-count
comparison straight from the bundled topology sets in
`data/reference/generated_topologies/` with **no training required**:

```bash
cd figures && python connection_histogram.py
```

See `figures/README.md` for the full figure → script list.

## How this maps to the paper

| Paper concept | Module |
|---|---|
| adjacency-matrix / image encoding of a topology | `gannoc.data` |
| WGAN-GP generator & critic | `gannoc.model` |
| gradient penalty + `LAMBDA` anneal | `gannoc.losses` |
| connection-count reward CNN | `gannoc.reward_network` |
| synthetic dataset ("Algorithm 1") | `gannoc.data.generate_dataset` |
| RWGAN training loop | `gannoc.training` |
| validity / novelty / distribution metrics | `gannoc.evaluate` |

## Citation

This repository reproduces:

- **GANNoC: A Framework for Automatic Generation of NoC Topologies using
  Generative Adversarial Networks.** Mirka, France-Pillois, Sassatelli,
  Gamatié. RAPIDO 2021 (Workshop on Rapid Simulation and Performance
  Evaluation).
- **PhD thesis, Chapter 5** ("GANNoC"), M. Mirka.

## Licence

Source code (`src/`, `scripts/`, `figures/`, `demos/`) is licensed under the
**Apache License 2.0**; see [`LICENSE`](LICENSE). Apache-2.0 rather than MIT for
the explicit patent grant, which matters for architecture work.

Prose and documentation (`README.md`, `figures/README.md`, the notebooks'
narrative cells) are **CC BY 4.0**.

Bundled data and trained checkpoints (`data/reference/generated_topologies/`,
`data/reference/reward_networks/`) are **CC BY 4.0**. They are outputs of the
models in this repository, not third-party data.

No third-party material is vendored here. See [`NOTICE`](NOTICE).

**Data format and trust.** The `.pkl` and `.h5` artefacts execute code when
deserialised; that is a property of the pickle and Keras-HDF5 formats, not of
these files. Load only copies obtained from this repository.
