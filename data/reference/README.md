# Reference data

Small, curated data committed directly with the repo (~10 MB) so the frozen
reward-network checkpoints and the paper's generated-topology comparison sets
need no separate download. Everything larger (synthetic training corpora,
trained generator weights) is regenerated locally (see the repo `README.md`).

## `reward_networks/`

Two pretrained reward CNNs. A reward network maps a `(None, 9, 9, 1)` adjacency
image to a single linear scalar, the min-max **normalized** number of physical
connections, in the `[-1, 1]` space of
`gannoc.data.NoCDataset.connections_norm`. Inside the RWGAN the network is
loaded with `trainable = False` (frozen) and its MSE against `--reward-target`
is blended into the generator loss.

| File | Architecture | Params | Role |
|---|---|---|---|
| `r_cnn_214.h5` | full-matrix first conv → two convs (last strided) → 1024-unit dense head; matches `gannoc.reward_network.build_reward_net` | ~2.19 M | the paper's production reward CNN; loaded by `scripts/train_gannoc.py` by default |
| `r_model14.h5` | row-wise `1×9` convolutions → 64-unit dense head; matches `gannoc.reward_network.build_reward_net_small` | ~148 k | smaller variant kept for reference / ablation |

Both were saved with an older Keras. Load them with
`tf.keras.models.load_model(path, compile=False)`, which is what
`gannoc.reward_network.load_reward_network` does; `compile=False` skips the
absent training configuration.

## `generated_topologies/`

Two pickled NumPy arrays, each shape `(446, 9, 9)`, values `{0, 1}`, symmetric
with zero diagonal:

| File | Contents |
|---|---|
| `wgan_valid_topologies.pkl` | the paper's WGAN-GP-generated structurally-valid 9-router topologies |
| `rwgan_valid_topologies.pkl` | the paper's RWGAN-generated structurally-valid topologies |

Both sets were produced from the **same 446 latent-noise seeds**, so they line
up sample-for-sample for a paired comparison. Verified by loading them through
this repo's own `gannoc.data` helpers: all 446 matrices in each set are
symmetric, connected, and within the degree ≤ 4 cap, and the mean physical
connection count is **12.0** for the WGAN-GP set versus **14.3** for the RWGAN
set: the reward term visibly shifts generation toward denser topologies. That
comparison is `figures/connection_histogram.py`. The sets carry graph structure
only; they are unscored (no latency labels).

## Not included

- the multi-GB synthetic training corpora, regenerate with
  `scripts/generate_dataset.py`;
- trained GAN **generator** weights, retrain with `scripts/train_gannoc.py`;
- anything from an external NoC simulator (latency, saturation curves).
