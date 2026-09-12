#!/usr/bin/env python3
"""Instantiate the full GANNoC architecture and report it.

Builds the three networks of the framework -- the WGAN-GP generator, the
WGAN-GP critic, and the reward CNN that scores a topology's connection count --
then assembles the composite critic/generator training models exactly as
``gannoc.training`` would. Prints every model's ``.summary()`` and a per-model
parameter table; it never trains anything.

Dimensioning follows RAPIDO 2021 section 4.3.2 / PhD Thesis table
``tab:NN:RWGAN``: generator {162, 162, 81}; critic and reward alike are
Conv 64 (9x9, s1) -> Conv 128 (3x3, s2) -> Dense 512 -> Dense 1.

Edit the settings below, then run:

    python scripts/build_gannoc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gannoc import training  # noqa: E402
from gannoc.gpu import configure_gpu  # noqa: E402
from gannoc.model import build_critic, build_generator  # noqa: E402
from gannoc.reward_network import build_reward_net, load_reward_network  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings                                                                      #
# --------------------------------------------------------------------------- #
# None -> build a fresh, untrained reward net. Set a path (e.g.
# "data/reference/reward_networks/r_cnn_214.h5") to load a frozen pretrained one
# and configure the reward-guided RWGAN composite instead.
REWARD_CHECKPOINT = None
PLOT_DIR = None     # e.g. "build/plots" -- needs pydot + Graphviz `dot`
SAVE_DIR = None     # e.g. "build/untrained" -- writes the untrained models as .h5
FORCE_CPU = False


def _param_counts(model) -> "tuple[int, int, int]":
    """Return ``(trainable, non_trainable, total)`` weight counts for ``model``.

    Uses the same per-weight counting Keras' own ``.summary()`` does, so a
    frozen reward network correctly shows up as all-non-trainable.
    """
    from tensorflow.keras import backend as keras_backend

    trainable = int(sum(keras_backend.count_params(w) for w in model.trainable_weights))
    non_trainable = int(
        sum(keras_backend.count_params(w) for w in model.non_trainable_weights)
    )
    return trainable, non_trainable, trainable + non_trainable


def _print_summary(title: str, model) -> None:
    """Print ``model.summary()`` under a titled banner, routed through ``print``."""
    print()
    print(f"=== {title} ===")
    model.summary(print_fn=print)


def _render_plots(models_by_name: dict, plot_dir: Path) -> None:
    """Best-effort ``plot_model`` render of each model; never fatal."""
    try:
        from tensorflow.keras.utils import plot_model

        plot_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models_by_name.items():
            out = plot_dir / f"{name}.png"
            plot_model(model, to_file=str(out), show_shapes=True, show_layer_names=True)
            print(f"wrote {out}")
    except ImportError:
        print(
            "note: PLOT_DIR needs `pydot` and Graphviz `dot` on PATH; "
            "skipping model plots."
        )
    except Exception as exc:  # e.g. `dot` executable missing from PATH
        print(
            f"note: could not render model plots ({exc}); "
            "check that Graphviz `dot` is installed and on PATH."
        )


def _save_models(models_by_name: dict, save_dir: Path) -> None:
    """Write each untrained model to ``save_dir/<name>.h5``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models_by_name.items():
        out = save_dir / f"{name}.h5"
        model.save(str(out))
        print(f"wrote {out}")


def main() -> int:
    """Build every GANNoC model, print summaries + a parameter table, return 0."""
    print(configure_gpu(force_cpu=FORCE_CPU))

    from_checkpoint = REWARD_CHECKPOINT is not None

    generator = build_generator(latent_dim=training.LATENT_DIM, n_routers=training.N_ROUTERS)
    critic = build_critic(n_routers=training.N_ROUTERS)
    if from_checkpoint:
        reward = load_reward_network(REWARD_CHECKPOINT, trainable=False, name="reward")
        reward_kind = f"pretrained checkpoint ({REWARD_CHECKPOINT})"
    else:
        reward = build_reward_net(n_routers=training.N_ROUTERS)
        reward_kind = "fresh untrained reward net (critic-shaped)"

    print()
    print("=" * 78)
    print("GANNoC architecture")
    print(
        f"  latent_dim={training.LATENT_DIM}  n_routers={training.N_ROUTERS}  "
        f"reward: {reward_kind}"
    )
    print("=" * 78)

    _print_summary("generator", generator)
    _print_summary("critic", critic)
    _print_summary("reward", reward)

    # The per-model summaries above are the guaranteed part; the composite
    # WGAN-GP training models depend on gannoc.training internals, so guard
    # them and still return 0 if assembly fails.
    try:
        models = training.build_models(
            reward=reward if from_checkpoint else None,
            lambda_floor=training.LAMBDA_FLOOR if from_checkpoint else 1.0,
        )
        _print_summary("critic training model", models["critic_model"])
        _print_summary("generator training model", models["generator_model"])
    except Exception as exc:
        print()
        print(
            f"note: could not assemble the composite training models ({exc}). "
            "The per-model summaries above are unaffected."
        )

    print()
    print("Parameter counts")
    print(f"  {'model':<12}{'trainable':>15}{'non-trainable':>16}{'total':>15}")
    grand_total = 0
    for name, model in (("generator", generator), ("critic", critic), ("reward", reward)):
        trainable, non_trainable, total = _param_counts(model)
        grand_total += total
        print(f"  {name:<12}{trainable:>15,}{non_trainable:>16,}{total:>15,}")
    print(f"  {'grand total':<12}{'':>15}{'':>16}{grand_total:>15,}")

    core_models = {"generator": generator, "critic": critic, "reward": reward}
    if PLOT_DIR:
        _render_plots(core_models, Path(PLOT_DIR))
    if SAVE_DIR:
        _save_models(core_models, Path(SAVE_DIR))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
