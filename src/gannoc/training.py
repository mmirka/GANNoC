"""GANNoC training loop: a WGAN-GP over 9x9 NoC adjacency images, optionally
reward-steered (the "RWGAN" variant).

The critic sees a plain single-channel ``(N, R, R, 1)`` image. Training follows
the standard WGAN-GP recipe (``N_CRITIC`` critic updates per generator update, a
gradient-penalty term enforcing the 1-Lipschitz constraint). In the RWGAN a
frozen reward network's MSE against a target connection count is blended into
the generator loss under a :class:`~gannoc.losses.LambdaSchedule`: the balance
starts fully adversarial and is annealed toward the reward objective.

Two entry points:

* :func:`train_wgangp` -- the plain baseline. LAMBDA is pinned at 1.0, so the
  reward term's ``(1 - LAMBDA)`` weight stays 0 and no reward net is needed.
* :func:`train_rwgan` -- the reward-guided variant, driven by a pretrained
  frozen reward checkpoint.

Parameter provenance
--------------------
Every constant below is the published value (RAPIDO 2021 section 4.3.2 and
PhD Thesis Chapter "GANNoC", table ``tab:NN:RWGAN``) as implemented by the
reference code ``GANs/RewardGAN/RWGAN/RWGANgp_pretrainedR.py``:

* ``LATENT_DIM`` 100, ``BATCH_SIZE`` 64, ``N_CRITIC`` 5,
  ``GRADIENT_PENALTY_WEIGHT`` 10, RMSprop at ``LEARNING_RATE`` 5e-5.
* ``EPOCHS`` 250 and the LAMBDA schedule reproduce the paper's phases: Phase 2
  (epochs 0-100) trains without the reward (LAMBDA = 1.0); Phase 3 (epochs
  101-250) progressively tailors the split to 10% reward / 90% critic, i.e.
  LAMBDA descends by ``LAMBDA_DELTA`` = -0.005 every ``LAMBDA_EVERY_N_EPOCHS``
  = 5 epochs down to ``LAMBDA_FLOOR`` = 0.9.
* ``REWARD_TARGET`` 0.5 is the reference implementation's aimed score.
* ``REWARD_BETA`` is the beta of paper Eq. 3,
  ``L_G = (1 - lambda) L_C + lambda [beta L_R]``. The paper text sets beta = 3
  (stating it remains effective within 1 to 5); the reference implementation ran
  ``loss_weights=[1, 1]``. We keep 1.0 so training dynamics match the code that
  produced the published figures -- change the constant to 3.0 to follow the
  paper text instead.

Sign convention: this module labels real samples ``y_true = -1`` and fakes
``y_true = +1``, the sign-flip of the reference implementation's ``+1`` / ``-1``.
The two are equivalent -- it only flips the sign of the critic's output -- and
the choice here makes "higher critic score" mean "more real". See the comment at
the ground-truth vectors in :func:`_train`.
"""
from __future__ import annotations

import functools
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# gannoc.losses.gradient_penalty_loss builds the WGAN-GP penalty with
# K.gradients / tf.gradients, which only connects a gradient in graph mode.
# Under eager execution (the TF >= 2.4 default) tf.gradients returns None at
# functional-model *construction* time ("None values not supported"), so the
# gradient-penalty branch of ``critic_model`` below would fail to build.
# Disable eager for this process; training here is pure compiled
# ``model.train_on_batch`` and never needs eager execution.
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

from .data import load_dataset  # noqa: E402
from .evaluate import evaluate_generator  # noqa: E402
from .gpu import configure_gpu  # noqa: E402
from .losses import (  # noqa: E402
    LambdaSchedule,
    RandomWeightedAverage,
    gradient_penalty_loss,
    make_generator_wasserstein_loss,
    make_reward_mse_loss,
    wasserstein_loss,
)
from .model import BinaryTanh, build_critic, build_generator  # noqa: E402
from .reward_network import build_reward_net, load_reward_network  # noqa: E402

# --------------------------------------------------------------------------- #
# Hyper-parameters. See "Parameter provenance" above -- these are the paper's   #
# values, not per-run knobs. Edit here to experiment.                           #
# --------------------------------------------------------------------------- #
N_ROUTERS = 9                       # routers per topology => 9x9 adjacency image
LATENT_DIM = 100                    # generator input noise dimension
BATCH_SIZE = 64
N_CRITIC = 5                        # critic updates per generator update
GRADIENT_PENALTY_WEIGHT = 10.0
LEARNING_RATE = 5e-5                # RMSprop, for both critic and generator
EPOCHS = 250

LAMBDA_INITIAL = 1.0                # 1.0 => purely adversarial (Phase 2)
LAMBDA_DELTA = -0.005               # per anneal step
LAMBDA_START_EPOCH = 100            # Phase 3 begins here
LAMBDA_EVERY_N_EPOCHS = 5
LAMBDA_FLOOR = 0.9                  # 0.9 critic / 0.1 reward at the end
REWARD_TARGET = 0.5                 # connections_norm target in [-1, 1]
REWARD_BETA = 1.0                   # beta of paper Eq. 3

EVAL_SAMPLES = 1000                 # generated topologies scored per evaluation
EVAL_EVERY = 25                     # epochs between evaluations
LOG_EVERY = 10                      # generator updates between history samples
FIXED_NOISE_SAMPLES = 100           # latent batch reused for the logged means
SEED = None                         # set an int for a reproducible run


def build_models(reward=None, lambda_floor: float = LAMBDA_FLOOR) -> dict:
    """Build and compile the two training models.

    ``reward`` is a frozen reward network; pass ``None`` only for the plain
    WGAN-GP baseline (``lambda_floor == 1.0``), where a fresh untrained net is
    built purely so the generator graph has a well-formed second output.

    Returns ``{"generator", "critic", "reward", "critic_model",
    "generator_model", "lambda_schedule"}``. See the module docstring for the
    training recipe.
    """
    generator = build_generator(LATENT_DIM, N_ROUTERS)
    critic = build_critic(N_ROUTERS)

    if reward is None:
        if lambda_floor < 1.0:
            raise ValueError(
                "A pretrained reward network is required when lambda_floor < 1.0 "
                "(the reward term becomes active as LAMBDA anneals down). Use "
                "train_rwgan() with a reward checkpoint, or train_wgangp() for "
                "the pure WGAN-GP baseline."
            )
        # Pure WGAN-GP baseline (lambda_floor == 1.0): LAMBDA is pinned at its
        # initial 1.0 for the whole run, so make_reward_mse_loss()'s (1 - LAMBDA)
        # factor stays 0 and this net never influences training.
        reward = build_reward_net(N_ROUTERS)
    reward.trainable = False

    lambda_schedule = LambdaSchedule(
        initial=LAMBDA_INITIAL,
        delta=LAMBDA_DELTA,
        start_epoch=LAMBDA_START_EPOCH,
        every_n_epochs=LAMBDA_EVERY_N_EPOCHS,
        floor=lambda_floor,
    )

    image_shape = (N_ROUTERS, N_ROUTERS, 1)

    # ---- critic training model (generator + reward frozen) ----
    critic.trainable = True
    generator.trainable = False

    real_img = Input(shape=image_shape, name="real_image")
    z_disc = Input(shape=(LATENT_DIM,), name="critic_noise")

    fake_img = generator(z_disc)
    valid = critic(real_img)
    fake = critic(fake_img)

    interp = RandomWeightedAverage(BATCH_SIZE)([real_img, fake_img])
    valid_interp = critic(interp)

    # gradient_penalty_loss differentiates the critic w.r.t. the interpolated
    # batch, so it needs that batch bound in; give the partial a __name__ so
    # Keras can name the loss in its logs.
    partial_gp = functools.partial(
        gradient_penalty_loss,
        averaged_samples=interp,
        gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
    )
    partial_gp.__name__ = "gradient_penalty"

    critic_model = Model([real_img, z_disc], [valid, fake, valid_interp], name="critic_model")
    critic_model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        loss=[wasserstein_loss, wasserstein_loss, partial_gp],
    )

    # ---- generator training model (critic + reward frozen) ----
    critic.trainable = False
    generator.trainable = True

    z_gen = Input(shape=(LATENT_DIM,), name="generator_noise")
    gen_img = generator(z_gen)
    gen_valid = critic(gen_img)
    # BinaryTanh straight-through-binarizes the generator's tanh output to
    # {-1, +1} before the reward net scores it, so the reward sees what a
    # realised (hard 0/1) topology would look like rather than soft logits.
    gen_reward = reward(BinaryTanh()(gen_img))

    generator_model = Model(z_gen, [gen_valid, gen_reward], name="generator_model")
    generator_model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        # loss_weights[1] is beta of paper Eq. 3; the LAMBDA / (1 - LAMBDA)
        # split itself lives inside the two loss closures.
        loss=[
            make_generator_wasserstein_loss(lambda_schedule),
            make_reward_mse_loss(lambda_schedule),
        ],
        loss_weights=[1, REWARD_BETA],
    )

    return {
        "generator": generator,
        "critic": critic,
        "reward": reward,
        "critic_model": critic_model,
        "generator_model": generator_model,
        "lambda_schedule": lambda_schedule,
    }


def train_wgangp(
    dataset_path: "str | Path",
    output_dir: "str | Path",
    run_name: str,
    epochs: int = EPOCHS,
    force_cpu: bool = False,
) -> dict:
    """Train the plain WGAN-GP baseline: no reward network, LAMBDA pinned at 1.0."""
    return _train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        run_name=run_name,
        epochs=epochs,
        reward_checkpoint=None,
        lambda_floor=1.0,
        force_cpu=force_cpu,
        variant="WGAN-GP",
    )


def train_rwgan(
    dataset_path: "str | Path",
    reward_checkpoint: "str | Path",
    output_dir: "str | Path",
    run_name: str,
    epochs: int = EPOCHS,
    lambda_floor: float = LAMBDA_FLOOR,
    force_cpu: bool = False,
) -> dict:
    """Train the reward-guided RWGAN against a pretrained frozen reward checkpoint."""
    if not reward_checkpoint:
        raise ValueError(
            "train_rwgan() needs a pretrained reward checkpoint. Train one with "
            "scripts/train_reward.py, or use train_wgangp() for the baseline."
        )
    return _train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        run_name=run_name,
        epochs=epochs,
        reward_checkpoint=reward_checkpoint,
        lambda_floor=lambda_floor,
        force_cpu=force_cpu,
        variant="RWGAN",
    )


def _train(
    dataset_path,
    output_dir,
    run_name: str,
    epochs: int,
    reward_checkpoint,
    lambda_floor: float,
    force_cpu: bool,
    variant: str,
) -> dict:
    """Build the models, run the WGAN-GP loop with the annealed reward term,
    evaluate periodically, and save the artifacts.

    Returns the artifacts dict from :func:`_save_artifacts` (generator/critic
    checkpoints, pickled history, training time, and final generator metrics).
    """
    print(configure_gpu(force_cpu))

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
    real_images = data.scaled_images().astype(np.float32)  # (N, R, R, 1) in [-1, 1]

    reward = None
    if reward_checkpoint:
        reward = load_reward_network(reward_checkpoint, trainable=False)

    models = build_models(reward, lambda_floor=lambda_floor)
    generator = models["generator"]
    critic = models["critic"]
    critic_model = models["critic_model"]
    generator_model = models["generator_model"]
    lambda_schedule = models["lambda_schedule"]

    # WGAN sign convention. wasserstein_loss(y_true, y_pred) = mean(y_true * y_pred)
    # and the critic emits a raw score (higher => "more real"). Labelling real
    # samples y_true = -1 and fakes y_true = +1 makes the critic's Wasserstein
    # loss  mean(-score_real) + mean(score_fake), minimised by driving real
    # scores up and fake scores down. The generator trains its own fakes with
    # y_true = -1, so it instead pushes the critic's score on them up. The
    # gradient-penalty output takes an all-zero dummy target -- the penalty is
    # computed entirely inside gradient_penalty_loss from the interpolates.
    valid = -np.ones((BATCH_SIZE, 1), dtype=np.float32)
    fake = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    dummy = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    reward_aim = np.full((BATCH_SIZE, 1), REWARD_TARGET, dtype=np.float32)

    # A fixed latent batch so the logged mean critic / reward outputs track the
    # same points in latent space across the whole run.
    fixed_noise = np.random.uniform(
        0.0, 1.0, size=(FIXED_NOISE_SAMPLES, LATENT_DIM)
    ).astype(np.float32)

    history_keys = (
        "step", "epoch", "lambda", "critic_loss", "critic_real", "critic_fake",
        "critic_gp", "generator_wasserstein", "reward_mse",
        "mean_reward_output", "mean_critic_output",
    )
    history: dict = {key: [] for key in history_keys}

    step = 0
    start_train = time.time()
    for epoch in range(epochs):
        np.random.shuffle(real_images)
        n_steps = len(real_images) // (BATCH_SIZE * N_CRITIC)

        c_loss = g_loss = None
        for _ in range(n_steps):
            for _ in range(N_CRITIC):
                idx = np.random.randint(0, len(real_images), size=BATCH_SIZE)
                real_batch = real_images[idx]
                noise = np.random.uniform(
                    0.0, 1.0, size=(BATCH_SIZE, LATENT_DIM)
                ).astype(np.float32)
                # critic_model.train_on_batch -> [total, w_real, w_fake, gp]
                c_loss = critic_model.train_on_batch(
                    [real_batch, noise], [valid, fake, dummy]
                )

            noise = np.random.uniform(
                0.0, 1.0, size=(BATCH_SIZE, LATENT_DIM)
            ).astype(np.float32)
            # generator_model.train_on_batch -> [total, g_wasserstein, reward_mse]
            g_loss = generator_model.train_on_batch(noise, [valid, reward_aim])
            step += 1

            # Sample the history every ``LOG_EVERY`` generator updates (and on the
            # first step). mean critic / reward outputs come from the frozen
            # fixed_noise batch through generator_model.predict, whose outputs are
            # [critic_score, reward_score]; skipping most steps keeps that extra
            # forward pass off the hot path on long runs.
            if step % LOG_EVERY and step != 1:
                continue
            fixed_valid, fixed_reward = generator_model.predict(fixed_noise)
            history["step"].append(step)
            history["epoch"].append(epoch)
            history["lambda"].append(float(lambda_schedule.value))
            history["critic_loss"].append(float(c_loss[0]))
            history["critic_real"].append(float(c_loss[1]))
            history["critic_fake"].append(float(c_loss[2]))
            history["critic_gp"].append(float(c_loss[3]))
            history["generator_wasserstein"].append(float(g_loss[1]))
            history["reward_mse"].append(float(g_loss[2]))
            history["mean_reward_output"].append(float(np.mean(fixed_reward)))
            history["mean_critic_output"].append(float(np.mean(fixed_valid)))

        # Anneal the adversarial / reward balance once per epoch.
        lambda_schedule.step(epoch)

        if c_loss is not None:
            print(
                f"epoch {epoch + 1:4d}/{epochs}  "
                f"lambda={float(lambda_schedule.value):.3f}  "
                f"critic_loss={float(c_loss[0]):.4f}  "
                f"gen_wasserstein={float(g_loss[1]):.4f}  "
                f"reward_mse={float(g_loss[2]):.4f}"
            )

        is_last = epoch == epochs - 1
        if not is_last and (epoch + 1) % EVAL_EVERY == 0:
            metrics = evaluate_generator(
                generator, data.matrices,
                n_samples=EVAL_SAMPLES, latent_dim=LATENT_DIM,
            )
            print(metrics.summary())

    training_time = time.time() - start_train
    print(f"{variant} training complete in {training_time:.1f}s")

    # Always score the final generator.
    metrics = evaluate_generator(
        generator, data.matrices,
        n_samples=EVAL_SAMPLES, latent_dim=LATENT_DIM,
    )
    print(metrics.summary())

    extra = {
        "training_time_seconds": training_time,
        "final_valid_rate": {t: metrics.valid_rate(t) for t in metrics.thresholds},
        "final_novelty_rate": {t: metrics.novelty_rate(t) for t in metrics.thresholds},
        "final_mean_connections": {
            t: metrics.mean_connections(t) for t in metrics.thresholds
        },
    }
    return _save_artifacts(output_dir, run_name, generator, critic, history, extra)


def _save_artifacts(output_dir, run_name: str, generator, critic, history: dict,
                    extra: dict) -> dict:
    """Save generator/critic checkpoints and the pickled history under
    ``<output_dir>/{checkpoints,logs}/<run_name>/``; return the paths plus
    ``extra`` (training time, final metrics)."""
    checkpoint_dir = Path(output_dir) / "checkpoints" / run_name
    log_dir = Path(output_dir) / "logs" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generator_path = checkpoint_dir / "generator.h5"
    critic_path = checkpoint_dir / "critic.h5"
    history_path = log_dir / "history.pkl"

    generator.save(generator_path)
    critic.save(critic_path)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    artifacts = {
        "run_name": run_name,
        "generator_checkpoint": str(generator_path),
        "critic_checkpoint": str(critic_path),
        "history": str(history_path),
    }
    artifacts.update(extra)
    return artifacts
