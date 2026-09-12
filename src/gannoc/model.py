"""GANNoC generator, critic, and the straight-through binarization layer.

The model-construction code lives here; the WGAN-GP training loop itself lives
in :mod:`gannoc.training`.

A NoC topology over ``n_routers`` routers is a binary, symmetric adjacency
matrix treated as a single-channel ``n_routers x n_routers`` image. The
generator is an MLP mapping a noise vector to such an image; the critic is a
CNN scoring it (WGAN-GP: a single linear scalar, no sigmoid).

Both architectures follow the published dimensioning of RAPIDO 2021 section
4.3.2 and PhD Thesis table ``tab:NN:RWGAN``; the reference implementation is
``GANs/RewardGAN/RWGAN/RWGANgp_pretrainedR.py`` in the original workspace
(``build_generator_2`` / ``build_critic_3``).

:func:`binary_tanh` and its helpers are a clean reimplementation of the
straight-through sign activation from BinaryNet (Courbariaux et al. 2016,
arXiv:1602.02830). GANNoC applies it to a generated matrix to snap the entries
to ``{-1, +1}`` before the frozen reward network scores it, while still
passing a usable (hard-tanh) gradient back to the generator.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential

LEAKY_RELU_ALPHA = 0.2
"""Negative slope shared by every LeakyReLU in the GANNoC generator and critic."""


# --------------------------------------------------------------------------- #
# Straight-through binarization                                                #
# (BinaryNet, Courbariaux et al. 2016, arXiv:1602.02830).                      #
# --------------------------------------------------------------------------- #
def round_through(x: tf.Tensor) -> tf.Tensor:
    """Round ``x`` with a straight-through gradient: forward rounds, backward is identity.

    Implemented as ``x + stop_gradient(round(x) - x)`` so the rounded value is
    what flows forward while autodiff still sees ``d/dx x == 1`` -- the rounding
    op, which has a zero gradient almost everywhere, is hidden behind
    ``stop_gradient``.
    """
    return x + K.stop_gradient(K.round(x) - x)


def hard_sigmoid(x: tf.Tensor) -> tf.Tensor:
    """Piecewise-linear sigmoid approximation ``clip(0.5 * x + 0.5, 0, 1)``."""
    return K.clip(0.5 * x + 0.5, 0.0, 1.0)


def binary_tanh(x: tf.Tensor) -> tf.Tensor:
    """Straight-through sign activation.

    Forward pass returns sign-like values in ``{-1, +1}``; backward pass uses
    the hard-tanh gradient (non-zero only where :func:`hard_sigmoid` is not
    saturated, i.e. ``-1 < x < 1``). Equal to
    ``2 * round_through(hard_sigmoid(x)) - 1``.
    """
    return 2.0 * round_through(hard_sigmoid(x)) - 1.0


class BinaryTanh(tf.keras.layers.Layer):
    """Keras layer wrapping :func:`binary_tanh`.

    Used to binarize a generated adjacency matrix to ``{-1, +1}`` before the
    frozen reward network scores it, keeping a straight-through gradient path
    open back to the generator.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return binary_tanh(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_generator(latent_dim: int = 100, n_routers: int = 9, name: str = "generator") -> Model:
    """MLP generator: a ``latent_dim`` noise vector -> ``n_routers x n_routers x 1`` image.

    Two hidden ``Dense`` blocks of ``2 * n_routers ** 2`` units (162 for the
    paper's 9 routers), each followed by BatchNorm and LeakyReLU, then a
    ``tanh`` projection to ``n_routers ** 2`` values reshaped into the
    single-channel adjacency-matrix image. ``tanh`` keeps the generated
    entries in ``[-1, 1]``, matching the ``{-1, +1}`` encoding of the real
    matrices.

    Layer sizes ``{162, 162, 81}`` are the published dimensioning (RAPIDO 2021
    section 4.3.2; PhD Thesis table ``tab:NN:RWGAN``), implemented in the
    reference code as ``RWGANgp_pretrainedR.build_generator_2``. The deeper
    ``{512, 2048, 324, 162, 81}`` stack in the pre-publication
    ``RWGANgp_base/RWGANgp_9x9mat.py`` is an earlier exploratory variant and is
    deliberately not reproduced here.
    """
    hidden_units = 2 * n_routers * n_routers
    model = Sequential(name=name)

    model.add(Dense(hidden_units, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Dense(hidden_units))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Dense(n_routers * n_routers, activation="tanh"))
    model.add(Reshape((n_routers, n_routers, 1)))
    return model


def build_critic(n_routers: int = 9, name: str = "critic") -> Model:
    """CNN critic for WGAN-GP: ``n_routers x n_routers x 1`` image -> linear scalar.

    No sigmoid -- the critic estimates the Wasserstein distance, so its output
    is an unbounded score. The first convolution uses a full
    ``n_routers x n_routers`` kernel so every unit sees the whole matrix at
    once; a strided 3x3 convolution then shrinks the spatial size before the
    dense head.

    Dimensioning is the published one (RAPIDO 2021 section 4.3.2; PhD Thesis
    table ``tab:NN:RWGAN``): Conv 64 (9x9, stride 1) -> Conv 128 (3x3, stride
    2) -> Dense 512 -> Dense 1, LeakyReLU alpha 0.2 throughout. The reference
    code (``RWGANgp_pretrainedR.build_critic_3``) used a wider 1024-unit dense
    layer; where the two disagree this repo follows the published table.

    The reward network reuses this architecture -- see
    :func:`gannoc.reward_network.build_reward_net`.
    """
    model = Sequential(name=name)

    model.add(
        Conv2D(
            64,
            (n_routers, n_routers),
            strides=1,
            padding="same",
            input_shape=(n_routers, n_routers, 1),
        )
    )
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Conv2D(128, (3, 3), strides=2, padding="valid"))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Dense(1))
    return model
