"""Reward-network architectures: frozen regressors that score generated topologies.

A reward network maps a ``n_routers x n_routers x 1`` adjacency image to a
single linear scalar -- the predicted *normalized* number of physical
connections, in the ``[-1, 1]`` space of
``gannoc.data.NoCDataset.connections_norm``. It is trained on its own (see
:mod:`gannoc.reward_training`) and then used frozen inside the RWGAN, where its
MSE against a target connection count is blended into the generator loss.

The paper states that "the reward network reproduces the same architecture as
the critic network" (RAPIDO 2021 section 4.3.2; PhD Thesis table
``tab:NN:RWGAN``), so :func:`build_reward_net` is exactly
:func:`gannoc.model.build_critic` under another name. The two other builders
here exist only to reconstruct the shipped checkpoints, whose architectures the
original workspace had explored separately:

===============================  ==========================================
Builder                          Matching checkpoint / original source
===============================  ==========================================
:func:`build_reward_net`         -- (the published critic-shaped reward)
:func:`build_reward_net_cnn2`    ``r_cnn_214.h5`` -- ``build_reward_CNN_2``
                                 in ``Reward/workspace/reward_NbC_test-all.py``
:func:`build_reward_net_small`   ``r_model14.h5`` -- ``build_reward`` in
                                 ``Reward/Reward_CNN_NbC/reward.py``
===============================  ==========================================
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU
from tensorflow.keras.models import Model, Sequential

from .model import LEAKY_RELU_ALPHA, build_critic


def build_reward_net(n_routers: int = 9, name: str = "reward") -> Model:
    """CNN reward network: ``n_routers x n_routers x 1`` adjacency image -> linear scalar.

    Predicts the normalized connection count (the ``[-1, 1]`` space of
    ``gannoc.data.NoCDataset.connections_norm``).

    The published architecture is the critic's -- Conv 64 (9x9, stride 1) ->
    Conv 128 (3x3, stride 2) -> Dense 512 -> Dense 1 -- so this delegates to
    :func:`gannoc.model.build_critic` rather than restating it. Architecture per
    RAPIDO 2021 section 4.3.2 / PhD Thesis table ``tab:NN:RWGAN``.
    """
    return build_critic(n_routers=n_routers, name=name)


def build_reward_net_cnn2(n_routers: int = 9, name: str = "reward_cnn2") -> Model:
    """Reward CNN matching the bundled ``r_cnn_214.h5`` checkpoint.

    A full-size ``n_routers x n_routers`` first convolution so every unit sees
    the whole matrix, two further convolutions (the last strided), then a
    1024-unit dense head -- ``build_reward_CNN_2`` from the original
    ``Reward/workspace/reward_NbC_test-all.py``.

    Reproduces ``data/reference/reward_networks/r_cnn_214.h5`` (input
    ``(None, 9, 9, 1)``, output ``(None, 1)``, ~2.19M parameters). This is *not*
    the architecture the paper describes for the reward -- see
    :func:`build_reward_net` -- it is kept so the shipped checkpoint can be
    rebuilt and inspected.
    """
    model = Sequential(name=name)

    model.add(
        Conv2D(
            32,
            (n_routers, n_routers),
            strides=1,
            padding="same",
            input_shape=(n_routers, n_routers, 1),
        )
    )
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Conv2D(64, (3, 3), strides=1, padding="valid"))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Conv2D(128, (3, 3), strides=2, padding="same"))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Dense(1))
    return model


def build_reward_net_small(n_routers: int = 9, name: str = "reward_small") -> Model:
    """Smaller row-wise reward CNN matching the bundled ``r_model14.h5`` checkpoint.

    Convolves the adjacency image with ``1 x n_routers`` kernels, so each unit
    reads one router's full row of connections at a time, then a 64-unit dense
    head -- ``build_reward`` from the original ``Reward/Reward_CNN_NbC/reward.py``,
    which trained it against the topology's connection count.

    Reproduces ``data/reference/reward_networks/r_model14.h5`` (~148k
    parameters).
    """
    model = Sequential(name=name)

    model.add(
        Conv2D(
            64,
            (1, n_routers),
            strides=1,
            padding="same",
            input_shape=(n_routers, n_routers, 1),
        )
    )
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Conv2D(128, (1, n_routers), strides=1, padding="valid"))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Dense(1))
    return model


def load_reward_network(
    path: "str | Path", trainable: bool = False, name: str = "reward"
) -> Model:
    """Load a pretrained reward network from an HDF5 checkpoint.

    Loads with ``tf.keras.models.load_model(path, compile=False)``, sets
    ``.trainable = trainable`` (default ``False`` -- frozen, as used inside the
    RWGAN), renames the model to ``name`` and returns it.

    The shipped ``.h5`` checkpoints were saved by an older Keras;
    ``compile=False`` skips restoring the (absent) training configuration and
    so avoids the missing-training-configuration warning path. The builders
    above are never consulted when loading -- a checkpoint carries its own
    architecture.
    """
    model = tf.keras.models.load_model(str(path), compile=False)
    model.trainable = trainable
    model._name = name
    return model
