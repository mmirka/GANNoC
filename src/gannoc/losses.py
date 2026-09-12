"""WGAN-GP loss machinery and the RWGAN critic-vs-reward annealing schedule.

Contains the Wasserstein loss, the :class:`RandomWeightedAverage` interpolation
layer and the two-sided :func:`gradient_penalty_loss` for WGAN-GP, and
:class:`LambdaSchedule` -- a shared ``LAMBDA`` backend variable that anneals the
balance between the generator's adversarial (Wasserstein) term and its reward
(MSE) term in the RWGAN.

The gradient penalty is computed with ``K.gradients``, which only connects a
gradient in graph mode -- see :mod:`gannoc.training` for why this stack
disables eager execution.

Loss layout follows RAPIDO 2021 / PhD Thesis Chapter 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow.keras import backend as K


def wasserstein_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Wasserstein loss ``mean(y_true * y_pred)``.

    ``gannoc.training`` labels real samples ``y_true = -1`` and fakes
    ``y_true = +1``, so minimizing this drives the critic's real scores up and
    its fake scores down; the generator trains its fakes against ``y_true = -1``.
    """
    return K.mean(y_true * y_pred)


class RandomWeightedAverage(tf.keras.layers.Layer):
    """Random per-sample interpolation of a real and a fake batch.

    Produces ``alpha * real + (1 - alpha) * fake`` with ``alpha ~ U[0, 1]``
    drawn once per sample -- the point at which the WGAN-GP gradient penalty
    constrains the critic's gradient norm to 1. Inputs are 4-D
    ``(N, R, R, 1)`` adjacency images.
    """

    def __init__(self, batch_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        real, fake = inputs
        # Per-sample alpha, broadcast over the R x R x 1 image axes.
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * real) + ((1.0 - alpha) * fake)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def gradient_penalty_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    averaged_samples: tf.Tensor,
    gradient_penalty_weight: float,
) -> tf.Tensor:
    """Two-sided WGAN-GP penalty on the critic gradient norm at ``averaged_samples``.

    Bind ``averaged_samples`` and ``gradient_penalty_weight`` with
    :func:`functools.partial` to obtain a two-argument Keras loss; ``y_true``
    is unused (Keras still passes dummy targets). The term is
    ``weight * mean((1 - ||grad||_2) ** 2)`` over the batch, pulling the
    gradient norm towards 1 from both sides.

    Computed with ``K.gradients``, which only connects a gradient in graph
    mode -- see :mod:`gannoc.training` for why this stack disables eager
    execution.
    """
    grads = K.gradients(y_pred, averaged_samples)[0]
    norm = K.sqrt(K.sum(K.square(grads), axis=[1, 2, 3]))
    return gradient_penalty_weight * K.mean(K.square(1.0 - norm))


@dataclass
class LambdaSchedule:
    """Anneals the critic-vs-reward mix in the RWGAN generator loss.

    Holds a tf.keras backend variable ``LAMBDA`` (float32). ``LAMBDA``
    multiplies the generator's Wasserstein term and ``(1 - LAMBDA)`` the
    reward MSE term. It starts at ``initial`` (``1.0`` => pure WGAN-GP, reward
    weight 0) and, from epoch ``start_epoch`` onward, every ``every_n_epochs``
    epochs, is nudged by ``delta`` (negative) but never below ``floor``.

    Setting ``floor == initial == 1.0`` keeps the model a pure WGAN-GP
    baseline for the whole run; ``floor = 0.1`` gives the reward network a
    strong influence.
    """

    initial: float = 1.0
    delta: float = -0.005
    start_epoch: int = 100
    every_n_epochs: int = 5
    floor: float = 0.9
    # Set in __post_init__; K.variable (not a bare tf.Variable) so the Keras
    # backend session initializes it -- see gannoc.training for graph mode.
    variable: tf.Variable = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.variable = K.variable(self.initial, dtype="float32", name="LAMBDA")

    @property
    def value(self) -> float:
        """Current ``LAMBDA`` as a plain Python float."""
        return float(K.get_value(self.variable))

    def step(self, epoch: int) -> float:
        """Apply the schedule for ``epoch`` (call once per epoch). Returns the new LAMBDA.

        A no-op before ``start_epoch``, off the ``every_n_epochs`` cadence, or
        once ``floor`` has been reached; otherwise ``LAMBDA`` is decremented by
        ``delta`` and clamped at ``floor``.
        """
        current = self.value
        if epoch >= self.start_epoch and epoch % self.every_n_epochs == 0 and current > self.floor:
            new_value = max(self.floor, current + self.delta)
            K.set_value(self.variable, new_value)
            return new_value
        return current


def make_generator_wasserstein_loss(schedule: "LambdaSchedule"):
    """Return a Keras loss ``LAMBDA * mean(y_true * y_pred)``.

    Closes over ``schedule.variable`` so each training step picks up the
    running annealed value.
    """

    def generator_wasserstein_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return schedule.variable * K.mean(y_true * y_pred)

    return generator_wasserstein_loss


def make_reward_mse_loss(schedule: "LambdaSchedule"):
    """Return a Keras loss ``(1 - LAMBDA) * mean(sum((y_true - y_pred) ** 2, axis=-1))``.

    Closes over ``schedule.variable``; as ``LAMBDA`` anneals down the frozen
    reward network's target connection count gains influence over the
    generator.
    """

    def reward_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return (1.0 - schedule.variable) * K.mean(K.sum(K.square(y_true - y_pred), axis=-1))

    return reward_mse_loss
