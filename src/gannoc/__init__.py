"""GANNoC: reward-guided generation of Network-on-Chip topologies with a WGAN-GP.

A NoC topology over a fixed number of routers is represented as a binary
adjacency matrix and treated as a single-channel image. A WGAN-GP learns to
generate valid topologies (connected, router degree <= 4); a separately
pretrained, frozen *reward* network then scores each generated topology on a
target property (its number of physical connections) and its loss is blended
into the generator objective, biasing generation toward that property. The
reward-guided variant is RWGAN ("Reward-Wasserstein GAN"); disabling the reward
term recovers the plain WGAN-GP baseline.

The package provides the data representation and synthetic dataset generator,
the generator/critic/reward model builders, the loss machinery, the reward and
RWGAN training loops, and a generator-evaluation routine.
"""
