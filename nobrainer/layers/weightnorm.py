"""Custom Convolution layers for nobrainer that implement weight normalization.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Conv3DWeightNorm(layers.Conv3D):
    """Volumetric convolution that implements weight normalization.

    Reference
    ---------
    https://arxiv.org/abs/1602.07868
    """

    def build(self, input_shape):
        self.g = self.add_weight(
            name='g_weightnorm',
            shape=(1, 1, 1, 1, self.filters),
            dtype=self.dtype,
            initializer='ones',
            trainable=True)
        super().build(input_shape=input_shape)
        v_over_vnorm = tf.nn.l2_normalize(self.kernel, axis=(0, 1, 2, 3))
        self.kernel = self.g * v_over_vnorm


class DenseWeightNorm(layers.Dense):
    """Densely connected layer implements weight normalization.

    Reference
    ---------
    https://arxiv.org/abs/1602.07868
    """

    def build(self, input_shape):
        self.g = self.add_weight(
            name='g_weightnorm',
            shape=(1, self.units),
            dtype=self.dtype,
            initializer='ones',
            trainable=True)
        super().build(input_shape=input_shape)
        v_over_vnorm = tf.nn.l2_normalize(self.kernel, axis=(0,))
        self.kernel = self.g * v_over_vnorm
