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
        g = self.add_weight(
            name='v',
            shape=(1, 1, 1, 1, self.filters),
            initializer='ones',
            trainable=True)
        super().build(input_shape=input_shape)
        # l2_normalize calculates v / ||v||
        self.kernel = g * tf.nn.l2_normalize(self.kernel, axis=[0, 1, 2, 3])
