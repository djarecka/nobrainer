import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

from nobrainer import layers


def test_zeropadding3dchannels():
    # This test function is a much shorter version of
    # `tensorflow.python.keras.testing_utils.layer_test`.
    input_data_shape = (4, 32, 32, 32, 1)
    input_data = 10 * np.random.random(input_data_shape)

    x = tf.keras.layers.Input(shape=input_data_shape[1:], dtype=input_data.dtype)
    y = layers.ZeroPadding3DChannels(4)(x)
    model = tf.keras.Model(x, y)

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert actual_output_shape == (4, 32, 32, 32, 9)
    assert not actual_output[..., :4].any()
    assert actual_output[..., 4].any()
    assert not actual_output[..., 5:].any()

    return actual_output


def test_conv3dweightnorm():
    v = np.random.rand(4, 4, 4, 1, 1).astype(np.float32)
    g = 1
    w = g * v / np.sqrt(np.square(v).sum((0, 1, 2, 3)))
    layer = layers.Conv3DWeightNorm(
        1, 4, padding='same',
        kernel_initializer=tf.keras.initializers.Constant(v))
    layer.build((None, 8, 8, 8, 1))
    assert_allclose(layer.kernel.numpy(), w, atol=1e-07)


def test_denseweightnorm():
    v = np.random.rand(10, 10).astype(np.float32)
    g = 1
    w = g * v / np.sqrt(np.square(v).sum(0))
    layer = layers.DenseWeightNorm(
        10, kernel_initializer=tf.keras.initializers.Constant(v))
    layer.build((None, 10))
    assert_allclose(layer.kernel.numpy(), w, atol=1e-07)
