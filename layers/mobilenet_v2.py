import tensorflow as tf

from layers.utils import CustomLayer, IdentityLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
"""

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling."""

    input_size = input_shape[1: 3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class _InvertedResBlock(CustomLayer):
    def __init__(self, expansion, stride, alpha, filters):
        super().__init__(expansion=expansion, stride=stride, alpha=alpha, filters=filters)
        self.expansion = expansion
        self.stride = stride
        self.alpha = alpha
        self.filters = filters

    def build(self, input_shape):

        in_channels = input_shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        self.use_residual = in_channels == pointwise_filters and self.stride == 1

        if self.expansion != 1:
            self.preact = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.expansion * in_channels, kernel_size=1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
                tf.keras.layers.ReLU(6.)
            ])
        else:
            self.preact = IdentityLayer()

        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=_correct_pad(input_shape, 3)) if self.stride == 2 else IdentityLayer(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=self.stride, use_bias=False,
                                            padding='same' if self.stride == 1 else 'valid'),
            tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
            tf.keras.layers.ReLU(6.),
            tf.keras.layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
        ])

    def call(self, inputs):
        x = inputs
        x = self.preact(x)
        x = self.conv_block(x)
        if self.use_residual:
            x = tf.keras.layers.add([inputs, x])
        return x


class PreprocessBlock(CustomLayer):
    def __init__(self, alpha):
        super().__init__(alpha=alpha)
        self.alpha = alpha

    def build(self, input_shape):

        first_block_filters = _make_divisible(32 * self.alpha, 8)

        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=_correct_pad(input_shape, 3)),
            tf.keras.layers.Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
            tf.keras.layers.ReLU(6.)
        ])

    def call(self, inputs):
        x = inputs
        x = self.conv_block(x)
        return x


class PostprocessBlock(CustomLayer):
    def __init__(self, alpha):
        super().__init__(alpha=alpha)
        self.alpha = alpha

    def build(self, input_shape):

        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
            tf.keras.layers.ReLU(6.)
        ])

    def call(self, inputs):
        x = inputs
        x = self.conv_block(x)
        return x


def MobileNetV2(alpha=1.0):
    return tf.keras.Sequential([
        PreprocessBlock(alpha),
        _InvertedResBlock(filters=16, alpha=alpha, stride=1, expansion=1),
        _InvertedResBlock(filters=24, alpha=alpha, stride=2, expansion=6),
        _InvertedResBlock(filters=24, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=32, alpha=alpha, stride=2, expansion=6),
        _InvertedResBlock(filters=32, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=32, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=64, alpha=alpha, stride=2, expansion=6),
        _InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=64, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=96, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=160, alpha=alpha, stride=2, expansion=6),
        _InvertedResBlock(filters=160, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=160, alpha=alpha, stride=1, expansion=6),
        _InvertedResBlock(filters=320, alpha=alpha, stride=1, expansion=6),
        PostprocessBlock(alpha)
    ])
