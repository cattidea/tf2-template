import tensorflow as tf

from layers.utils import CustomLayer, IdentityLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py
"""

def _ConvBlock(filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
        tf.keras.layers.Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.)
    ])

def _DepthwiseConvBlock(pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1)):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    return tf.keras.Sequential([
        IdentityLayer() if strides == (1, 1) else tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1))),
        tf.keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.)
    ])

def MobileNet(alpha=1.0, depth_multiplier=1):
    return tf.keras.Sequential([
        _ConvBlock(32, alpha, strides=(2, 2)),
        _DepthwiseConvBlock(64, alpha, depth_multiplier),
        _DepthwiseConvBlock(128, alpha, depth_multiplier, strides=(2, 2)),
        _DepthwiseConvBlock(128, alpha, depth_multiplier),
        _DepthwiseConvBlock(256, alpha, depth_multiplier, strides=(2, 2)),
        _DepthwiseConvBlock(256, alpha, depth_multiplier),
        _DepthwiseConvBlock(512, alpha, depth_multiplier, strides=(2, 2)),
        _DepthwiseConvBlock(512, alpha, depth_multiplier),
        _DepthwiseConvBlock(512, alpha, depth_multiplier),
        _DepthwiseConvBlock(512, alpha, depth_multiplier),
        _DepthwiseConvBlock(512, alpha, depth_multiplier),
        _DepthwiseConvBlock(512, alpha, depth_multiplier),
        _DepthwiseConvBlock(1024, alpha, depth_multiplier, strides=(2, 2)),
        _DepthwiseConvBlock(1024, alpha, depth_multiplier)
    ])
