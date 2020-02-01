import tensorflow as tf

from layers.utils import CustomLayer
from layers.activation import Maxout

class AnonymousNetBlock(CustomLayer):
    def __init__(self, units, strides=1, activation=None, batch_norm=True):
        super().__init__(units=units, strides=strides, activation=activation, batch_norm=batch_norm)
        self.units = units
        self.strides = strides
        assert units % 2 == 0
        self.units = units * 2 if activation else units
        self.activation = activation
        self.batch_norm = batch_norm

    def build(self, input_shape):

        self.dw_conv_bottleneck = tf.keras.layers.Conv2D(
            filters=self.units // 2,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=[3, 3],
            strides=1,
            padding='same'
        )
        self.res = tf.keras.layers.Conv2D(
            filters=self.units,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )

        if self.strides != 1:
            self.size_reduce_pool_bottleneck = tf.keras.layers.Conv2D(
                filters=self.units//2,
                kernel_size=[1, 1],
                strides=1,
                padding='same'
            )
            self.size_reduce_pool = tf.keras.layers.MaxPool2D(
                pool_size=[3, 3],
                strides=self.strides,
                padding='same'
            )
            self.size_reduce_dw_conv_bottleneck = tf.keras.layers.Conv2D(
                filters=self.units//2,
                kernel_size=[1, 1],
                strides=self.strides,
                padding='same'
            )
            self.size_reduce_dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=[3, 3],
                strides=1,
                padding='same'
            )

        if self.activation:
            self.activate_bottleneck = tf.keras.layers.Conv2D(
                filters=self.units//4*3,
                kernel_size=[1, 1],
                strides=1,
                padding='same'
            )
            self.activate = Maxout(units=self.units//4)
            self.no_activate = tf.keras.layers.Conv2D(
                filters=self.units//4,
                kernel_size=[1, 1],
                strides=1,
                padding='same'
            )
        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x_dw_conv_bottleneck = self.dw_conv_bottleneck(inputs)
        x_dw_conv = self.dw_conv(x_dw_conv_bottleneck)
        x_res = self.res(inputs)
        x = tf.concat([x_dw_conv, x_res], axis=-1)

        if self.strides != 1:
            x_size_reduce_pool_bottleneck = self.size_reduce_pool_bottleneck(x)
            x_size_reduce_pool = self.size_reduce_pool(x_size_reduce_pool_bottleneck)
            x_size_reduce_dw_conv_bottleneck = self.size_reduce_dw_conv_bottleneck(x)
            x_size_reduce_dw_conv = self.size_reduce_dw_conv(x_size_reduce_dw_conv_bottleneck)
            x = tf.concat([x_size_reduce_pool, x_size_reduce_dw_conv], axis=-1)

        if self.activation:
            x_activate_bottleneck = self.activate_bottleneck(x)
            x_activate = self.activate(x_activate_bottleneck)
            x_no_activate = self.no_activate(x)
            x = tf.concat([x_activate, x_no_activate], axis=-1)

        if self.batch_norm:
            x = self.bn(x)
        return x
