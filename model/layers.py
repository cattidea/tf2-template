import tensorflow as tf


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        assert units % 32 == 0
        self.k = units // 32
        self.activation = activation

    def build(self, input_shape):

        k = self.k
        self.conv_1_1 = tf.keras.layers.Conv2D(
            filters=8*k,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )
        self.conv_1_1_t3 = tf.keras.layers.Conv2D(
            filters=12*k,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )
        self.conv_3_3 = tf.keras.layers.Conv2D(
            filters=16*k,
            kernel_size=[3, 3],
            strides=1,
            padding='same'
        )
        self.conv_1_1_t5 = tf.keras.layers.Conv2D(
            filters=k,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )
        self.conv_3_3_t5 = tf.keras.layers.Conv2D(
            filters=2*k,
            kernel_size=[3, 3],
            strides=1,
            padding='same'
        )
        self.conv_5_5 = tf.keras.layers.Conv2D(
            filters=4*k,
            kernel_size=[3, 3],
            strides=1,
            padding='same'
        )
        self.pool_t = tf.keras.layers.MaxPool2D(
            pool_size=[3, 3],
            strides=1,
            padding='same'
        )
        self.pool = tf.keras.layers.Conv2D(
            filters=4*k,
            kernel_size=[1, 1],
            strides=1,
            padding='same'
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x_1_1 = self.conv_1_1(inputs)
        x_1_1_t3 = self.conv_1_1_t3(inputs)
        x_3_3 = self.conv_3_3(x_1_1_t3)
        x_1_1_t5 = self.conv_1_1_t5(inputs)
        x_3_3_t5 = self.conv_3_3_t5(x_1_1_t5)
        x_5_5 = self.conv_5_5(x_3_3_t5)
        x_pool_t = self.pool_t(inputs)
        x_pool = self.pool(x_pool_t)
        x = tf.concat([x_1_1, x_3_3, x_5_5, x_pool], axis=-1)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = {"units": self.units, 'activation': self.activation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Maxout(tf.keras.layers.Layer):
    def __init__(self, units, axis=None):
        super().__init__()
        self.units = units
        self.axis = axis
        if axis is None:
            self.axis = -1


    def build(self, input_shape):

        self.num_channels = input_shape[self.axis]
        assert self.num_channels % self.units == 0
        self.out_shape = input_shape.as_list()
        self.out_shape[self.axis] = self.units
        self.out_shape += [self.num_channels // self.units]
        for i in range(len(self.out_shape)):
            if self.out_shape[i] is None:
                self.out_shape[i] = -1

    def call(self, inputs):

        x = tf.reduce_max(tf.reshape(inputs, self.out_shape), -1, keepdims=False)
        return x

    def get_config(self):
        config = {"units": self.units, 'axis': self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AnonymousNetBlock(tf.keras.layers.Layer):
    def __init__(self, units, strides=1, activation=None, batch_norm=True):
        super().__init__()
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

    def get_config(self):
        config = {"units": self.units, 'strides': self.strides, 'activation': self.activation, 'batch_norm': self.batch_norm}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
