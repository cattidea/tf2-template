import tensorflow as tf
from layers.utils import CustomLayer

class InceptionModule(CustomLayer):
    def __init__(self, units, activation=None):
        super().__init__(units=units, activation=activation)
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
