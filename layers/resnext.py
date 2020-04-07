import numpy as np
import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
"""

class Block3(CustomLayer):
    def __init__(self, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True):
        super().__init__(filters=filters, kernel_size=kernel_size, stride=stride, groups=groups, conv_shortcut=conv_shortcut)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv_shortcut = conv_shortcut
        self.channels_per_group = filters // groups
        self.conv_kernel = np.zeros((1, 1, filters * self.channels_per_group, filters), dtype=np.float32)


    def build(self, input_shape):

        c = self.channels_per_group
        for i in range(self.filters):
            start = (i // c) * c * c + i % c
            end = start + c * c
            self.conv_kernel[:, :, start:end:c, i] = 1.

        if self.conv_shortcut:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D((64 // self.groups) * self.filters, 1, strides=self.stride, use_bias=False),
                tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
            ])
        else:
            self.shortcut = lambda x: x

        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            tf.keras.layers.DepthwiseConv2D(self.kernel_size, strides=self.stride, depth_multiplier=c, use_bias=False),
            tf.keras.layers.Conv2D(self.filters, 1, use_bias=False, trainable=False, kernel_initializer=tf.constant_initializer(self.conv_kernel)),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D((64 // self.groups) * self.filters, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        ])
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = inputs
        x = tf.keras.layers.add([self.shortcut(x), self.conv_block(x)])
        x = self.act(x)
        return x


def Stack3(filters, blocks, stride1=2, groups=32):
    return tf.keras.Sequential([
        Block3(filters, stride=stride1, groups=groups),
    ] + [
        Block3(filters, groups=groups, conv_shortcut=False) for i in range(blocks-1)
    ])

def ResNeXt(blocks):
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3))),
        tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False),
        tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        Stack3(128, blocks[0], stride1=1),
        Stack3(256, blocks[1]),
        Stack3(512, blocks[2]),
        Stack3(1024, blocks[3]),
    ])

def ResNeXt50():
    return ResNeXt([3, 4, 6, 3])

def ResNeXt101():
    return ResNeXt([3, 4, 23, 3])

def ResNeXt152():
    return ResNeXt([3, 8, 36, 3])
