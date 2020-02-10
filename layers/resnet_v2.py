import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
"""

class Block2(CustomLayer):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=False):
        super().__init__(filters=filters, kernel_size=kernel_size, stride=stride, conv_shortcut=conv_shortcut)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut

    def build(self, input_shape):

        self.preact = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU()
        ])
        if self.conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2D(4 * self.filters, 1, strides=self.stride)
        elif self.stride > 1:
            self.shortcut = tf.keras.layers.MaxPooling2D(1, strides=self.stride)
        else:
            self.shortcut = lambda x: x
        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters, 1, strides=1, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, use_bias=False),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(4 * self.filters, 1)
        ])

    def call(self, inputs):
        x = inputs
        x = self.preact(x)
        x = tf.keras.layers.add([self.shortcut(x), self.conv_block(x)])
        return x


def Stack2(filters, blocks, stride1=2):
    return tf.keras.Sequential([
        Block2(filters, conv_shortcut=True),
    ] + [
        Block2(filters) for i in range(blocks-2)
    ] + [
        Block2(filters, stride=stride1)
    ])

def ResNetV2(blocks):
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3))),
        tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True),
        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        Stack2(64, blocks[0], stride1=1),
        Stack2(128, blocks[1]),
        Stack2(256, blocks[2]),
        Stack2(512, blocks[3]),
        tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
        tf.keras.layers.ReLU(),
    ])

def ResNet50V2():
    return ResNetV2([3, 4, 6, 3])

def ResNet101V2():
    return ResNetV2([3, 4, 23, 3])

def ResNet152V2():
    return ResNetV2([3, 8, 36, 3])
