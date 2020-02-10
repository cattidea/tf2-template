import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
"""

class Block1(CustomLayer):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True):
        super().__init__(filters=filters, kernel_size=kernel_size, stride=stride, conv_shortcut=conv_shortcut)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut

    def build(self, input_shape):

        if self.conv_shortcut:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(4 * self.filters, 1, strides=self.stride),
                tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
            ])
        else:
            self.shortcut = lambda x: x
        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters, 1, strides=self.stride),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(4 * self.filters, 1),
            tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        ])
        self.act = tf.keras.layers.ReLU()


    def call(self, inputs):
        x = inputs
        x = tf.keras.layers.add([self.shortcut(x), self.conv_block(x)])
        x = self.act(x)
        return x


def Stack1(filters, blocks, stride1=2):
    return tf.keras.Sequential([
        Block1(filters, stride=stride1),
    ] + [
        Block1(filters, conv_shortcut=False) for i in range(blocks-1)
    ])

def ResNet(blocks):
    return tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3))),
        tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True),
        tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        Stack1(64, blocks[0], stride1=1),
        Stack1(128, blocks[1]),
        Stack1(256, blocks[2]),
        Stack1(512, blocks[3]),
    ])

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])
