import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
"""

def _Conv2d_BN(filters, kernel_size, padding='same', strides=(1, 1)):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False),
        tf.keras.layers.BatchNormalization(scale=False),
        tf.keras.layers.ReLU()
    ])

def PreprocessBlock():
    return tf.keras.Sequential([
        _Conv2d_BN(32, (3, 3), strides=(2, 2), padding='valid'),
        _Conv2d_BN(32, (3, 3), padding='valid'),
        _Conv2d_BN(64, (3, 3)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
        _Conv2d_BN(80, (1, 1), padding='valid'),
        _Conv2d_BN(192, (3, 3), padding='valid'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
    ])

class Mixed0(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(64, (1, 1))
        self.branch5x5 = tf.keras.Sequential([
            _Conv2d_BN(48, (1, 1)),
            _Conv2d_BN(64, (5, 5))
        ])
        self.branch3x3dbl = tf.keras.Sequential([
            _Conv2d_BN(64, (1, 1)),
            _Conv2d_BN(96, (3, 3)),
            _Conv2d_BN(96, (3, 3))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(32, (1, 1))
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.branch5x5(inputs),
            self.branch3x3dbl(inputs),
            self.branch_pool(inputs)
        ])


class Mixed1(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(64, (1, 1))
        self.branch5x5 = tf.keras.Sequential([
            _Conv2d_BN(48, (1, 1)),
            _Conv2d_BN(64, (5, 5))
        ])
        self.branch3x3dbl = tf.keras.Sequential([
            _Conv2d_BN(64, (1, 1)),
            _Conv2d_BN(96, (3, 3)),
            _Conv2d_BN(96, (3, 3))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(64, (1, 1))
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.branch5x5(inputs),
            self.branch3x3dbl(inputs),
            self.branch_pool(inputs)
        ])


class Mixed3(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch3x3 = _Conv2d_BN(384, (3, 3), strides=(2, 2), padding='valid')
        self.branch3x3dbl = tf.keras.Sequential([
            _Conv2d_BN(64, (1, 1)),
            _Conv2d_BN(96, (3, 3)),
            _Conv2d_BN(96, (3, 3), strides=(2, 2), padding='valid')
        ])
        self.branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch3x3(inputs),
            self.branch3x3dbl(inputs),
            self.branch_pool(inputs)
        ])


class Mixed4(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(192, (1, 1))
        self.branch7x7 = tf.keras.Sequential([
            _Conv2d_BN(128, (1, 1)),
            _Conv2d_BN(128, (1, 7)),
            _Conv2d_BN(192, (7, 1))
        ])
        self.branch7x7dbl = tf.keras.Sequential([
            _Conv2d_BN(128, (1, 1)),
            _Conv2d_BN(128, (7, 1)),
            _Conv2d_BN(128, (1, 7)),
            _Conv2d_BN(128, (7, 1)),
            _Conv2d_BN(192, (1, 7))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(192, (1, 1))
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.branch7x7(inputs),
            self.branch7x7dbl(inputs),
            self.branch_pool(inputs)
        ])


class Mixed5(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(192, (1, 1))
        self.branch7x7 = tf.keras.Sequential([
            _Conv2d_BN(160, (1, 1)),
            _Conv2d_BN(160, (1, 7)),
            _Conv2d_BN(192, (7, 1))
        ])
        self.branch7x7dbl = tf.keras.Sequential([
            _Conv2d_BN(160, (1, 1)),
            _Conv2d_BN(160, (7, 1)),
            _Conv2d_BN(160, (1, 7)),
            _Conv2d_BN(160, (7, 1)),
            _Conv2d_BN(192, (1, 7))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(192, (1, 1))
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.branch7x7(inputs),
            self.branch7x7dbl(inputs),
            self.branch_pool(inputs)
        ])


class Mixed7(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(192, (1, 1))
        self.branch7x7 = tf.keras.Sequential([
            _Conv2d_BN(192, (1, 1)),
            _Conv2d_BN(192, (1, 7)),
            _Conv2d_BN(192, (7, 1))
        ])
        self.branch7x7dbl = tf.keras.Sequential([
            _Conv2d_BN(192, (1, 1)),
            _Conv2d_BN(192, (7, 1)),
            _Conv2d_BN(192, (1, 7)),
            _Conv2d_BN(192, (7, 1)),
            _Conv2d_BN(192, (1, 7))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(192, (1, 1))
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.branch7x7(inputs),
            self.branch7x7dbl(inputs),
            self.branch_pool(inputs)
        ])

class Mixed8(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch3x3 = tf.keras.Sequential([
            _Conv2d_BN(192, (1, 1)),
            _Conv2d_BN(320, (3, 3), strides=(2, 2), padding='valid')
        ])
        self.branch7x7x3 = tf.keras.Sequential([
            _Conv2d_BN(192, (1, 1)),
            _Conv2d_BN(192, (1, 7)),
            _Conv2d_BN(192, (7, 1)),
            _Conv2d_BN(192, (3, 3), strides=(2, 2), padding='valid')
        ])
        self.branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch3x3(inputs),
            self.branch7x7x3(inputs),
            self.branch_pool(inputs)
        ])


class Mixed9(CustomLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.branch1x1 = _Conv2d_BN(320, (1, 1))
        self.branch3x3 = _Conv2d_BN(384, (1, 1))
        self.branch3x3_1 = tf.keras.Sequential([
            self.branch3x3,
            _Conv2d_BN(384, (1, 3))
        ])
        self.branch3x3_2 = tf.keras.Sequential([
            self.branch3x3,
            _Conv2d_BN(384, (3, 1))
        ])
        self.branch3x3db1 = tf.keras.Sequential([
            _Conv2d_BN(448, (1, 1)),
            _Conv2d_BN(384, (3, 3))
        ])
        self.branch3x3db1_1 = tf.keras.Sequential([
            self.branch3x3db1,
            _Conv2d_BN(384, (1, 3))
        ])
        self.branch3x3db1_2 = tf.keras.Sequential([
            self.branch3x3db1,
            _Conv2d_BN(384, (3, 1))
        ])
        self.branch_pool = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same'),
            _Conv2d_BN(192, (1, 1))
        ])
        self.concat3x3 = tf.keras.layers.Concatenate(axis=-1)
        self.concat3x3db1 = tf.keras.layers.Concatenate(axis=-1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        return self.concat([
            self.branch1x1(inputs),
            self.concat3x3([
                self.branch3x3_1(inputs),
                self.branch3x3_2(inputs)
            ]),
            self.concat3x3db1([
                self.branch3x3db1_1(inputs),
                self.branch3x3db1_2(inputs)
            ]),
            self.branch_pool(inputs)
        ])

def InceptionV3():
    return tf.keras.Sequential([
        PreprocessBlock(),
        Mixed0(),
        Mixed1(),
        Mixed1(),
        Mixed3(),
        Mixed4(),
        Mixed5(),
        Mixed5(),
        Mixed7(),
        Mixed8(),
        Mixed9(),
        Mixed9()
    ])
