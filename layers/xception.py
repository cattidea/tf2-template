import tensorflow as tf

from layers.utils import CustomLayer

"""
ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/xcepiton.py
"""

class Xception(CustomLayer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.residual_0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.separable_conv_0 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])
        self.residual_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.separable_conv_1 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])
        self.residual_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.separable_conv_2 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])
        for i in range(8):
            setattr(self, 'separable_conv_'+str(i+3), tf.keras.Sequential([
                tf.keras.layers.ReLU(),
                tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]))
        self.residual_11 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])
        self.separable_conv_11 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])
        self.separable_conv_12 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])



    def call(self, inputs):
        x = inputs
        x = self.preprocess(x)
        x = tf.keras.layers.add([self.residual_0(x), self.separable_conv_0(x)])
        x = tf.keras.layers.add([self.residual_1(x), self.separable_conv_1(x)])
        x = tf.keras.layers.add([self.residual_2(x), self.separable_conv_2(x)])
        x = tf.keras.layers.add([x, self.separable_conv_3(x)])
        x = tf.keras.layers.add([x, self.separable_conv_4(x)])
        x = tf.keras.layers.add([x, self.separable_conv_5(x)])
        x = tf.keras.layers.add([x, self.separable_conv_6(x)])
        x = tf.keras.layers.add([x, self.separable_conv_7(x)])
        x = tf.keras.layers.add([x, self.separable_conv_8(x)])
        x = tf.keras.layers.add([x, self.separable_conv_9(x)])
        x = tf.keras.layers.add([x, self.separable_conv_10(x)])
        x = tf.keras.layers.add([self.residual_11(x), self.separable_conv_11(x)])
        x = self.separable_conv_12(x)
        return x
