import tensorflow as tf

from layers.utils import CustomLayer

class SELayer(CustomLayer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):

        B, H, W, C = input_shape
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential([
            tf.keras.layers.Dense(C//16),
            tf.keras.layers.Dense(C, activation='sigmoid')
        ])
        self.multi = tf.keras.layers.Multiply()


    def call(self, inputs):
        scale = inputs
        scale = self.squeeze(scale)
        scale = self.excitation(scale)
        outputs = self.multi([inputs, scale])
        return outputs
