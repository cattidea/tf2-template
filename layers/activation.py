import tensorflow as tf

from layers.utils import CustomLayer

class Maxout(CustomLayer):
    def __init__(self, units, axis=None):
        super().__init__(units=units, axis=axis)
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
