import numpy as np
import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=257)

    def call(self, inputs):         # [batch_size, 96, 96, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,             # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv4 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=257)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 96, 96, 8]
        x = self.pool1(x)                       # [batch_size, 48, 48, 8]
        x = self.conv2(x)                       # [batch_size, 48, 48, 8]
        x = self.pool2(x)                       # [batch_size, 24, 24, 8]
        x = self.conv3(x)                       # [batch_size, 24, 24, 8]
        x = self.pool3(x)                       # [batch_size, 12, 12, 8]
        x = self.conv4(x)                       # [batch_size, 12, 12, 8]
        x = self.pool4(x)                       # [batch_size, 6, 6, 8]
        x = self.flatten(x)                     # [batch_size, 6 * 6 * 8]
        x = self.dense1(x)                      # [batch_size, 128]
        x = self.dense2(x)                      # [batch_size, 257]
        output = tf.nn.softmax(x)
        return output
