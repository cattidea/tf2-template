from config_parser.config import CONFIG
from data_loader.data_loader import load_data

from sklearn.model_selection import train_test_split
from data_loader.batch_loader import batch_loader, batch_path_to_img
from data_loader.image_processor import ImageProcessorFromPath, rotate, flip, blur, noise, read_and_resize

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt

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
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv4 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(6 * 6 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=257)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 96, 96, 32]
        x = self.pool1(x)                       # [batch_size, 48, 48, 32]
        x = self.conv2(x)                       # [batch_size, 48, 48, 64]
        x = self.pool2(x)                       # [batch_size, 24, 24, 64]
        x = self.conv3(x)                       # [batch_size, 24, 24, 64]
        x = self.pool3(x)                       # [batch_size, 12, 12, 64]
        x = self.conv4(x)                       # [batch_size, 12, 12, 64]
        x = self.pool4(x)                       # [batch_size, 6, 6, 64]
        x = self.flatten(x)                     # [batch_size, 6 * 6 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


def train():

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

    # tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    epoch = 10
    batch_size = 64
    learning_rate = 0.001

    model = MLP()
    # model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    X, y = load_data(data_type='train')
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.5, random_state=42)

    train_size, dev_size = len(X_train), len(X_dev)

    processor = ImageProcessorFromPath()
    processor.apply(blur, 0.3)\
            .apply(rotate, 1)\
            .apply(flip, 0.1)\
            .apply(noise, 0.2)\
            .resize_to((CONFIG.image.W, CONFIG.image.H))\
            .compile()

    for e in range(epoch):

        for path_batch, label_batch in batch_loader(X_train, y_train, batch_size=batch_size):
            # real_batch_size = len(path_batch)
            # X_batch = np.zeros(shape=(real_batch_size, CONFIG.image.H, CONFIG.image.W, 3), dtype=np.uint8)
            # for i, path in enumerate(path_batch):
            #     # print(processor(path))
            #     # print(X[i].shape)
            #     X_batch[i] = processor(path)
            #     # plt.imshow(X_batch[i])
            #     # plt.show()

            # # print(X_batch, label_batch)
            t1 = time.time()
            X_batch = batch_path_to_img(path_batch, processor)
            t2 = time.time()

            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_batch, y_pred=y_pred)
                # print(y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (e, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

            t3 = time.time()
            print(t2 - t1, t3-t2)

        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for path_batch, label_batch in batch_loader(X_dev, y_dev, batch_size=batch_size):


            y_pred = model.predict(batch_path_to_img(path_batch, read_and_resize))
            sparse_categorical_accuracy.update_state(y_true=label_batch, y_pred=y_pred)
        print("test accuracy: %f" % sparse_categorical_accuracy.result())



