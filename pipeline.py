#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

# tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# In[2]:


from config_parser.config import CONFIG
from model.models import CNN, MLP
from model.layers import Maxout, InceptionModule, AnonymousNetBlock


# In[3]:


def get_name_list():
    num_classes = 257
    label_to_name = [None for i in range(num_classes)]
    base_dir = CONFIG['train'].data_dir
    dir_names = os.listdir(base_dir)
    for dir_name in dir_names:
        if not os.path.isdir(os.path.join(base_dir, dir_name)):
            continue
        label, name = dir_name.split('.')
        label_to_name[int(label)-1] = name
    assert all(label_to_name)
    return label_to_name

def load_data(data_type='train'):

    assert data_type in ['train', 'test']
    config = CONFIG[data_type]

    csv=pd.read_csv(config.data_labels_file)
    img_paths = []
    for name in csv[:]['Name'].values:
        name = name.strip("'").replace('\\', '/')
        img_path = os.path.normpath(os.path.join(config.data_dir, name))
        img_paths.append(img_path)
    img_paths = np.array(img_paths)

    labels = csv[:]['Label'].values - 1

    return img_paths, labels

def load_data_from_path(img_paths, shape='random', part='random'):
    size = min(CONFIG.image.W, CONFIG.image.H)
    size = int(size * (np.random.random()/2 + 1))
    if shape == 'random':
        shape = (size, size)
    data_size = len(img_paths)
    data = np.zeros((data_size, shape[1], shape[0], 3), dtype=np.uint8)
    for i, img_path in enumerate(img_paths):
        print("{}/{} load {} -> {}".format(i, data_size, img_path, shape), end='\r')
        if part == 'random':
            part = -1 if np.random.random() < 0.5 else np.random.random()
        data[i] = read_and_resize(img_path, shape=shape, part=part)
#         plt.imshow(data[i])
#         plt.show()
    return data

def read_and_resize(img_path, shape=(96, 96), part=-1):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if part != -1:
        h, w = img.shape[: 2]
        start = int(part * abs(w-h))
        if h < w:
            img = img[:, start: start+h]
        elif w > h:
            img = img[start: start+w]
    return cv2.resize(img, shape)

def batch_loader(X, y, batch_size=64):

    data_size = X.shape[0]
    permutation = np.random.permutation(data_size)
    batch_permutation_indices = [permutation[i: i + batch_size] for i in range(0, data_size, batch_size)]
    for batch_permutation in batch_permutation_indices:
        yield X[batch_permutation], y[batch_permutation]


# In[4]:


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10), # rotate by -45 to +45 degrees
            shear=(-3, 3), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
            iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
        ]),
        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
        iaa.Resize({"height":CONFIG.image.H, "width":CONFIG.image.W})
    ],
    random_order=True
)

resizer = iaa.Sequential(
    [
        iaa.Resize({"height":CONFIG.image.H, "width":CONFIG.image.W})
    ]
)


# In[5]:


num_epoch = 1000
batch_size = 16
learning_rate = 1e-4
reload_data_step = 5

# model = MLP()
# model = CNN()
# model = tf.keras.models.load_model(CONFIG.model.model_file)
# model = tf.keras.applications.MobileNetV2(weights=None, classes=257)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

test_input = np.zeros(shape=(1, CONFIG.image.H, CONFIG.image.W, 3), dtype=np.float32)


# In[6]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# In[7]:


base_model = tf.keras.applications.ResNet50(input_shape=(CONFIG.image.H, CONFIG.image.W, 3), weights='imagenet', include_top=False)
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(257),
    tf.keras.layers.Softmax()
])
# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(1, 1), padding='same'),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.SeparableConv2D(filters=257, kernel_size=(7, 7), padding='valid'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Softmax()
# ])
# print(model(test_input).shape)
# base_model.trainable = True
# print("Number of layers in the base model: ", len(base_model.layers))
# fine_tune_at = 300
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable =  False


# In[8]:


inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs
x = AnonymousNetBlock(units=16, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=16, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=32, strides=2, activation=True, batch_norm=True)(x)

x = AnonymousNetBlock(units=32, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=32, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=64, strides=2, activation=True, batch_norm=True)(x)

for i in range(8):
    x = AnonymousNetBlock(units=64, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=128, strides=2, activation=True, batch_norm=True)(x)

x = AnonymousNetBlock(units=128, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=128, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=256, strides=2, activation=True, batch_norm=True)(x)

x = AnonymousNetBlock(units=256, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=256, strides=1, activation=True, batch_norm=True)(x)
x = AnonymousNetBlock(units=512, strides=2, activation=True, batch_norm=True)(x)

x = tf.keras.layers.Conv2D(filters=257, kernel_size=[1, 1], padding='same')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Softmax()(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.summary()


# # Load Data

# In[9]:


X_path, y = load_data(data_type='train')
X_path_train, X_path_dev, y_train, y_dev = train_test_split(X_path, y, test_size=0.05, random_state=CONFIG.seed)

train_size, dev_size = len(X_path_train), len(X_path_dev)
X_dev = load_data_from_path(X_path_dev, shape=(CONFIG.image.W, CONFIG.image.H), part=-1)
X_train = load_data_from_path(X_path_train, shape=(CONFIG.image.W, CONFIG.image.H), part=-1)


# In[10]:


name_list = get_name_list()


# # Training

# In[ ]:


@tf.function
def train_on_batch(X_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        # tf.print(y_pred.shape)
        loss = loss_object(y_true=y_batch, y_pred=y_pred)
        # print(y_pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    train_loss(loss)
    train_accuracy(y_batch, y_pred)
    return loss

@tf.function
def test_on_batch(X_batch, y_batch):
    y_pred = model(X_batch)
    t_loss = loss_object(y_batch, y_pred)

    test_loss(t_loss)
    test_accuracy(y_batch, y_pred)
    return t_loss


for epoch in range(num_epoch):

#     # Reload Data
#     if (epoch+1) % reload_data_step == 0:
#         X_train = load_data_from_path(X_path_train)

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for batch_index, (X_batch, y_batch) in enumerate(batch_loader(X_train, y_train, batch_size=batch_size)):
        X_batch = np.divide(np.array(seq(images=X_batch)), 255, dtype=np.float32)  # done by the library
#         for i in range(len(X_batch)):
#             plt.imshow(X_batch[i])
#             plt.title('Real: {}'.format(name_list[y_batch[i]]))
#             plt.show()
        loss = train_on_batch(X_batch, y_batch)
        template = '[Training] Epoch {}, Batch {}/{}, Loss: {}, Accuracy: {:.2%} '
        print(template.format(epoch+1,
                              batch_index,
                              train_size // batch_size,
                              loss,
                              train_accuracy.result()),
             end='\r')

    for batch_index, (X_batch, y_batch) in enumerate(batch_loader(X_dev, y_dev, batch_size=batch_size)):
        X_batch = np.divide(X_batch, 255, dtype=np.float32)
        loss = test_on_batch(X_batch, y_batch)
        template = '[Testing] Epoch {}, Batch {}/{}, Loss: {}, Accuracy: {:.2%} '
        print(template.format(epoch+1,
                              batch_index,
                              dev_size // batch_size,
                              loss,
                              train_accuracy.result()),
             end='\r')

    template = 'Epoch {}, Loss: {}, Accuracy: {:.2%}, Test Loss: {}, Test Accuracy: {:.2%} '
    print(template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result(),
                         test_loss.result(),
                         test_accuracy.result()))

    model.save(CONFIG.model.model_file)


# # Testing

# In[ ]:


X_path_test, y_test = load_data(data_type='test')
X_test = load_data_from_path(X_path_test, shape=(CONFIG.image.W, CONFIG.image.H), part=-1)
test_size = len(X_path_test)


# In[ ]:


test_accuracy.reset_states()
for batch_index, (X_batch, y_batch) in enumerate(batch_loader(X_test, y_test, batch_size=batch_size)):
    X_batch = np.divide(X_batch, 255, dtype=np.float32)
    y_pred = model(X_batch)
    test_accuracy(y_batch, y_pred)
    y_pred_label = np.array([np.argmax(one_hot) for one_hot in y_pred.numpy()])
    template = '[Testing] Batch {}/{}, Accuracy: {:.2%} '
    print(template.format(batch_index,
                          test_size // batch_size,
                          test_accuracy.result()),
             end='\r')
    for i in range(X_batch.shape[0]):
        plt.imshow(X_batch[i])
        plt.title('Real: {}, Predict: {}'.format(name_list[y_batch[i]], name_list[y_pred_label[i]]))
        plt.show()

template = 'Test Accuracy: {:.2%} '
print(template.format(test_accuracy.result()))


# In[ ]:




