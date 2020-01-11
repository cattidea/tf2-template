import time

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from config_parser.config import CONFIG
from data_loader.batch_loader import batch_loader
from data_loader.data_loader import load_data
from model.models import CNN, MLP

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                # iaa.OneOf([
                #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                # ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                # iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                # iaa.Grayscale(alpha=(0.0, 1.0)),
                # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        ),
        iaa.Resize({"height":CONFIG.image.H, "width":CONFIG.image.W})
    ],
    random_order=True
)

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


def train():

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

    # tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    epoch = 10
    batch_size = 64
    learning_rate = 0.001
    reload_data_step = 5

    # model = MLP()
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    X_path, y = load_data(data_type='train')
    X_path_train, X_path_dev, y_train, y_dev = train_test_split(X_path, y, test_size=0.5, random_state=42)

    train_size, dev_size = len(X_path_train), len(X_path_dev)
    X_dev = load_data_from_path(X_path_dev, shape=(CONFIG.image.W, CONFIG.image.W), part=-1)

    for e in range(epoch):

        if e % reload_data_step == 0:
            X_train = load_data_from_path(X_path_train)

        for X_batch, y_batch in batch_loader(X_train, y_train, batch_size=batch_size):
            t1 = time.time()
            X_batch = np.divide(seq(images=X_batch) / 255, dtype=np.float32)  # done by the library
            t2 = time.time()

            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_batch, y_pred=y_pred)
                # print(y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (e, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

            t3 = time.time()
            print(t2 - t1, t3-t2)

        # sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        # for path_batch, y_batch in batch_loader(X_dev, y_dev, batch_size=batch_size):


        #     y_pred = model.predict(batch_path_to_img(path_batch, read_and_resize))
        #     sparse_categorical_accuracy.update_state(y_true=y_batch, y_pred=y_pred)
        # print("test accuracy: %f" % sparse_categorical_accuracy.result())
