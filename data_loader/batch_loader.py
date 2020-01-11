import numpy as np

from config_parser.config import CONFIG

import matplotlib.pyplot as plt


def batch_loader(X, y, batch_size=64):

    data_size = len(X)
    permutation = np.random.permutation(data_size)
    batch_permutation_indices = [permutation[i: i + batch_size] for i in range(0, data_size, batch_size)]
    for batch_permutation in batch_permutation_indices:
        yield X[batch_permutation], y[batch_permutation]

def batch_path_to_img(path_batch, processor):
    batch_size = len(path_batch)
    X_batch = np.zeros(shape=(batch_size, CONFIG.image.H, CONFIG.image.W, 3), dtype=np.uint8)
    for i, path in enumerate(path_batch):
        X_batch[i] = processor(path)
        # plt.imshow(X_batch[i])
        # plt.show()
    return np.divide(X_batch, 255, dtype=np.float32)
