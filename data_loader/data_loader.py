import numpy as np
import pandas as pd
import imageio



import os

from config_parser.config import CONFIG

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

# X, y = load_data(data_type='train')
# X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.5, random_state=42)

# print(len(X_train), len(X_dev))

# print(img_paths)
