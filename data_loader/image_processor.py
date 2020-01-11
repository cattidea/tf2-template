import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import imageio
import skimage
import copy
import cv2
from scipy import misc, ndimage

random.seed(None)
np.random.seed(None)

def rotate(img):
    """ 旋转扩增 """
    angle = np.random.randint(-20, 20)
    cval = np.random.randint(0, 256)
    reshape = np.random.random() < 0.5
    return ndimage.rotate(img, angle, reshape=reshape, cval=cval)

def flip(img):
    """ 水平镜像扩增 """
    return np.fliplr(img)

def blur(img):
    """ Median Filter """
    sigma_t = np.random.randint(1, 4)
    return ndimage.median_filter(img, size=2*sigma_t+1)

def noise(img):
    """ Random Noise """
    mode = random.choice(['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
    return (skimage.util.random_noise(img, mode=mode) * 255).astype(np.uint8)

def resize(img, shape=(96, 96)):
    if np.random.random() < 0.5:
        h, w = img.shape[: 2]
        if h < w:
            start = np.random.randint(0, w - h)
            img = img[:, start: start+h]
        elif w > h:
            start = np.random.randint(0, h - w)
            img = img[start: start+w]
    return cv2.resize(img, shape)



class Process():

    def __init__(self, method, *args, **kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img):
        return self.method(img, *self.args, **self.kwargs)


class ImageProcessor():

    def __init__(self):
        self.__processing_list = []
        self.processes = None
        self.out_size = None

    def apply(self, method, probability=0.5, *args, **kwargs):
        process = Process(method, *args, **kwargs)
        self.__processing_list.append((process, probability))
        return self

    def resize_to(self, shape):
        self.out_size = shape
        return self

    def compile(self):
        processes_with_probability_list = [([], 1)]
        for new_process, new_probability in self.__processing_list:
            new_processes_with_probability_list = []
            for processes, probability in processes_with_probability_list:
                new_processes_with_probability_list.append((
                    processes + [new_process], probability * new_probability
                ))
                new_processes_with_probability_list.append((
                    processes, probability * (1-new_probability)
                ))
            processes_with_probability_list = new_processes_with_probability_list
        assert abs(sum([p for _, p in processes_with_probability_list]) - 1) < 1e-6

        self.processes_with_probability_list = processes_with_probability_list

    def __call__(self, img):
        r = np.random.random()
        threshold = 0
        for processes, probability in self.processes_with_probability_list:
            threshold += probability
            if r < threshold:
                break
        random.shuffle(processes)
        for process in processes:
            img = process(img)
        if self.out_size is not None:
            img = resize(img, self.out_size)
        return img

class ImageProcessorFromPath(ImageProcessor):

    def __init__(self):
        super().__init__()
        self.cache = {}

    def __call__(self, img_path):
        if self.cache.get(img_path):
            img = self.cache[img_path]
        else:
            # img = imageio.imread(img_path)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            assert len(img.shape) == 3, "%s error" % img_path
            self.cache[img_path] = img
        return super().__call__(img)




# face = misc.face(gray=False)
# path = 'data/TrainSet/001.ak47/Train_001_0001.jpg'
# path = 'data/TestSet/257.clutter/Test_257_0010.jpg'
# # img = imageio.imread()
# img = Image(path)
# processor = ImageProcessor()
# processor.apply(filter, 0.3)\
#         .apply(rotate, 1)\
#         .apply(flip, 0.1)\
#         .apply(noise, 0.2)
# processor.compile()

# for i in range(10):
#     image = processor(img.data)
#     # print(image)
#     # print(help(ndimage))
#     # help(ndimage.rotate)
#     image = resize(image, (96, 96))
#     plt.imshow(image)
#     plt.show()


