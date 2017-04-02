import cv2
from numba.types import double
from skimage.morphology import skeletonize, thin
from skimage.util import invert
import preprocess as pre
import utility as util
from matplotlib import pyplot as plt
from skimage import measure
import numpy as np
import features
from scipy import signal

img = cv2.imread('/home/sarthak/PycharmProjects/OCR/Fnt/Sample023/img023-00046.png', 0)

a, b, c = features.pre_init(img)

# plt.imshow(c, plt.cm.gray)
# plt.show()

l,inte = features.count_intersect(c)
print(l)
