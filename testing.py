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

img = cv2.imread('/home/sarthak/PycharmProjects/OCR/Fnt/Sample004/img004-00001.png', 0)

aspect, old, new = features.pre_init(img)
l = features.get_dct(old)
print(l)