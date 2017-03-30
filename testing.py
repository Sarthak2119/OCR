import cv2
from numba.types import double
from skimage.morphology import skeletonize, thin
from skimage.util import invert
import preprocess as pre
import utility as util
from matplotlib import pyplot as plt
from skimage import measure
import numpy as np

img = cv2.imread('/home/sarthak/PycharmProjects/OCR/Fnt/Sample009/img009-00033.png', 0)
cutt = pre.cut_image(img)

util.display_image(img)
util.display_image(cutt)

img = pre.thin_image(img)
cutt = pre.thin_image(cutt)

print(img.shape)
print(cutt.shape)