import preprocess as pre
import utility as util
import segmentation as seg
import histogram as hist
from matplotlib import pyplot as plt

def run(img):
    #input image in grayscale
    # img = pre.input_image('/home/sarthak/ip/text1.png')
    # util.display_image(img, 'original')

    #filtering of the input image
    img = pre.filter_image(img)
    # util.display_image(img, 'filtered')

    #smoothening the image
    # img = preprocess.gaussian_blur(img)
    # utility.display_image(img, 'Smoothened')

    #binarization/thresholding of the image
    img = pre.otsu_thresh(img)
    # util.display_image(img, 'binarized')

    #horizontal histogram for line segmentation
    lines = hist.line_seg(img)

    #displaying lines as recieved
    # for sline in lines:
    #     util.display_image(sline)

    #vertical histogram for words segmentation
    words = []
    for ekline in lines:
        # util.display_image(ekline)
        curr = hist.word_seg(ekline)
        words = words + curr
        # break #delete it

    #displaying words of lines as recieved
    # for ekword in words:
    #     util.display_image(ekword)

    chars = []
    #separating characters from words
    # cnt = 0
    for ekword in words:
        clist = hist.char_seg(ekword)
        chars.append(clist)

    # displaying characters as recieved
    # for clist in chars:
    #     for ch in clist:
    #         util.display_image(ch)
    return chars