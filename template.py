import preprocess as pre
import utility as util
import segmentation as seg
import histogram as hist
from matplotlib import pyplot as plt
import cv2
import  numpy as np

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

    # print(len(lines))

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

    # print(len(words))

    # displaying words of lines as recieved
    # for ekword in words:
    #     util.display_image(ekword)

    chars = []
    #separating characters from words
    # cnt = 0
    for ekword in words:
        # util.display_image(ekword)
        clist = hist.char_seg(ekword)
        chars.append(clist)

    # displaying characters as recieved
    # for clist in chars:
    #     for ch in clist:
    #         util.display_image(ch)
    #     break

    return chars

    sum = 0
    cnt = 0
    maxx = -1
    for word in chars:
        for charr in word:
            cnt += 1
            maxx = max(maxx, len(charr[0]))
            sum += charr.shape[1]
    sum = sum/cnt
    print(sum)
    print(chars[0][2].shape)

    final = []
    for word in chars:
        new_word = []
        for charr in word:
            if charr.shape[1] > maxx:
                histo = img.sum(axis=0)
                minn = 1000
                pos = -1
                for i in range(0, len(histo)):
                    if histo[i] < minn:
                        minn = histo[i]
                        pos = i
                new_word.append(charr[:,:pos+1])
                new_word.append(charr[:, pos:])
            else:
                new_word.append(charr)
        final.append(new_word)

    # for clist in final:
    #     for ch in clist:
    #         util.display_image(ch)
    #     break

    return final

# img = cv2.imread('/home/sarthak/ip/hello.jpg', 0)
# run(img)