import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import preprocess as pre

def line_seg(img):
    img_not = abs(255 - img)
    #horizontal histogram
    histo = img_not.sum(axis=1)

    #smoothening histogram for better detection of lines
    y_smooth = savgol_filter(histo, 27 , 3)

    #comparing histogram after smoothening
    # plt.plot(histo)
    # plt.plot(y_smooth)

    minimas = argrelextrema(y_smooth, np.less)
    arr = minimas[0]
    # print(arr)

    # plotting minimas in histogram i.e. line breaks
    # for i in arr:
    #     plt.plot(i, y_smooth[i], 'ro')
    # plt.show()

    #creating a list of lines
    lines = []
    prev = 0
    for i in arr:
        if(prev != -1):
            curr = img_not[prev:i + 1, :]
            if(np.sum(curr) >= 6*255):
                curr = img[prev:i+1, :]
                curr = pre.cut_image(curr)
                lines.append(curr)
        prev = i

    i = len(histo)-1
    curr = img_not[prev:i + 1, :]
    if (np.sum(curr) >= 6 * 255):
        curr = img[prev:i + 1, :]
        curr = pre.cut_image(curr)
        lines.append(curr)
    return lines

def word_seg(img):
    img_not = abs(255 - img)
    #vertical histogram
    histo = img_not.sum(axis=0)

    cnt=0
    sum = 0
    avg = 0
    for i in range(0, len(histo)):
        if(histo[i] == 0):
            sum += 1
        else:
            avg += sum
            if sum > 0:
                cnt +=1
            sum = 0
    avg = avg/cnt
    l = []
    l.append(0)
    sum = 0
    for i in range(0, len(histo)):
        if(histo[i] == 0):
            sum += 1
        else:
            start = i-sum
            end = i-1
            if start <= end and sum >= avg:
                l.append(start)
                l.append(end)
            sum = 0
    l.append(len(histo)-1)

    # #smoothening histogram for better detection of words
    # y_smooth = savgol_filter(histo, 19, 3)
    #
    # # comparing histogram after smoothening
    # plt.plot(histo)
    # plt.plot(y_smooth)
    #
    # minimas = argrelextrema(y_smooth, np.less)
    # arr = minimas[0]
    # # print(arr)
    #
    # #remooving unwanted minimas
    # minn = []
    # for i in arr:
    #     if(y_smooth[i] <= 0 or histo[i] == 0):
    #         minn.append(i)
    #
    #
    #
    # # plotting minimas in histogram i.e. word breaks
    # for i in minn:
    #     plt.plot(i, y_smooth[i], 'ro')
    # plt.show()

    #creating a list of words
    words = []
    prev = -1
    for i in l:
        if(prev != -1):
            curr = img_not[: , prev:i+1]
            if(np.sum(curr) >= 6*255):
                words.append(img[:, prev:i + 1])
        prev = i

    return words

def char_seg(img):
    img_not = abs(255 - img)
    #vertical histogram
    histo = img_not.sum(axis=0)

    #smoothening histogram for better detection of words
    if(len(histo) < 7):
        # print("got it")
        l = []
        l.append(img)
        return l
    y_smooth = savgol_filter(histo, 7, 3)

    # comparing histogram after smoothening
    # plt.plot(histo)
    # plt.plot(y_smooth)

    minimas = argrelextrema(y_smooth, np.less)
    arr = minimas[0]
    # print(arr)

    #remooving unwanted minimas
    minn = []
    minn.append(0)
    for i in arr:
        if(histo[i] == 0):
            minn.append(i)
        else:
            for j in range(i-5, i+6):
                if(j < 0):
                    continue
                if(histo[j] == 0):
                    minn.append(j)
                    break

    minn.append(len(histo)-1)

    # plotting minimas in histogram i.e. word breaks
    # for i in minn:
    #     plt.plot(i, y_smooth[i], 'ro')
    # plt.show()

    #creating a list of chars
    clist = []
    prev = -1
    for i in minn:
        if(prev != -1):
            ch = img[: , prev:i+1]
            ch_not = abs(255-ch)
            if(np.sum(ch_not) >= 6*255):
                clist.append(ch)
        prev = i

    return clist