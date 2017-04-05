import cv2
import template
import dataset_preprocess
import out_vecorize
import training_and_testing
import datetime
from features import get_data
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
import utility as util

def image_to_text(str):
    img = cv2.imread(str, 0)
    # img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)

    with open('KNNClassifier.pkl', 'rb') as f:
        clf2 = pickle.load(f)
    with open('ExtraTreesClassifier.pkl', 'rb') as f:
        clf1 = pickle.load(f)
    with open('MPLClassifier4.pkl', 'rb') as f:
        clf3 = pickle.load(f)
    with open('scaler4.pkl', 'rb') as f:
        scaler = pickle.load(f)
    list_chars = template.run(img)

    # print(len(list_chars))
    # print(len(list_chars[0]))

    ret_str = ""
    for word_list in list_chars:
        chars = []
        for char_img in word_list:
            datas = get_data(char_img)
            # util.display_image(char_img)
            chars.append(datas)

        chars = scaler.transform(chars)
        out_vec1 = clf1.predict(chars)
        out_vec2 = clf2.predict(chars)
        out_vec3 = clf3.predict(chars)

        x1 = ""
        for vec in out_vec1:
            cnt = 0
            for i in vec:
                cnt = cnt + 1
                if i == 1:
                    break
            val = ""
            if cnt < 11:
                cnt = cnt - 1
                val = chr(48 + cnt)
            elif cnt > 10 and cnt < 37:
                cnt = cnt - 11
                val = chr(65 + cnt)
            else:
                cnt -= 37
                val = chr(97 + cnt)
            x1 = x1 + val

        x2 = ""
        for vec in out_vec2:
            cnt = 0
            for i in vec:
                cnt = cnt + 1
                if i == 1:
                    break
            val = ""
            if cnt < 11:
                cnt = cnt - 1
                val = chr(48 + cnt)
            elif cnt > 10 and cnt < 37:
                cnt = cnt - 11
                val = chr(65 + cnt)
            else:
                cnt -= 37
                val = chr(97 + cnt)
            x2 = x2 + val

        x3 = ""
        for vec in out_vec3:
            cnt = 0
            for i in vec:
                cnt = cnt + 1
                if i == 1:
                    break
            val = ""
            if cnt < 11:
                cnt = cnt - 1
                val = chr(48 + cnt)
            elif cnt > 10 and cnt < 37:
                cnt = cnt - 11
                val = chr(65 + cnt)
            else:
                cnt -= 37
                val = chr(97 + cnt)
            x3 = x3 + val

        finalx =""
        for i in range(0, len(x1)):
            l = []
            if x1[i] != 'z':
                l.append(x1[i])
            if x2[i] != 'z':
                l.append(x2[i])
            if x3[i] != 'z':
                l.append(x3[i])
            if len(l) == 0:
                finalx += 'z'
            else:
                finalx += l[0]
        ret_str += finalx
        ret_str += " "
    return  ret_str