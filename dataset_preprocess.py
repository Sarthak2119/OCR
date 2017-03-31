import cv2
import os
from features import get_data

def run():
    f = 0
    test_data = []
    output = []
    train_data = []
    predicated_output = []
    x = 0
    y = 0
    cwd = os.getcwd()
    train_in = 'train_input3.txt'
    test_in = 'test_input3.txt'
    train_out = 'train_out3.txt'
    test_out = 'test_out3.txt'
    train_in = open(train_in, 'w')
    train_out = open(train_out, 'w')
    test_in = open(test_in, 'w')
    test_out = open(test_out, 'w')
    path = cwd+'/Fnt3/'
    for files in os.listdir(path):
        character_value = int((files[6:len(files)]))
        # print (files)
        #cnt += 1
        # print (character_value)
        real_value = 0
        # print(files)
        if character_value > 0 and character_value < 11:
            real_value = str(character_value)
        elif character_value > 10 and character_value < 37:
            character_value = character_value - 10
            real_value = chr(65 + character_value - 1)
        else:
            character_value = character_value - 36
            real_value = chr(97 + character_value - 1)
        # print(real_value)

        path2 = os.listdir(path  + files+'/')
        y = (len(path2))
        #print (path2)
        cnt = 0
        boudary = (y * 90) / 100
        for imges in path2:

            val = path+files+'/' + imges
            #print(val)
            img = cv2.imread(val, 0)
            #print  (img)
            # print(img)
            cnt = cnt + 1
            new_list = get_data(img)
            # print (new_list)
            if cnt > boudary:
                test_data.append(new_list)
                x = x + 1
                output.append(real_value)
                for i in new_list:
                    test_in.write(str(i))
                    test_in.write(" ")

                test_in.write('\n')
                test_out.write(str(real_value))
                test_out.write('\n')
            else:
                train_data.append(new_list)
                y = y + 1
                predicated_output.append(real_value)
                for i in new_list:
                    train_in.write(str(i))
                    train_in.write(" ")
                # train_in.write(str(new_list))
                train_in.write('\n')
                train_out.write(str(real_value))
                train_out.write('\n')

    train_out.close()
    train_in.close()
    test_out.close()
    test_in.close()

