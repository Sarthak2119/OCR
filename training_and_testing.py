from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pickle

def read_data(data):
    ans=[]
    for i in data:
        d=i.split(' ')
        new_list=[]
        for j in d:
            if len(j)==0:
                continue
            # print (j)
            new_list.append(j)
        ans.append(new_list)
        # print(ans[0])
    ret = np.empty([len(ans)-1, len(ans[0])])
    i=0
    for l in ans:
        curr = np.array(l, dtype=float)
        # print(len(curr))
        # print(i)
        if(len(curr) == 0):
            continue
        ret[i] = curr
        i += 1
    return  ret
def file_read(file):
    f=open(file,'r')
    f=f.read()
    f=f.split('\n')
    f=read_data(f)
    return f

def run():
    x_train=file_read('train_input.txt')
    y_train=file_read('train_out2.txt')
    x_test=file_read('test_input.txt')
    y_test=file_read('test_out2.txt')

    # #normalization of training and text data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    x_test = scaler.transform(x_test)
    with open('scaler3.pkl','wb') as f:
        pickle.dump(scaler,f)

    #training
    print(x_train[0])
    clf = MLPClassifier(hidden_layer_sizes=(80,80))
    clf.fit(x_train, y_train)
    with open('MPLClassifier3.pkl','wb') as f:
        pickle.dump(clf,f)

    #testing
    predictions = clf.predict(x_test)
    # print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))