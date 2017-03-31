import cv2
import template
import dataset_preprocess
import  out_vecorize
import training_and_testing
import datetime
from features import get_data
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle


img = cv2.imread('/home/sarthak/ip/fIMage.jpg', 0)
#img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)

with open('MPLClassifier.pkl','rb')as f:
	clf = pickle.load(f)
with open('scaler.pkl','rb')as f:
	scaler= pickle.load(f)
list_chars =template.run(img)
x=""
for word_list in list_chars:
	print ("1")
	for char_img in word_list:
		datas = get_data(char_img)

		#print(datas)
		datas=scaler.transform(datas)
		#print(datas)
		out_vec=clf.predict(datas)
		cnt=0
		#print (out_vec)
		for i in out_vec[0]:
			cnt = cnt+1
			if i==1:
				break
			else :
				continue
		val =""
		#print (cnt)
		if cnt<11:
			cnt=cnt-1
			val  = chr(48+cnt)
		elif cnt>10 and cnt<37:
			cnt=cnt-11
			val = chr(65+cnt)
		else:
			cnt-=37
			val=chr(97+cnt)
		x=x+val
		print (val)
	x=x+" "

print (x)
	



