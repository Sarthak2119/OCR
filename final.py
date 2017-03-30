import dataset_preprocess
import  out_vecorize
import training_and_testing
import datetime

dataset_preprocess.run()
print ("Preprocess compplete")
print (datetime.datetime.now())
out_vecorize.run()
print ("output vectorize")
print (datetime.datetime.now())
training_and_testing.run()
print ("Training and testing")
print (datetime.datetime.now())