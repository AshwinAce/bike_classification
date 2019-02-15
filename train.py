# Usage: python train.py --dataset folder_containing_train_images --model path to save model after training
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

#Needed only for two class problems
def one_hot_encoding(label):
	label = LabelBinarizer().fit_transform(label)
	return np.hstack((label,1-label))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")

ap.add_argument("-m" , "--model" ,required =False, help="path to save model")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("Loading images of bikes...")
fileNames=glob.glob(args["dataset"]+"**/*.jpg",recursive=True)

data=[]
labels=[]
#load data
for f in fileNames:
	img_temp=cv2.imread(f,1)
	img_temp=cv2.resize(img_temp,(32,32))
	data.append(img_temp)
	end=f.rfind("/")
	start=f[0:end].rfind("/")
	labels.append(f[start+1:end])
data=np.array(data)
data = data.astype("float") / 255.0
labels=np.array(labels)

# partition the data into a 70-10 training and validation split from 80% of data, 20% has been left for test data 
(trainX, validX, trainY, validY) = train_test_split(data, labels, test_size=0.125)


# convert the labels from integers to vectors, using hstack as LabelBinarizer returns 0 or 1 for a 2-class label
trainY = one_hot_encoding(trainY)
validY = one_hot_encoding(validY)

# initialize the optimizer and create model
print("Training model...")
opt = SGD(lr=0.005)
size=32
model = Sequential()
if K.image_data_format() == "channels_first":
	inputShape = (3, size, size)
else:
	inputShape = (size,size,3)

model.add(Conv2D(32 , kernel_size=(3,3), activation ='relu' , input_shape=inputShape))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Tensorboard setup
callback=TensorBoard(log_dir='./graph_train_test_separate_files',histogram_freq=0,write_graph=True,write_images=True)
# train the network
print("Training network...")
num_epochs=10
H = model.fit(trainX, trainY, validation_data=(validX, validY),
	batch_size=16, epochs=num_epochs, verbose=1, callbacks=[callback])

model.save(args["model"])


