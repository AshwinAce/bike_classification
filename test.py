# Usage: python test.py --dataset test_images_folder --model trained_model.hdf5
# In my case, the model is bike.hdf5
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2

#Needed only for two class problems
def one_hot_encoding(label):
	label = LabelBinarizer().fit_transform(label)
	return np.hstack((label,1-label))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to test dataset")

ap.add_argument("-m" , "--model" ,required =True, help="path to load model")
args = vars(ap.parse_args())

# get the list of test images
print("Loading test images of bikes...")
fileNames=glob.glob(args["dataset"]+"**/*.jpg",recursive=True)

# load the saved model
print("Loading saved network...")
model = load_model(args["model"])

classes=["road_bike", "mountain_bike"]
size=32
testX=[]
testY=[]
#load test data and evaluate and display on each image
for f in fileNames:
	print(f)
	img=cv2.imread(f,1)
	input_to_cnn=cv2.resize(img,(size,size))
	input_to_cnn=input_to_cnn.astype("float") / 255.0
	testX.append(input_to_cnn)
	input_to_cnn=input_to_cnn.reshape(1,size,size,3)
	
	end=f.rfind("_")
	start=f.rfind("/")
	original_class=f[start+1:end]
	testY.append(original_class)
	prediction=model.predict(input_to_cnn)
	#Red if wrong, green if correct
	color=(0,0,255)
	if classes[prediction.argmax()]==original_class:
		color=(0,255,0)
	cv2.putText(img, classes[prediction.argmax()]+ " Confidence : "+str(np.max(prediction)), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
	cv2.imshow("IMG",img)
	cv2.waitKey(1000)

cv2.destroyAllWindows()

#show precison,recall and f1 score for both classes
testX=np.array(testX)
testY=np.array(testY)
testY = one_hot_encoding(testY)

# evaluate the network
print("Evaluating network...")
predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=classes))


