This repository consists of four python files train.py, test.py, bike.py and split_into_train_and_test.py

split_into_train_and_test.py modifies a dataset, by moving 20% of images to a new folder.
Thus bikes_train has  80% of the images while bikes_test has 20% of the images.
train.py trains on bikes_train and saves a model [bike.hdf5]
test.py uses that model and runs on bikes_test, displaying the class as well the confidence interval on each image. A classification report is also displayed.
The tensorboard graphs for loss and accuracy are stored in the folder graph_train_test_separate_files.

bikes.py combines the functionality of train.py and test.py except for displaying the results on each image.
The corresponding tensorboard graphs are in the Graph folder.

Each file has its usage description at the top.

The links given in the question PDF file did not have a test.py in them, the links were slightly incorrect.
So I wrote test.py based on the description in the PDF.
