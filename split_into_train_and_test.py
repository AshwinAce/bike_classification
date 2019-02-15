# Usage : python split_into_train_and_test.py --dataset f1 --output f2
# f1 is the folder containing all images that will eventually become the train+ validation images
# f2 is the folder that stores the test images folder. 
from shutil import move,copy
from sklearn.model_selection import train_test_split
import argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")

ap.add_argument("-o", "--output", required=True,
	help="path to output folder")

args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images of bikes...")
fileNames=glob.glob(args["dataset"]+"**/*.jpg",recursive=True)
print(fileNames)
(train_files, test_files, _, _) = train_test_split(fileNames, fileNames, test_size=0.2)
print(test_files)

dst=args["output"]

for f in test_files:
	move(f,dst)



