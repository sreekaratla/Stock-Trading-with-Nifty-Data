# Stock-Trading-with-Nifty-Data

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse
import os

# which help us to parse the argument of as model name
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
                help="various type of machine learning model, which used")
args = vars(ap.parse_args())
used_model = {
	"knn": KNeighborsClassifier(n_neighbors=4)
}
# declearation of variables

data = []
labels = []
print("start reading the dataset")
# Now let's open the data file
import csv
with open('NIFTY_50.csv','rt')as f:
  data1 = csv.reader(f)
  for row in data1:
        data.append(row)
print(*data)
