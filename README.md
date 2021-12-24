# Stock-Trading-with-Nifty-Data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import argparse
import os
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


#first we have to construct the argument parser
#which help us to parse the argument of as model name
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
                help="various type of machine learning model, which used")
args = vars(ap.parse_args())
allmodels = {
    "knn": KNeighborsClassifier(n_neighbors=4),
    "random_forest": RandomForestClassifier(n_estimators=300)
}
#declearation of variables

data = []
labels = []
print("start reading the dataset")
#Now let's open the data file
import csv
with open('NIFTY_50.csv','rt')as f:
  data1 = csv.reader(f)
  for row in data1:
        data.append(row)
#print(*data)
#preprocessing of data

newtotaldata = []
for i in range(1, len(data)):
  d=[]
  for j in range(1, len(data[i])):
      if j == 9:
        if float(data[i][j]) < 0.8 :
            labels.append("1")
        elif float(data[i][j]) < 1 :
            labels.append("2")
        elif float(data[i][j]) < 1.2 :
            labels.append("3")
        else:
            labels.append("4")
      else:
          data[i][j] = float(data[i][j])
          d.append(data[i][j])
  newtotaldata.append(d)

le = LabelEncoder()
labels = le.fit_transform(labels)
#define the  training  and testing dataset with the actual dataset with 10% of testing and  90% for training
(trainX, testX, trainY, testY) = train_test_split(newtotaldata, labels,test_size=0.1)
#Now lets train the model
print("training the model using '{}' model".format(args["model"]))
model = allmodels[args["model"]]
model.fit(trainX, trainY)
#report of classification
print("classification report")
predictions = model.predict(testX)
#plot the confusion matrix and then print classification report 
ConfusionMatrixDisplay.from_estimator(model, testX, testY)
plt.show()
print(classification_report(testY, predictions,
                          target_names=le.classes_))

