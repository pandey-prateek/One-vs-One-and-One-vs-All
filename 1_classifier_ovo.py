import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from collections import defaultdict
import sys
from sklearn.decomposition import PCA

class OneVsOne:
  def __init__(self,X,Y):
    self.X=X
    self.Y=Y
    self.classes=np.unique(Y)
    self.num_classes=len(self.classes)
    self.clfs={}

  def fit(self):

    for i in range(self.num_classes):
      for j in range(i + 1, self.num_classes):
          index = np.where((self.Y == self.classes[i])|(self.Y == self.classes[j]))
          train=self.X[index]
          
          label=self.Y[index]
          # y_binary = np.where(label == self.classes[i], 1, -1)
          clf=svm.SVC()
          clf.fit(train,label)
          self.clfs[(self.classes[i],self.classes[j])]=clf
  def predict(self,X):
    class_votes = defaultdict(int)
    y_pred=np.zeros((len(X),3))
    for pair,classifier in self.clfs.items():
      pred = classifier.predict(X)
      for i in range(len(X)):
          y_pred[i][pred[i]] += 1
          # if pred[i] == 1:
          #     y_pred[i][pred[i]] += 1
          # else:
          #     y_pred[i][pair[1]] += 1
    predictions = [_.argmax() for _ in y_pred]
    return np.array(predictions)



def preprocess(data):
  data=data.dropna()
  encoder=preprocessing.LabelEncoder()
  std=preprocessing.StandardScaler()
  data['Culmen Length (mm)']=std.fit_transform(data[['Culmen Length (mm)']])
  data['Culmen Depth (mm)']=std.fit_transform(data[['Culmen Depth (mm)']])
  data['Flipper Length (mm)']=std.fit_transform(data[['Flipper Length (mm)']])
  data['Body Mass (g)']=std.fit_transform(data[['Body Mass (g)']])
  data['Delta 15 N (o/oo)']=std.fit_transform(data[['Delta 15 N (o/oo)']])
  
  data['Delta 13 C (o/oo)']=std.fit_transform(data[['Delta 13 C (o/oo)']])
  
  data['Island']=encoder.fit_transform(data['Island'])
  data['Clutch Completion']=encoder.fit_transform(data['Clutch Completion'])
  data['Sex']=encoder.fit_transform(data['Sex'])
  #data['Species']=encoder.fit_transform(data['Species'])
  return data

data_train=pd.read_csv("penguins_train.csv")

data=preprocess(data_train)

encoder=preprocessing.LabelEncoder()
X=data.drop('Species',axis=1)
pca = PCA(n_components=7)
X=pca.fit_transform(X)
Y=encoder.fit_transform(data['Species'])

ovo=OneVsOne(X,Y)
ovo.fit()
testfilename=sys.argv[1]

print(testfilename)
data_test=pd.read_csv(testfilename)
test=preprocess(data_test)
pca = PCA(n_components=7)
test=pca.fit_transform(test)
y_pred=ovo.predict(test)

test_df=pd.DataFrame(encoder.inverse_transform(y_pred),columns=['predicted'])
test_df.to_csv("ovo.csv")
