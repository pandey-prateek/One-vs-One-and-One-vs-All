import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sys
from sklearn.decomposition import PCA

class OneVsRest:
  def __init__(self,X,Y):
    self.X=X
    self.Y=Y
    self.classes=np.unique(Y)
    self.num_classes=len(self.classes)
    self.clfs=[]

  def fit(self):

    for i in self.classes:
      label = (self.Y == i).astype(int)
      clf=svm.SVC(probability=True)
      clf.fit(self.X,label)
      self.clfs.append(clf)
  def predict(self,test):
    y_pred = np.zeros((test.shape[0], self.num_classes))
    y=[]
    for i,classifier in enumerate(self.clfs):
      y_pred[:, i] = classifier.predict_proba(test)[:,1]
    return np.argmax(y_pred, axis=1)




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

ovr=OneVsRest(X,Y)
ovr.fit()
testfilename=sys.argv[1]

print(testfilename)
data_test=pd.read_csv(testfilename)
test=preprocess(data_test)
pca = PCA(n_components=7)
test=pca.fit_transform(test)
y_pred=ovr.predict(test)

test_df=pd.DataFrame(encoder.inverse_transform(y_pred),columns=['predicted'])
test_df.to_csv("ova.csv")
