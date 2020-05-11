# This code makes a MLP model to diagnose the breast cancer

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

#-----------------PREPROCESSING DATA-------------------

class importData:
    def __init__(self,path):
        cols = list(range(1,12)) # Colums of interest in the data set
        data = pd.read_excel(path,sheet_name='Hoja1',usecols=cols,na_values='?')
        data = data.values # taking values into np.matrix
        X_cols = list(range(0,9)) # X columns from data
        Y_cols = 9 # Y colums from data
        self.X = data[:,X_cols]
        self.Y = data[:,Y_cols]
        self.Y = (self.Y/2)-1 # Converting 2 to 0 and 4 to 1
        self.Y = self.Y.astype(int)      
        # Shuffle data for mini-batch convenience
        self.X,self.Y = shuffle(self.X,self.Y)
    def imput(self):
        # This function finds the missing values in the dataset
        kc=KNeighborsClassifier(n_neighbors=1,weights='uniform')
        imp = IterativeImputer(estimator=kc,max_iter=50)
        imp.fit(self.X)
        self.X = imp.transform(self.X)
    def split(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.15)
    def merge(self):
        importData.imput(self)
        importData.split(self)
        return self.X_train,self.X_test,self.Y_train,self.Y_test

#----------------------MODEL-----------------------------------

class nnmodel:
    def __init__(self,X_train,X_test,Y_train,Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    def merge(self):
        h_layers=(100,100,100)
        clf = MLPClassifier(hidden_layer_sizes=h_layers,batch_size=128,max_iter=500,verbose=True)
        clf.fit(self.X_train,self.Y_train)
        s_train=clf.score(self.X_train,Y_train)
        s_test=clf.score(self.X_test,self.Y_test)
        print('El estandar score en el training set es de: '+str(s_train*100)+'%')
        print('El score en el test set es: '+str(s_test*100)+'%')



# Defining file path
path ='C:/Users/Andres/Documents/Andres/Electrica/Python/Python/Datasets/Wisconsin Breast Cancer/breast-cancer-wisconsin.xlsx'
#Importing and preprocessing data
X_train,X_test,Y_train,Y_test=importData(path).merge()

nnmodel(X_train,X_test,Y_train,Y_test).merge()

