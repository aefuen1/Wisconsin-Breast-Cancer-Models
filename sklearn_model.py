import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
import joblib

#-----------------PREPROCESSING DATA-------------------
np.random.seed(2015)

class importData:
    def __init__(self,path):
        cols = list(range(1,12)) # Colums of interest in the data set
        data = pd.read_excel(path,sheet_name='Hoja1',usecols=cols,na_values='?')
        data = data.values # taking values into np.matrix
        X_cols = list(range(9)) # X columns from data
        Y_cols = 9 # Y colums from data
        self.X = data[:,X_cols]
        self.Y = data[:,Y_cols]
        self.Y = (self.Y/2)-1 # Converting 2 to 0 and 4 to 1
        self.Y = self.Y.astype(int)
        # Shuffle data for mini-batch convenience
        self.X,self.Y = shuffle(self.X,self.Y,random_state=0)
    def impute(self):
        # This function finds the missing values in the dataset
        kc=KNeighborsClassifier(n_neighbors=1,weights='uniform')
        imp = IterativeImputer(estimator=kc,max_iter=50)
        imp.fit(self.X)
        self.X = imp.transform(self.X)
    def split(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.10,shuffle=False)
    def merge(self):
        importData.impute(self)
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
        h_layers=(64,64,64)
        clf = MLPClassifier(hidden_layer_sizes=h_layers,batch_size=64,max_iter=500,shuffle=False,verbose=False)
        LR=[]
        for i in range(50):
            r = -3+1.67*np.random.rand()
            lr= 10**r
            LR.append(lr)
        parameters = {'learning_rate_init': LR}
        gsc = GridSearchCV (clf,parameters)
        grid_result = gsc.fit(self.X_train,self.Y_train)
        best_params = grid_result.best_params_
        train=gsc.score(self.X_train,self.Y_train)
        test=gsc.score(self.X_test,self.Y_test)
        print("The best learning rate is: "+str(best_params["learning_rate_init"]))
        print("The accuracy in training set is: "+str(train)+" %")
        print("The accuracy in dev set is: "+str(test)+" %\n\n")
        clf = MLPClassifier(hidden_layer_sizes=h_layers,learning_rate_init=best_params["learning_rate_init"],batch_size=64,max_iter=1000,shuffle=False,verbose=False)
        clf.fit(self.X_train,self.Y_train)
        print("FINAL MODEL TRAIN ACCURACY: "+str(str(clf.score(self.X_train,self.Y_train))))
        print("FINAL MODEL CV ACCURACY: "+str(clf.score(self.X_test,self.Y_test)))
        file = "model_sklearn.sav"
        joblib.dump(clf,file)

# Defining file path
path ='C:/Users/Andres/Documents/Andres/Electrica/Python/Python/Datasets/Wisconsin Breast Cancer/breast-cancer-wisconsin.xlsx'
#Importing and preprocessing data
X_train,X_test,Y_train,Y_test=importData(path).merge()
nnmodel(X_train,X_test,Y_train,Y_test).merge()
