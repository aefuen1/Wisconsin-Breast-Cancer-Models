import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

#My data
path ='C:/Users/Andres/Documents/Andres/Electrica/Python/Python/Datasets/Wisconsin Breast Cancer/breast-cancer-wisconsin.xlsx'
X_train,X_test,Y_train,Y_test=importData(path).merge()
m,n = X_train.shape

# My model
model = keras.Sequential(name='Mymodel1')
model.add(layers.InputLayer(n))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam')

model.fit(X_train,Y_train,epochs=2000)
