import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#For time measuring
w_start=time.time()
tf.random.set_seed(7)
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
        # ----------------------- BIEN HASTA AQUI --------------------------------
    def impute(self):
        # This function finds the missing values in the dataset
        kc=KNeighborsClassifier(n_neighbors=1,weights='uniform')
        imp = IterativeImputer(estimator=kc,max_iter=50)
        imp.fit(self.X)
        self.X = imp.transform(self.X)
    def split(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.15,shuffle=False)
    def merge(self):
        importData.impute(self)
        importData.split(self)
        return self.X_train,self.X_test,self.Y_train,self.Y_test
#My data
path ='C:/Users/Andres/Documents/Andres/Electrica/Python/Python/Datasets/Wisconsin Breast Cancer/breast-cancer-wisconsin.xlsx'
X_train,X_test,Y_train,Y_test=importData(path).merge()
m,n = X_train.shape
#Baseline: Graph of the model
def create_baseline(n,layers_dims,ol):
    #Declare the sequential model
    model = keras.Sequential(name='Mymodel1')
    #Building the structure
    model.add(layers.InputLayer(n))
    model.add(layers.Dense(layers_dims[0], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dropout(0.06))
    model.add(layers.Dense(layers_dims[1], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dense(layers_dims[2], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dense(ol, activation='sigmoid'))
    return model
#Defining Hyperparameters
layers_dims=[64,64,64]
lr = 0.0032835138849929
model = create_baseline(n,layers_dims,1)
epochs = 800
#Compile the Keras Sequential model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
#Train the model
h = model.fit(X_train,Y_train,epochs=epochs,batch_size=64,validation_data=(X_test,Y_test),verbose=1,shuffle=False)
cv_accuracy = h.history["val_binary_accuracy"][-1]
train_accuracy = h.history["binary_accuracy"][-1]
print('-----------------------------RESULTS---------------------------------\n\n\n')
print('The best cross validation accuracy is: '+str(cv_accuracy*100)+'%')
print('The training accuracy for that value is: '+str(train_accuracy*100)+'%')
model.save('keras_model.h5')
print('Model was saved succesfully!')
#Print time of execution
w_end=time.time()
tt =(w_end - w_start)/60
print('\nRunning time of the whole code is: %.2f minutes' %tt)
