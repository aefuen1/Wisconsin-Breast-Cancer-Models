# This is the code used to tune the hyperparameters Learning Rate
import pandas as pd
import numpy as np
import time
from tensorflow import keras,random
from tensorflow.keras import layers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# To see how much it takes to run
random.set_seed(7)
w_start=time.time()
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
#------------------------------------Model---------------------------------------------
def create_baseline(n,layers_dims,ol,dp1,dp2):
    #Declare the sequential model
    model = keras.Sequential(name='Mymodel1')
    #Building the structure
    model.add(layers.InputLayer(n))
    model.add(layers.Dense(layers_dims[0], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dropout(dp1))
    model.add(layers.Dense(layers_dims[1], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dropout(dp2))
    model.add(layers.Dense(layers_dims[2], activation='relu', kernel_initializer=keras.initializers.he_normal()))
    model.add(layers.Dense(ol, activation='sigmoid'))
    return model
#Defining Hyperparameters
layers_dims=[64,64,64];dp1=0;dp2=0
model = create_baseline(n,layers_dims,1,dp1,dp2)
#Helper function for Learning rate tunning
def tune_lr(model,lr_iter=100):
    lr_list = []
    cv_accuracy=[]
    train_accuracy=[]
    for i in range(lr_iter):
        start=time.time()
        r = -3+1.60206*np.random.rand()
        lr= 10**r
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
        h = model.fit(X_train,Y_train,epochs=800,batch_size=64,validation_data=(X_test,Y_test),verbose=0,shuffle=False)
        cv_accuracy.append(h.history["val_binary_accuracy"][-1])
        lr_list.append(lr)
        train_accuracy.append(h.history["binary_accuracy"][-1])
        if i % 5 == 0:
            print ('Iteration number: '+str(i)+'\n')
            end = time.time()
            t = end-start
            print ('This iteration took: %.2f seconds\n' %t)
    return lr_list,cv_accuracy,train_accuracy
#Performing the model evaluation
lr_list,cv_accuracy,train_accuracy = tune_lr(model)
print('-----------------------------RESULTS---------------------------------\n\n\n')
print ('The best cross validation accuracy is: '+str(max(cv_accuracy)*100)+'%')
idx = cv_accuracy.index(max(cv_accuracy))
print ('The learning rate corresponding to that value is: '+str(lr_list[idx]))
print ('The training accuracy for that value is: '+str(train_accuracy[idx]*100)+'%')
#Save results to manually check
a = np.column_stack(([lr_list,cv_accuracy,train_accuracy]))
df = pd.DataFrame(a)
df.to_csv('fourthRun_001_04.csv',header=False,index=False)
w_end=time.time()
tt =(w_end - w_start)/60
print('\nRunning time of the whole code is: %.2f minutes' %tt)
