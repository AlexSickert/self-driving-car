import time
#from tqdm import tqdm
import os
import numpy as np
import pandas as pd
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import BaseLogger
from sklearn.utils import shuffle


save_model_and_data = True

# ====================================================
# ====================================================
# ====================================================
batch = 10
epoch = 1
validation = 0.2
#use_model_version = 3  # it was not good at all 
#use_model_version = 5  # it was not good at all had batch 20 and 3 epochs validation 0.2
#use_model_version = 1 # was not good at all wiht batch 20 and 2 epochs and validation o.1
#use_model_version = 2  # did not work at all  batch 20 and 2 epochs and validation o.1
#use_model_version = 6  # and reduced training set  batch 10 epoch 1 valid 0.1 
#use_model_version = 5  # and reduced training set  batch 10 epoch 1 valid 0.1
use_model_version = 8

# ====================================================
# ====================================================
# ====================================================

print('Modules loaded.')
#os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")
os.chdir("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/")





print("--------------- LOAD -----------------")
myarray = np.load('data-array.npy')
print("--------------- LOADED -----------------")


X_train = myarray
# TODO: Load the label data to the variable y_train
#y_train = np.asarray(str(df['steering'].values))

#df = pd.read_csv('./data/data/driving_log.csv', header=0)
#df = pd.read_csv('F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3\\simulator-windows-64\\ALL\\driving_log.csv', header=0)
df = pd.read_csv('/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/data2/driving_log.csv', header=0)

 
print('csv loaded.')  

#   center,left,right,steering,throttle,brake,speed
df = df[['center', 'steering']]

#yyy = np.asarray(df['steering'].values)

y_train = df['steering'].values.tolist()
#print(y_train)
#y_train = df['steering']

#print(type(y_train))
#print(y_train)
#exit(1)


#y_train = np.char.mod('%d', yyy)
#y_train = ["%.5f" % number for number in yyy]
#print(y_train)
#
#print(len(X_train))
#print('data loaded.')



# TODO: Shuffle the data

X_train, y_train = shuffle(X_train, y_train) 

#print(type(y_train))
#print(y_train)
#print(np.unique(y_train))
#x = len(df['steering'].unique())

# TODO: One Hot encode the labels to the variable y_one_hot
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()

print("shape of X_train")
print(X_train.shape)

#y_one_hot = label_binarizer.fit_transform(y_train)



#print(y_one_hot.shape)

print("--------------- BUILDING THE MODEL  -----------------")

# TODO: Build a model
model = None
model = Sequential()

if use_model_version == 1: 
    print("using model 1")
    #model.add(Convolution2D(124, 3, 3, input_shape=(160, 320, 1)))
    model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Dense(384))
    model.add(Dense(124))
    #model.add(Activation('relu'))
    model.add(Activation('tanh'))
    #model.add(Dense(43))
    model.add(Dense(62))
    #model.add(Activation('softmax'))
    model.add(Activation('tanh'))
    
    #model.add(Dense(31, init='normal', activation='relu'))
    model.add(Dense(1))
    
    #model.add(Dense(1, init='uniform'))
    model.add(Activation('linear'))

 
    
#softmax
#softplus
#softsign
#relu
#tanh
#sigmoid
#hard_sigmoid
#linear

if use_model_version == 2: 

    model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('sigmoid'))
    model.add(Flatten())
    model.add(Dense(124))
    model.add(Activation('sigmoid'))
    model.add(Dense(62))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))  

if use_model_version == 3: 

    model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(124))
    model.add(Activation('relu'))
    model.add(Dense(62))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))      
    
if use_model_version == 4: 

    model.add(Convolution2D(300, 4, 4, input_shape=(50, 160, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('linear'))    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))    # this cannot work unless we substract -0.5
    
if use_model_version == 5: 

    model.add(Convolution2D(300, 4, 4, input_shape=(50, 160, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('tanh'))    
    model.add(Dense(1))
    model.add(Activation('tanh'))    
    
    
if use_model_version == 6:     
    print("using model 6")
    model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
    print(model.output_shape)
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(124))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(62))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(1))
    model.add(Activation('tanh'))
    print(model.output_shape)
    
    
if use_model_version == 7:     
    print("using model 7")
    model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
    print(model.output_shape)
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    
    model.add(Convolution2D(62, 3, 3, border_mode='same'))
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    model.add(Dropout(0.5))
#    model.add(Activation('relu'))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1000))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(50))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(1))
    model.add(Activation('tanh'))
    print(model.output_shape)   
    
if use_model_version == 8:     
    print("using model 8")
    model.add(Convolution2D(64, 3, 3, input_shape=(50, 160, 1)))
    print(model.output_shape)
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    model.add(Dropout(0.3))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    model.add(Dropout(0.3))
#    model.add(Activation('relu'))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1000))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(50))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(1))
    model.add(Activation('tanh'))
    print(model.output_shape)  
    
    
if use_model_version == 9:     
    print("using model 9")
    model.add(Convolution2D(64, 3, 3, input_shape=(50, 160, 1)))
    print(model.output_shape)
    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
    
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
#    model.add(MaxPooling2D((2, 2)))
    print(model.output_shape)
#    model.add(Dropout(0.5))
#    model.add(Activation('relu'))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1000))
    model.add(Activation('sigmoid'))
    print(model.output_shape)
    model.add(Dense(50))
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(Dense(1))
    model.add(Activation('tanh'))
    print(model.output_shape)  

    
    
if save_model_and_data:

    print("--------------- Save model architecture  -----------------")
    json_string = model.to_json()
    f = open('model.json', 'w')
    f.write(json_string)
    f.close()
    
    print("--------------- TRAINING THE MODEL  -----------------")

#    model.compile(loss='mse', optimizer='rmsprop')
    model.compile(loss='mse', optimizer='adam')
#    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=batch, nb_epoch=epoch, verbose=1, callbacks=[BaseLogger()], validation_split=validation)

    print("--------------- Save weights  -----------------")
    model.save_weights('model.h5')
print("--------------- ALL DONE  -----------------")