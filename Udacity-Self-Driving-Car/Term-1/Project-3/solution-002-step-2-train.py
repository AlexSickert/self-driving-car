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

save_model_and_data = True

print('Modules loaded.')
os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")

# ====================================================
# ====================================================
# ====================================================
batch = 10
epoch = 1
validation = 0.1
# ====================================================
# ====================================================
# ====================================================


print("--------------- LOAD -----------------")
myarray = np.load('data-array.npy')
print("--------------- LOADED -----------------")


X_train = myarray
# TODO: Load the label data to the variable y_train
#y_train = np.asarray(str(df['steering'].values))

df = pd.read_csv('./data/data/driving_log.csv', header=0)
 
print('csv loaded.')  

#   center,left,right,steering,throttle,brake,speed
df = df[['center', 'steering']]

yyy = np.asarray(df['steering'].values)

#y_train = np.char.mod('%d', yyy)
y_train = ["%.5f" % number for number in yyy]
#print(y_train)
#
#print(len(X_train))
#print('data loaded.')



# TODO: Shuffle the data
from sklearn.utils import shuffle
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

y_one_hot = label_binarizer.fit_transform(y_train)

import collections

print(y_one_hot.shape)

print("--------------- BUILDING THE MODEL  -----------------")

# TODO: Build a model

model = Sequential()
#model.add(Convolution2D(124, 3, 3, input_shape=(160, 320, 1)))
model.add(Convolution2D(124, 3, 3, input_shape=(50, 160, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
#model.add(Dense(128))
#model.add(Dense(384))
model.add(Dense(124))
model.add(Activation('relu'))
#model.add(Dense(43))
model.add(Dense(124))
model.add(Activation('softmax'))

#model.save('my_model.h5')

if save_model_and_data:

    print("--------------- Save model architecture  -----------------")
    json_string = model.to_json()
    f = open('model.json', 'w')
    f.write(json_string)
    f.close()
    
    print("--------------- TRAINING THE MODEL  -----------------")
    # TODO: Compile and train the model
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    #history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)
    
    #history = model.fit(X_train, y_one_hot, batch_size=20, nb_epoch=3, validation_split=0.2)
    history = model.fit(X_train, y_one_hot, batch_size=batch, nb_epoch=epoch, validation_split=validation)
    
    print("--------------- Save weights  -----------------")
    model.save_weights('model.h5')
print("--------------- ALL DONE  -----------------")