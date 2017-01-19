import pandas as pd
import time
#from tqdm import tqdm
import os
import pickle
import numpy as np
import math
import sys
import cv2
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf
import matplotlib.pyplot as plt

processFromScratch = True
#processFromScratch = False


print('Modules loaded.')
os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")

# ## Load the Data
df = pd.read_csv('./data/data/driving_log.csv', header=0)
 
print('csv loaded.')  

#   center,left,right,steering,throttle,brake,speed
df = df[['center', 'steering']]

print("rows {} columns {}".format(df.shape[0], df.shape[1]))

#df[['steering']].unique()

x = len(df['steering'].unique())
m = df['steering'].max()
m = df['steering'].min()
print(x)  # we have 124 levels

    # ok, we see that the range is from -1 to 1
    
#    img = cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg',0)
    
    #flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    #print(flags)
    
#    imgplot = plt.imshow(img, cmap='gray')
    
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )



if processFromScratch:    
    
    arr = list()
    #arr = np.empty([1, 360, 160])
    
    x = len(df['steering'])
    #x = 2000
    start = time.time()
    for i in range(0, x):
    #    print(i)
        path = './data/data/' + df['center'][i]
    #    print(path)
        img = cv2.imread(path,0)
    #    print(img.shape)
        arr.append(img)
    #    np.append(arr, cv2.imread(path,0), axis=1)
    
    end = time.time()
    elapsed = end - start
    print("elapsed time (sec):")
    print(elapsed)
    
    #print("--------------------------------")  
    #print(len(arr))
    #print("--------------------------------")  
    myarray = np.asarray(arr)
    #print("--------------------------------")  
    ##print(myarray[2])
    #print("--------------------------------")  
    ##print(type(myarray))
    #print("--------------------------------")  
    ##print(myarray.shape)
    #print("--------------------------------") 
    myarray = np.expand_dims(myarray, 3)
    
    myarray = normalize_grayscale(myarray)
    
    print(myarray.shape)
    print("--------------- SAVE -----------------") 
#    os.remove("data-array.npy")
    #np.save("data-array.npy", myarray, allow_pickle=True, fix_imports=True)

else:

    print("--------------- LOAD -----------------")
    #myarray = np.load('data-array.npy')
    print("--------------- LOADED -----------------")
    
print("--------------- READY TO GO  -----------------")
#with open('train.p', 'rb') as f:
#    data = pickle.load(f)

# load image as array

#X_train = data['features']
X_train = myarray
# TODO: Load the label data to the variable y_train
#y_train = np.asarray(str(df['steering'].values))

yyy = np.asarray(df['steering'].values)

#y_train = np.char.mod('%d', yyy)
y_train = ["%.5f" % number for number in yyy]
print(y_train)

print(len(X_train))
print('data loaded.')

#sys.exit("-------------- execution aborted -----------------") 

# STOP: Do not change the tests below. Your implementation should pass these tests. 
#assert np.array_equal(X_train, data['features']), 'X_train not set to data[\'features\'].'
#assert np.array_equal(y_train, data['labels']), 'y_train not set to data[\'labels\'].'
#print('Tests passed.')

# TODO: Shuffle the data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train) 

#print('suffled')
#sys.exit("-------------- execution aborted -----------------") 

# STOP: Do not change the tests below. Your implementation should pass these tests. 
#assert X_train.shape == data['features'].shape, 'X_train has changed shape. The shape shouldn\'t change when shuffling.'
#assert y_train.shape == data['labels'].shape, 'y_train has changed shape. The shape shouldn\'t change when shuffling.'
#assert not np.array_equal(X_train, data['features']), 'X_train not shuffled.'
#assert not np.array_equal(y_train, data['labels']), 'y_train not shuffled.'
#print('Tests passed.')


# ### Normalize the features
# Hint: You solved this in [TensorFlow lab](https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/lab.ipynb) Problem 1.


# TODO: Normalize the data features to the variable X_normalized


#print('normalized')
#sys.exit("-------------- execution aborted -----------------") 

# STOP: Do not change the tests below. Your implementation should pass these tests. 
#assert math.isclose(np.min(X_normalized), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_normalized), 0.5, abs_tol=1e-5), 'The range of the training data is: {} to {}.  It must be -0.5 to 0.5'.format(np.min(X_normalized), np.max(X_normalized))
#print('Tests passed.')

# ### One-Hot Encode the labels
# Hint: You can use the [scikit-learn LabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) function to one-hot encode the labels.

print(type(y_train))
print(y_train)
print(np.unique(y_train))
#x = len(df['steering'].unique())

# TODO: One Hot encode the labels to the variable y_one_hot
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()

y_one_hot = label_binarizer.fit_transform(y_train)

import collections

print(y_one_hot.shape)
#assert y_one_hot.shape == (8036, 124), 'y_one_hot is not the correct shape.  It\'s {}, it should be (8036, 124)'.format(y_one_hot.shape)
#assert next((False for y in y_one_hot if collections.Counter(y) != {0: 42, 1: 1}), True), 'y_one_hot not one-hot encoded.'
#print('Tests passed.')


# STOP: Do not change the tests below. Your implementation should pass these tests. 
#import collections
#assert y_one_hot.shape == (39209, 43), 'y_one_hot is not the correct shape.  It\'s {}, it should be (39209, 43)'.format(y_one_hot.shape)
#assert next((False for y in y_one_hot if collections.Counter(y) != {0: 42, 1: 1}), True), 'y_one_hot not one-hot encoded.'
#print('Tests passed.')
#
## set up model
#from keras.models import Sequential
#model = Sequential()
## TODO: Build a Multi-layer feedforward neural network with Keras here.
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten
#
#model.add(Flatten(input_shape=(32, 32, 3)))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dense(43))
#model.add(Activation('softmax'))
#
## STOP: Do not change the tests below. Your implementation should pass these tests.
#from keras.layers.core import Dense, Activation, Flatten
#from keras.activations import relu, softmax
#
#def check_layers(layers, true_layers):
#    assert len(true_layers) != 0, 'No layers found'
#    for layer_i in range(len(layers)):
#        assert isinstance(true_layers[layer_i], layers[layer_i]), 'Layer {} is not a {} layer'.format(layer_i+1, layers[layer_i].__name__)
#    assert len(true_layers) == len(layers), '{} layers found, should be {} layers'.format(len(true_layers), len(layers))
#
#check_layers([Flatten, Dense, Activation, Dense, Activation], model.layers)
#
#assert model.layers[0].input_shape == (None, 32, 32, 3), 'First layer input shape is wrong, it should be (32, 32, 3)'
#assert model.layers[1].output_shape == (None, 128), 'Second layer output is wrong, it should be (128)'
#assert model.layers[2].activation == relu, 'Third layer not a relu activation layer'
#assert model.layers[3].output_shape == (None, 43), 'Fourth layer output is wrong, it should be (43)'
#assert model.layers[4].activation == softmax, 'Fifth layer not a softmax activation layer'
#print('Tests passed.')



## TODO: Compile and train the model here.
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

#
## STOP: Do not change the tests below. Your implementation should pass these tests.
#from keras.optimizers import Adam
#
#assert model.loss == 'categorical_crossentropy', 'Not using categorical_crossentropy loss function'
#assert isinstance(model.optimizer, Adam), 'Not using adam optimizer'
#assert len(history.history['acc']) == 10, 'You\'re using {} epochs when you need to use 10 epochs.'.format(len(history.history['acc']))
#
#assert history.history['acc'][-1] > 0.92, 'The training accuracy was: %.3f. It shoud be greater than 0.92' % history.history['acc'][-1]
#assert history.history['val_acc'][-1] > 0.85, 'The validation accuracy is: %.3f. It shoud be greater than 0.85' % history.history['val_acc'][-1]
#print('Tests passed.')
#

#
## STOP: Do not change the tests below. Your implementation should pass these tests.
#from keras.layers.core import Dense, Activation, Flatten
#from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
#
#check_layers([Convolution2D, MaxPooling2D, Activation, Flatten, Dense, Activation, Dense, Activation], model.layers)
#assert model.layers[1].pool_size == (2, 2), 'Second layer must be a max pool layer with pool size of 2x2'
#
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=2, validation_split=0.2)
#assert(history.history['val_acc'][-1] > 0.91), "The validation accuracy is: %.3f.  It should be greater than 0.91" % history.history['val_acc'][-1]
#print('Tests passed.')
#

# ## Optimization

print("--------------- BUILDING THE MODEL  -----------------")

# TODO: Build a model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
#model.add(Convolution2D(360, 160, 1, input_shape=(360, 160, 1)))
#model.add(Convolution2D(160, 320, 1, input_shape=(160, 320, 1)))

model.add(Convolution2D(124, 3, 3, input_shape=(160, 320, 1)))

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

print("done.")

print("--------------- TRAINING THE MODEL  -----------------")
# TODO: Compile and train the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

history = model.fit(X_train, y_one_hot, batch_size=20, nb_epoch=3, validation_split=0.2)


#
## TODO: Load test data
#with open('test.p', 'rb') as f:
#    data_test = pickle.load(f)
#
#X_test = data_test['features']
#y_test = data_test['labels']
#
## TODO: Preprocess data & one-hot encode the labels
#X_normalized_test = normalize_grayscale(X_test)
#y_one_hot_test = label_binarizer.fit_transform(y_test)
#
## TODO: Evaluate model on test data
#metrics = model.evaluate(X_normalized_test, y_one_hot_test)
#for metric_i in range(len(model.metrics_names)):
#    metric_name = model.metrics_names[metric_i]
#    metric_value = metrics[metric_i]
#    print('{}: {}'.format(metric_name, metric_value))
