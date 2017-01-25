import pandas as pd
import time
import os
import numpy as np
import cv2
import sys
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


processFromScratch = True
#processFromScratch = False
from pathlib import Path

data_array_file_name = "data-array.npy"

my_file = Path(data_array_file_name)
if my_file.is_file():
    print('file exists... we remove it')
    os.remove(data_array_file_name)
    

print('Modules loaded.')
os.chdir("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/")

# ## Load the Data
#  F:\CODE\Udacity-Self-Driving-Car\Term-1\Project-3\simulator-windows-64\training2
#df = pd.read_csv('./data/data/driving_log.csv', header=0)
df = pd.read_csv('/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/data/driving_log.csv', header=0)

#F:\CODE\Udacity-Self-Driving-Car\Term-1\Project-3\simulator-windows-64\ALL 

print('csv loaded.')  

#   center,left,right,steering,throttle,brake,speed
df = df[['center', 'steering']]

print("rows {} columns {}".format(df.shape[0], df.shape[1]))

#df[['steering']].unique()

x = len(df['steering'].unique())
m = df['steering'].max()
m = df['steering'].min()
print(x)  # we have 124 levels

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

arr = list()
#arr = np.empty([1, 360, 160])

x = len(df['steering'])

start = time.time()

def cut_and_scale(img):
    img = img[60:160, 0:320]
    img = cv2.resize(img,(100, 160), interpolation = cv2.INTER_CUBIC)
#    img = cv2.blur(img,(5,5))
    return img
    



def optimize_contrast(img):
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    res = clahe.apply(img)
    res = cv2.equalizeHist(img)
    return res
    
#test
#img = cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg',0)
#imgplot = plt.imshow(img, cmap='gray')
#imgplot = plt.imshow(cut_and_scale(img), cmap='gray')    

for i in range(0, x):
#    path = './data/data/' + df['center'][i]  #this was for original
    path = df['center'][i]  #this was for original
    
    img = cv2.imread(path,0)   
    #cv2.imwrite('test-data-cut_and_scale.jpg',cut_and_scale(img))
    img = cut_and_scale(img)
    
#    img = optimize_contrast(img)
    
    if i == 1000:
        print("writing image")
        cv2.imwrite('test-camera-cut-and-scale-loading-2.jpg',img)
    img = normalize_grayscale(img)
    arr.append(img)
    

end = time.time()

elapsed = end - start
print("elapsed time (sec):")
print(elapsed)
 
myarray = np.asarray(arr) 
print(myarray.shape)
#print("exit")
#sys.exit()

myarray = np.expand_dims(myarray, 3)    
#myarray = normalize_grayscale(myarray)    
print(myarray.shape)
print("--------------- SAVE -----------------") 
np.save("data-array.npy", myarray, allow_pickle=True, fix_imports=True)
print("--------------- ALL DONE -----------------") 

