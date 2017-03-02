# utility functions

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pylab as pl

#=============================================================================

def show_image(file_path, text):
    
    img = mpimg.imread(file_path)  
    plt.imshow(img)  
    plt.title(text)
    plt.show()
    
def show_image_from_image(image, text):
    
    plt.imshow(image)  
    plt.title(text)
    plt.show()    
    
#=============================================================================
def save_image_debug(img, debug):
    
    path_full = debug['path'] + str(debug['counter']) + debug['id'] +  debug['ending']
    plt.imshow(img)  
    plt.title(debug['text'])
    pl.savefig(path_full)    
    
#=============================================================================

def save_image(img, path):
    cv2.imwrite(path,img)
#=============================================================================
    
def get_array_shape(array):
    
    x = np.array(array)
    print(x.shape)
#=============================================================================

#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0010.png", "xxx")
#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0011.png", "xxx")
#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0012.png", "xxx")

#=============================================================================

