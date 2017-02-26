# utility functions

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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

#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0010.png", "xxx")
#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0011.png", "xxx")
#show_image("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0012.png", "xxx")

#=============================================================================

