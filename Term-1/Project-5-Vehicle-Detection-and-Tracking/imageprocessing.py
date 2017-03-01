import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.feature import hog
#import training as trn
#import image_processing as proc



#==============================================================================
#  convert the image into a feature set

def image_to_featureset(image, color_space, s, hog_channel):
    
    image = cv2.resize(image, (s, s))
    
    features_spatial = bin_spatial(image, color_space, size=(s, s))
    
#    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
#    features_hog, hog_image = get_hog_features(gray, orient, 
#                        pix_per_cell, cell_per_block, 
#                        vis=True, feature_vec=False)
    
    
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(get_hog_features(image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    
    
#    get_array_shape(features_spatial)
#    get_array_shape(hog_features)
    
#    plt.imshow(hog_image)
#    plt.show() 
        
#    all_features = np.concatenate((features_spatial, hog_features))
    all_features = hog_features
    
    return all_features
    
#=============================================================================



#============================================================================= 
    
# convert image to color space and resize image    
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features    


# =============================================================================
# color histograms

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

        
#=============================================================================

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#=============================================================================

def draw_labeled_bboxes(img, heatmap):
    
    labels = label(heatmap)
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img

#=============================================================================
#
## Generate a random index to look at a car image
#ind = np.random.randint(0, len(cars))
## Read in the image
#image = mpimg.imread(cars[ind])
#
#gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
## Define HOG parameters
#orient = 9
#pix_per_cell = 8
#cell_per_block = 2
## Call our function with vis=True to see an image output
#features, hog_image = get_hog_features(gray, orient, 
#                        pix_per_cell, cell_per_block, 
#                        vis=True, feature_vec=False)
##
#
## Plot the examples
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(image, cmap='gray')
#plt.title('Example Car Image')
#plt.subplot(122)
#plt.imshow(hog_image, cmap='gray')
#plt.title('HOG Visualization')

# =============================================================================

## Define a function to return HOG features and visualization
#def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
#    if vis == True:
#        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
#                                  visualise=True, feature_vector=False)
#        return features, hog_image
#    else:      
#        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
#                       visualise=False, feature_vector=feature_vec)
#        return features




