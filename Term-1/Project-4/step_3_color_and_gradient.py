

# from 21. Sobel

# check until 30 !!!!!!!!!

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_combined_HLS_Sobel(img):

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary
    
    

## Define a function that takes an image, gradient orientation,
## and threshold min / max values.
#def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
#    # Convert to grayscale
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
##    gray = img
#    # Apply x or y gradient with the OpenCV Sobel() function
#    # and take the absolute value
#    if orient == 'x':
#        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
#    if orient == 'y':
#        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
#    # Rescale back to 8 bit integer
#    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#    # Create a copy and apply the threshold
#    binary_output = np.zeros_like(scaled_sobel)
#    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
#    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#
#    # Return the result
#    return binary_output
#    
##     from  22 mnagnitude of gradient
#
## Define a function to return the magnitude of the gradient
## for a given sobel kernel size and threshold values
#def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
#    # Convert to grayscale
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
#    # Take both Sobel x and y gradients
#    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#    # Calculate the gradient magnitude
#    gradmag = np.sqrt(sobelx**2 + sobely**2)
#    # Rescale to 8 bit
#    scale_factor = np.max(gradmag)/255 
#    gradmag = (gradmag/scale_factor).astype(np.uint8) 
#    # Create a binary image of ones where threshold is met, zeros otherwise
#    binary_output = np.zeros_like(gradmag)
#    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
#
#    # Return the binary image
#    return binary_output
#
#    
#def get_s_binary(img):
#    
#    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
#
#    
#    image = mpimg.imread('test6.jpg')
#    
#
#    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#    H = hls[:,:,0]
#    L = hls[:,:,1]
#    S = hls[:,:,2]
#
#    thresh = (90, 255)
#    binary = np.zeros_like(S)
#    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
#    

    
#def test():
#    
#    img_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg"
#   
#    # load image
#    img = cv2.imread(img_path)
##    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#    
#    plt.imshow(img, cmap='gray')
#    plt.show()
#    
#    binary = abs_sobel_thresh(img, orient='x', thresh_min=45, thresh_max=150)
#    
#    plt.imshow(binary, cmap='gray')
#    plt.show()
#    
#    binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(45, 150))
#        
#    plt.imshow(binary, cmap='gray')
#    plt.show()
  
    
def test():
    
    img_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg"
   
    # load image
    img = cv2.imread(img_path)
    
    plt.imshow(img)
    plt.show()

    combined_binary = get_combined_HLS_Sobel(img)
    
    plt.imshow(combined_binary, cmap='gray')
    plt.show()
            
#test()
    