import numpy as np
import cv2
import matplotlib.pyplot as plt


def to_geyscale(img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def warp(image, warp_parameter):
        
    img_size = (image.shape[1], image.shape[0])    
    warped = cv2.warpPerspective(image, warp_parameter, img_size)
    return warped
    
def get_warp_params():

    src = np.float32([[535,492],[751,492],[265,677],[1039,677]])    
#    src = np.float32([[605,443],[671,442],[265,677],[1039,677]])    
#    dest = np.float32([[265,443],[1039,442],[265,677],[1039,677]])  
    dest = np.float32([[265,492],[1039,492],[265,677],[1039,677]])
    M = cv2.getPerspectiveTransform(src, dest)
    return M
    
def get_unwarp_params():

    src = np.float32([[535,492],[751,492],[265,677],[1039,677]])    
    dest = np.float32([[265,492],[1039,492],[265,677],[1039,677]])    
    M = cv2.getPerspectiveTransform(dest, src)
    return M    
    


def test():
    
    img_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/straight_lines1.jpg"
    warp_parameter = get_warp_params()
    # load image
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.show()
    img = warp(img, warp_parameter)
    plt.imshow(img)
    plt.show()
    warp_parameter = get_unwarp_params()
    img = warp(img, warp_parameter)
    plt.imshow(img)
    plt.show()

    
#test()    
    
#Examples of Useful Code
#
#Compute the perspective transform, M, given source and destination points:
#
#M = cv2.getPerspectiveTransform(src, dst)
#Compute the inverse perspective transform:
#
#Minv = cv2.getPerspectiveTransform(dst, src)
#Warp an image using the perspective transform, M:
#
#warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
#
#Note: When you apply a perspective transform, choosing four source points manually,
# as we did in this video, is often not the best option. There are many other ways
# to select source points. For example, many perspective transform algorithms will
# programmatically detect four source points in an image based on edge or corner 
# detection and analyzing attributes like color and surrounding pixels.
#

#
## Define a function that takes an image, number of x and y points, 
## camera matrix and distortion coefficients
#def corners_unwarp(img, nx, ny, mtx, dist):
#    # Use the OpenCV undistort() function to remove distortion
#    undist = cv2.undistort(img, mtx, dist, None, mtx)
#    # Convert undistorted image to grayscale
#    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
#    # Search for corners in the grayscaled image
#    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#    if ret == True:
#        # If we found corners, draw them! (just for fun)
#        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
#        # Choose offset from image corners to plot detected corners
#        # This should be chosen to present the result at the proper aspect ratio
#        # My choice of 100 pixels is not exact, but close enough for our purpose here
#        offset = 100 # offset for dst points
#        # Grab the image shape
#        img_size = (gray.shape[1], gray.shape[0])
#
#        # For source points I'm grabbing the outer four detected corners
#        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#        # For destination points, I'm arbitrarily choosing some points to be
#        # a nice fit for displaying our warped result 
#        # again, not exact, but close enough for our purposes
#        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
#                                     [img_size[0]-offset, img_size[1]-offset], 
#                                     [offset, img_size[1]-offset]])
#        # Given src and dst points, calculate the perspective transform matrix
#        M = cv2.getPerspectiveTransform(src, dst)
#        # Warp the image using OpenCV warpPerspective()
#        warped = cv2.warpPerspective(undist, M, img_size)
#
#    # Return the resulting image and matrix
#    return warped, M