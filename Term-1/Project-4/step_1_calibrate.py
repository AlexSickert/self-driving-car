
# ------------------------------------------------------------------------
# function to calibrate a camera based on test images
# ------------------------------------------------------------------------   

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def calibrate(folder_calibration, test_image_path, show_images=False):
    
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob(folder_calibration + '/*.jpg')
    
    # Step through the list and search for chessboard corners
    image_number = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            
            plt.imshow(img)
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
            image_number += 1
            path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_corners/" + str(image_number) + ".jpg"
            cv2.imwrite(path, img)
            print("image saved")
            
       
    # Test undistortion on an image
    img = cv2.imread(test_image_path)
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist

    
    return dist_pickle
    
def un_distort(img, dist_pickle):  
    
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)    
    return dst
    
# --------------------
# test the function
#---------------------    

def test_calibration():
    folder_calibration = "/home/alex/Downloads/CarND-Camera-Calibration-master/calibration_wide/"
    test_image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_test/test_image.jpg"
    test_image_result_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_test/test_image_result.jpg"
    
    
    print("start calibration")
    params = calibrate(folder_calibration, test_image_path, show_images=True)
    print("calibration done")
    img = cv2.imread(test_image_path)

    print("show un-corrected version of image")
    plt.imshow(img, cmap='gray')
    plt.show()
    
    print("correct an image")
    img_corrected = un_distort(img, params)
    
    print("show corrected version of image")
    plt.imshow(img_corrected, cmap='gray')
    plt.show()
    
    print("save the image")
    cv2.imwrite(test_image_result_path, img_corrected)
    
    
