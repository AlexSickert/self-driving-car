
# -------------------------------------------------------------
#   This file tests the entire pipeline, meaning: 
#   1. calibrate
#   2. test distortion correction
#   3. step 3 - 6 on an image
#   4. run step 3 to 6 on a video
# -------------------------------------------------------------


import os
import step_1_calibrate as cal
import step_2_distortion_correction as dis
import matplotlib.pyplot as plt
import cv2

working_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/"
os.chdir(working_dir)    

# run all on one image

# calibrate

image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/test2.jpg"

folder_calibration = "/home/alex/Downloads/CarND-Camera-Calibration-master/calibration_wide/"
test_image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_test/test_image.jpg"

print("start calibration")
params = cal.calibrate(folder_calibration, test_image_path)
print("calibration done")
img = cv2.imread(image_path)

print("show un-corrected version of image")
plt.imshow(img)
plt.show()

# undistort
print("correct an image")
img_corrected = cal.un_distort(img, params)

print("show corrected version of image")
plt.imshow(img_corrected)
plt.show()

# perspective transform

warp_parameter = dis.get_warp_params()
img = to_geyscale(img_corrected)
img = dis.warp(img, warp_parameter)
plt.imshow(img, cmap='gray')
plt.show()
warp_parameter = dis.get_unwarp_params()
img_unwarped = dis.warp(img, warp_parameter)
plt.imshow(img_unwarped, cmap='gray')
plt.show()

# 




