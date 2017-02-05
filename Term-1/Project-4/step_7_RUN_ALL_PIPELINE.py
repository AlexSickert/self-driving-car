
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
import step_3_color_and_gradient as colgrad
import step_5_lane_pixel_and_boundary as lpbo
import step_6_curvature as cur
import step_7_draw_lines as draw
import matplotlib.pyplot as plt
import cv2
import numpy as np


working_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/"
os.chdir(working_dir)    

# run all on one image

# calibrate

image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/test2.jpg"

folder_calibration = "/home/alex/Downloads/CarND-Camera-Calibration-master/calibration_wide/"
test_image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_test/test_image.jpg"

print("start calibration")
params = cal.calibrate(folder_calibration, test_image_path, show_images=False)
print("calibration done")
#img = cv2.imread(image_path)
img = plt.imread(image_path)
original_image = img

print("show un-corrected version of image")
plt.imshow(img)
plt.show()

# use sobel and HLS color space for line detection
print("use sobel and HLS color space for line detection")
img = colgrad.get_combined_HLS_Sobel(img)
plt.imshow(img, cmap='gray')
plt.show()

# undistort
print("correct an image")
img_corrected = cal.un_distort(img, params)
print("show corrected version of image")
plt.imshow(img_corrected, cmap='gray')
plt.show()

# perspective transform
print("perspective transform - warp")
warp_parameter = dis.get_warp_params()
#img = to_geyscale(img_corrected)
img = dis.warp(img, warp_parameter)

# save the image in a variable for for testing purposes

warped_binary_image = img

plt.imshow(img, cmap='gray')
plt.show()

print("un-warp")
un_warp_parameter = dis.get_unwarp_params()
img_unwarped = dis.warp(img, un_warp_parameter)
plt.imshow(img_unwarped, cmap='gray')
plt.show()

print("finding polynomial")

polinomials = lpbo.find_lane(warped_binary_image)

print("calculate radius of curve")

#test(polinomials["left_fit"], polinomials["right_fit"])    
cur.calculate_radius(polinomials["left_fit"], polinomials["right_fit"])

#draw lines on the original image
print("draw lines on the original image")
unwarped_lane = draw.draw_lines_on_warped(warped_binary_image, polinomials["fit_leftx"], polinomials["fit_rightx"], polinomials["fity"] )
plt.imshow(unwarped_lane)
plt.show()

print("unwarp and combine")

combined_result = draw.combine_images_and_unwarp(original_image, unwarped_lane, un_warp_parameter)
plt.imshow(combined_result)
plt.show()





