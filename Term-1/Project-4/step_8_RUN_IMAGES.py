
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
import os.path
import pickle

working_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/"
os.chdir(working_dir)    

# run all on one image

# calibrate

#image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/test2.jpg"
image_base_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/CarND-Advanced-Lane-Lines-master/test_images/"


folder_calibration = "/home/alex/Downloads/CarND-Camera-Calibration-master/calibration_wide/"
test_image_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-4/calibration_test/test_image.jpg"

print("start calibration")


if os.path.isfile("calibration.p"):
    print("load calibration from file")
    with open('calibration.p', 'rb') as handle:
        params = pickle.load(handle)
else:
    print("create calibration and save to file")
    params = cal.calibrate(folder_calibration, test_image_path, show_images=False)
    with open('calibration.p', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)



print("calibration done")


def process_image(image_path):
    
    #img = cv2.imread(image_path)
    img = plt.imread(image_path)
    original_image = img
    
    plt.imshow(img)
    plt.show()
    
    # use sobel and HLS color space for line detection
    img = colgrad.get_combined_HLS_Sobel(img)

    # undistort
    img_corrected = cal.un_distort(img, params)

    # perspective transform
    warp_parameter = dis.get_warp_params()
    #img = to_geyscale(img_corrected)
    img = dis.warp(img, warp_parameter)
    
    # save the image in a variable for for testing purposes
    warped_binary_image = img
    un_warp_parameter = dis.get_unwarp_params()
    img_unwarped = dis.warp(img, un_warp_parameter)
    polinomials = lpbo.find_lane(warped_binary_image)
    text = cur.calculate_radius(polinomials["left_fit"], polinomials["right_fit"])
    
    original_image = cur.write_text_on_image(original_image, text)
    
    #draw lines on the original image
    unwarped_lane = draw.draw_lines_on_warped(warped_binary_image, polinomials["fit_leftx"], polinomials["fit_rightx"], polinomials["fity"] )

    combined_result = draw.combine_images_and_unwarp(original_image, unwarped_lane, un_warp_parameter)
    plt.imshow(combined_result)
    plt.show()
    

for x in range(1,6):
    image_path = image_base_path + "test" + str(x) + ".jpg"
    print(image_path) 
    process_image(image_path)
