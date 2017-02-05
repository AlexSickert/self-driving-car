import os
import step_1_calibrate as cal
import step_2_distortion_correction as dis
import step_3_color_and_gradient as colgrad
import matplotlib.pyplot as plt
import cv2
import numpy as np


def draw_lines_on_warped(warped, left_fitx, right_fitx, ploty ):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#    plt.imshow(color_warp)
    return color_warp
    
def combine_images_and_unwarp(image, color_warp, Minv):
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
#    result = cv2.addWeighted(image, 1, image, 0.3, 0)
    return result
    
    
def test():
        
    result1 = draw_lines_on_warped(warped_binary_image, polinomials["fit_leftx"], polinomials["fit_rightx"], polinomials["fity"] )
    plt.imshow(result1)
    
    result2 = combine_images_and_unwarp(original_image, result1, un_warp_parameter)
    plt.imshow(result2)
#    
#    plt.imshow(original_image)
#    
    
    
#test()