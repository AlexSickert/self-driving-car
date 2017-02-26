import numpy as np
import cv2


#=============================================================================

def crop_image(image, points_array):
    

#[0, 320, 64, 384]
#[32, 320, 96, 384]
#[64, 320, 128, 384]
#[96, 320, 160, 384]
#[128, 320, 192, 384]
#[160, 320, 224, 384]
#[192, 320, 256, 384]
#[224, 320, 288, 384]
#[256, 320, 320, 384]
#[288, 320, 352, 384]
#[320, 320, 384, 384]
#[352, 320, 416, 384]
#[384, 320, 448, 384]
#[416, 320, 480, 384]
#[448, 320, 512, 384]
#[480, 320, 544, 384]  
    
    
    startx = points_array[0]
    starty = points_array[1]
    endx = points_array[2]
    endy = points_array[3]
    crop_img = image[starty :endy, startx :endx]
    
   
    
    return crop_img
    
#=============================================================================
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-nx_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
#            window_list.append(((startx, starty), (endx, endy)))
            window_list.append([startx, starty, endx, endy])
    # Return the list of windows
    return window_list

#=============================================================================
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
#=============================================================================

    