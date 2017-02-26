import training as trn
import sliding_window as slw
import matplotlib.image as mpimg
import alsi_util as util
import numpy as np
import cv2

# thif file is for testing the pipleine on images
# then we use the pipeline and make it wokring on videos


# the svc is getting loaded and we briefly test it
#print("test SVM") 
#trn.smoke_test()

#=============================================================================

    # if it is a vehicle, then add the window to the boxlist
    
def process_image(img): 
    
    sliding_sale = [64, 128]
    boxlist = []
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    
    for s in sliding_sale:
        
        crop_arr = slw.slide_window(img,x_start_stop=[None, None], y_start_stop=[300, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
        counter = 0
        
        for x in crop_arr:
            counter += 1
            cropped_image = slw.crop_image(img, x)
            res = trn.predict(cropped_image)
    #        cv2.imwrite("./output_images/" + str(unter) + str(res) + ".png", cropped_image)
            if str(res[0]) == "vehicle":
                boxlist.append(x)
#                print("vehicle")
           
                
    #by now we have identified all the boxes that contain a car
    # now we add the boxes to a heatmap
    for b in boxlist:
#        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[b[1]:b[3], b[0]:b[2]] += 1
        
#=============================================================================

               
#        startx = points_array[0]
#    starty = points_array[1]
#    endx = points_array[2]
#    endy = points_array[3]
#    crop_img = image[starty :endy, startx :endx]



# load mage
print("load image")

file_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/test_images/test4.png"
img = mpimg.imread(file_path)  
util.show_image_from_image(img, "")




process_image(img)