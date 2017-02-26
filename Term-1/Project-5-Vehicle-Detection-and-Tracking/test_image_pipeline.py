
import image_processing as proc
import matplotlib.image as mpimg
import alsi_util as util

# thif file is for testing the pipleine on images
# then we use the pipeline and make it wokring on videos
# the svc is getting loaded and we briefly test it
#print("test SVM") 
#trn.smoke_test()

# load mage
print("load image")

for i in range(1,6):
    file_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/test_images/test" + str(i) + ".png"
    img = mpimg.imread(file_path)  
    util.show_image_from_image(img, "input")
    proc.process_image(img)