
import imageprocessing as proc
import matplotlib.image as mpimg
import alsi_util as util
import training as trn

# thif file is for testing the pipleine on images
# then we use the pipeline and make it wokring on videos
# the svc is getting loaded and we briefly test it
#print("test SVM") 
#trn.smoke_test()

# load mage
print("load image")

debug = {}
debug['path'] = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/output_images/"
debug['ending'] = ".jpg"
debug['debug'] = 1 # yes = 1

for i in range(1,6):
    debug['counter']  = i
    file_path = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/test_images/test" + str(i) + ".jpg"
    img = mpimg.imread(file_path)  
#    img = img * 255
#    util.show_image_from_image(img, "input")
    ret = trn.process_image(img, debug)
    util.show_image_from_image(ret, "output")
    
#    output_path = "./output_images/" + str(i) + ".jpg"
#    util.save_image(ret, output_path)