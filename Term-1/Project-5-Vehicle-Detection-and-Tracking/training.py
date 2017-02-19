import os as os
import image_processing as ip

# in order to train the SVM we need to convert the images the same way we would 
# later convert the video image. The steps we perform here are:
#     - loop through all filders and files in the training set
#     - convert each image into a feature set but apply certain filters a
#       and modifications to it
#     - create a big array that holds all feature sets
#     - train the support vector machine
     

# base directory where we hold all files for training and test
walk_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/"
counter = 0

# we loop only if we need to - if the test data exists already in the array
# then we do not need to do this step

for root, subdirs, files in os.walk(walk_dir):
    
    for filename in files:
        counter += 1
        print(counter)
        file_path = os.path.join(root, filename)
        print(file_path)
        
        # read the image
        
        # convert the image
        features = ip.image_to_featureset(img)
        
        # put it into the feature array
        
# save feature array

# train

# save model


    
    