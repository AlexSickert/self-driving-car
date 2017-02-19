import os as os
import image_processing as ip

# in order to train the SVM we need to convert the images the same way we would 
# later convert the video image. 


# loop through all files and create the training data

# loop

# convert the image

walk_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/"
counter = 0

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


    
    