import os as os
import image_processing as ip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import sys
from sklearn import svm



# in order to train the SVM we need to convert the images the same way we would 
# later convert the video image. The steps we perform here are:
#     - loop through all filders and files in the training set
#     - convert each image into a feature set but apply certain filters a
#       and modifications to it
#     - create a big array that holds all feature sets
#     - train the support vector machine
     

# base directory where we hold all files for training and test
walk_dir = "/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/"
counter = 0

# we loop only if we need to - if the test data exists already in the array
# then we do not need to do this step
counter_vehicles = 0
counter_other = 0

X = []
Y =[]

train_data = {}

if os.path.isfile("./train_data.p"):
    # load the file
    print("using data from existing pickle file")
    train_data = pickle.load( open( "./train_data.p", "rb" ) )
else:
    print("loading train data from original image files")
    for root, subdirs, files in os.walk(walk_dir):
        
        for filename in files:
            
    #        print(counter)
            file_path = os.path.join(root, filename)
    #        print(file_path)
            if file_path.endswith(".png"):
            
                # read the image
                img = mpimg.imread(file_path)        
        #        img = cv2.imread(img_path)
            
        #        plt.imshow(img)
        #        plt.show()               
                
                # convert the image
                features = ip.image_to_featureset(img, 'RGB', 32, 'ALL')
                
                X.append(features)
                
                if "non-vehicles" in file_path:
                    counter_other += 1
                    Y.append("other")
                else:
                    counter_vehicles += 1
                    Y.append("vehicle")
    #            print(features)
        #        sys.exit() 
            
            # put it into the feature array
            
    # check data
    print("counter_other {}".format(counter_other))
    print("counter_vehicles {}".format(counter_vehicles))

    # save feature array
    train_data['X'] = X
    train_data['Y'] = Y
    print("writing data to pickle file")
    pickle.dump( train_data, open( "./train_data.p", "wb" ) )
    

#check
X = train_data['X']
Y = train_data['Y']

print("X shape {}".format(len(X)))
print("Y shape {}".format(len(Y)))


# train
print("train...")
clf = svm.SVC()
clf.fit(X, Y)  

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print("training done")

print("done - end of program")


# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict



# =============================================================================
# train and test split

# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#X_train, X_test, y_train, y_test = train_test_split(
#    scaled_X, y, test_size=0.2, random_state=rand_state)







    
    