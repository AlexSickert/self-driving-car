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
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import alsi_util as util
import random as rnd


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
info = []

train_data = {}

if os.path.isfile("./pickle_files/train_data.p"):
    # load the file
    print("using data from existing pickle file")
    train_data = pickle.load( open( "./pickle_files/train_data.p", "rb" ) )
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
                # save path so that we can later verify prediction
                info.append(file_path)
                
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
    train_data['info'] = info         
    print("writing data to pickle file")
    pickle.dump( train_data, open( "./pickle_files/train_data.p", "wb" ) )
    

#check
X = train_data['X']
Y = train_data['Y']
info = train_data['info']

print("X shape {}".format(len(X)))
print("Y shape {}".format(len(Y)))
print("info shape {}".format(len(info)))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train
if os.path.isfile("./pickle_files/model.pkl"):
    print("use saved model")
    clf = joblib.load('./pickle_files/model.pkl') 
    
else:
    
    print("train...")
    clf = svm.SVC()
    clf.fit(X_train, y_train)  
    print("training done. now saving the model")    
    joblib.dump(clf, './pickle_files/model.pkl') 

def check_accuracy():
    
    print("check accuracy")
    
    print('Test Accuracy of SVC = ', clf.score(X_test, y_test))
    #print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
    #print('For labels: ', y_test[0:10])
    
    print("check_accuracy done")
    
def smoke_test():    
    # do some predictions
    print("do some predictions")
    
    # take some of the original pictures convert them and predict outcome.
    # i do this as a "smoke test" if pipeline works
    
    for x in range(5):
        r = rnd.randint(1, len(Y))
        path = info[r]
        img = mpimg.imread(path)  
        features = ip.image_to_featureset(img, 'RGB', 32, 'ALL')    
        features = np.expand_dims(features, axis=0)
        res = clf.predict(features) 
        text = "Image shows " + str(res)
        #print(res[0])
        util.show_image(path, "Image shows " + text)
    
    print("smoke_test done")
    
# this will be used by the video stream
def predict(img):
    
    features = ip.image_to_featureset(img, 'RGB', 32, 'ALL')    
    features = np.expand_dims(features, axis=0)
    res = clf.predict(features) 
    
    return res


#
#
#smoke_test()

#
#img = mpimg.imread("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/train-data/vehicles/GTI_Left/image0011.png") 
#features = ip.image_to_featureset(img, 'RGB', 32, 'ALL')
#print(np.shape(features))
#
#img = mpimg.imread("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-5-Vehicle-Detection-and-Tracking/output_images/7['vehicle'].png") 
#features = ip.image_to_featureset(img, 'RGB', 32, 'ALL')
#print(np.shape(features))
#predict(img)

#
## Define a function to return some characteristics of the dataset 
#def data_look(car_list, notcar_list):
#    data_dict = {}
#    # Define a key in data_dict "n_cars" and store the number of car images
#    data_dict["n_cars"] = len(car_list)
#    # Define a key "n_notcars" and store the number of notcar images
#    data_dict["n_notcars"] = len(notcar_list)
#    # Read in a test image, either car or notcar
#    example_img = mpimg.imread(car_list[0])
#    # Define a key "image_shape" and store the test image shape 3-tuple
#    data_dict["image_shape"] = example_img.shape
#    # Define a key "data_type" and store the data type of the test image.
#    data_dict["data_type"] = example_img.dtype
#    # Return data_dict
#    return data_dict
#
#
#
#
#    
    