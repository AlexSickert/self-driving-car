import os as os
import imageprocessing as ip
import matplotlib.image as mpimg
import numpy as np
import pickle
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import alsi_util as util
import random as rnd
import sklearn.preprocessing as prep
import sliding_window as slw
from sklearn.utils import shuffle


# in order to train the SVM we need to convert the images the same way we would 
# later convert the video image. The steps we perform here are:
#     - loop through all filders and files in the training set
#     - convert each image into a feature set but apply certain filters a
#       and modifications to it
#     - create a big array that holds all feature sets
#     - train the support vector machine
     

#==============================================================================
color_space = "HLS"# already tried RGB
image_size = 64
hog_channel = 1   # already tried: 2

#==============================================================================
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
                #scale image 
#               print("scale max val: " + str(np.amax(img)))
                img = img * 255
                
                features = ip.image_to_featureset(img, color_space, image_size, hog_channel)
                
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

print("scaling")
#scaling
X_scaler = prep.StandardScaler().fit(X)
# Apply the scaler to X
X = X_scaler.transform(X)
print("scaling done")

# shuffle
X, Y = shuffle(X, Y, random_state=0)

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


#=============================================================================
def check_accuracy():
    
    print("check accuracy")
    
    print('Test Accuracy of SVC = ', clf.score(X_test, y_test))
    #print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
    #print('For labels: ', y_test[0:10])
    
    print("check_accuracy done")

#=============================================================================    
def smoke_test():    
    # do some predictions
    print("do some predictions")
    
    # take some of the original pictures convert them and predict outcome.
    # i do this as a "smoke test" if pipeline works
    
    for x in range(5):
        r = rnd.randint(1, len(Y))
        path = info[r]
        img = mpimg.imread(path)  
        features = ip.image_to_featureset(img, color_space, image_size, hog_channel) 
        
        features = X_scaler.transform(features)
        print("length of featureset = " + str(len(features)))
        features = np.expand_dims(features, axis=0)
        res = clf.predict(features) 
        text = "Image shows " + str(res)
        #print(res[0])
        util.show_image(path, "Image shows " + text)
    
    print("smoke_test done")

#smoke_test()
#=============================================================================
# this will be used by the video stream
def predict(img):
    # RGB, HLS
    features = ip.image_to_featureset(img, color_space, image_size, hog_channel)  
    features = np.expand_dims(features, axis=0)
    features = X_scaler.transform(features)
    res = clf.predict(features) 
    
    return res

#=============================================================================
# this method processes one image

def process_image(img): 
    
    print("max value: " + str(np.amax(img) )) 
    
#    if np.amax(img) > 1:
#        img = img / 255
    
    sliding_sale = [64, 128, 256]
    boxlist = []
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap_threshold = 0
    counter = 0
    
    for s in sliding_sale:
        print("sliding_sale: " + str(s))
        crop_arr = slw.slide_window(img,x_start_stop=[None, None], y_start_stop=[300, None], xy_window=(s, s), xy_overlap=(0.5, 0.5))
        counter = 0
        
        for x in crop_arr:
            counter += 1
            cropped_image = slw.crop_image(img, x)
            res = predict(cropped_image)
            if str(res[0]) == "vehicle":
                boxlist.append(x)
                counter += 1

    print("boxes with cars found: " + str(counter))
    #by now we have identified all the boxes that contain a car
    # now we add the boxes to a heatmap
    for b in boxlist:
#        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[b[1]:b[3], b[0]:b[2]] += 1
               
    heatmap[heatmap <= heatmap_threshold] = 0
          
    res_image = ip.draw_labeled_bboxes(img, heatmap)
    
#    cv2.imwrite('result_output.jpg',res_image)
#    util.show_image_from_image(res_image, "output")
    
    return res_image



#=============================================================================
# this method processes one image
#==============================================================================
def process_video(img): 
    
#    print("max value: " + str(np.amax(img) )) 
    
#    if np.amax(img) > 1:
#        img = img / 255
    
    sliding_sale = [64, 128, 256]
    boxlist = []
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap_threshold = 12  #because we measure across several images
    counter = 0
    
    for s in sliding_sale:
#        print("sliding_sale: " + str(s))
        crop_arr = slw.slide_window(img,x_start_stop=[None, None], y_start_stop=[300, None], xy_window=(s, s), xy_overlap=(0.5, 0.5))
        counter = 0
        
        for x in crop_arr:
            counter += 1
            cropped_image = slw.crop_image(img, x)
            res = predict(cropped_image)
            if str(res[0]) == "vehicle":
                boxlist.append(x)
                counter += 1
    
    insert_boxlist(boxlist)
    
    #by now we have identified all the boxes that contain a car
    # now we add the boxes to a heatmap but before we do this we aggregate
    # the boxlists across several images to have a smoother vizualtionation
    # and to get rid of false positives
    
    boxlist = get_consolidated_boxlist()
    
    for b in boxlist:
#        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[b[1]:b[3], b[0]:b[2]] += 1
               
    heatmap[heatmap <= heatmap_threshold] = 0
          
    res_image = ip.draw_labeled_bboxes(img, heatmap)
    
#    cv2.imwrite('result_output.jpg',res_image)
#    util.show_image_from_image(res_image, "output")
    
    return res_image
#==============================================================================
# we create something like a queue to store heatmaps across several images

boxlists = []
boxlist_length = 6

def insert_boxlist(b):
    global boxlists
    
    if len(boxlists) > boxlist_length:  
#        print("removing. lenght is " + str(len(boxlists)))
        boxlists.pop()
        
    boxlists.insert(0,b)

def get_consolidated_boxlist():
    boxlist_consolidated = []
    for boxlist in boxlists:
        for b in boxlist:
            boxlist_consolidated.append(b)
    return boxlist_consolidated
        
    

#==============================================================================
   
