import pandas as pd
import time
import os
import numpy as np
import cv2
import sys
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

left_lines = [[0,0,0,0]]  
right_lines = [[0,0,0,0]]
start_of_left = True
start_of_right = True

left_slopes = [[0,0]]  
right_slopes = [[0,0]]

processFromScratch = True
#processFromScratch = False
from pathlib import Path

data_array_file_name = "data-array.npy"

my_file = Path(data_array_file_name)
if my_file.is_file():
    print('file exists... we remove it')
    os.remove(data_array_file_name)
    

print('Modules loaded.')
os.chdir("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/")

# ## Load the Data
#  F:\CODE\Udacity-Self-Driving-Car\Term-1\Project-3\simulator-windows-64\training2
#df = pd.read_csv('./data/data/driving_log.csv', header=0)
df = pd.read_csv('/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/data2/driving_log.csv', header=0)

#F:\CODE\Udacity-Self-Driving-Car\Term-1\Project-3\simulator-windows-64\ALL 

print('csv loaded.')  

#   center,left,right,steering,throttle,brake,speed
df = df[['center', 'steering']]

print("rows {} columns {}".format(df.shape[0], df.shape[1]))

#df[['steering']].unique()

x = len(df['steering'].unique())
m = df['steering'].max()
m = df['steering'].min()
print(x)  # we have 124 levels

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

arr = list()
#arr = np.empty([1, 360, 160])

x = len(df['steering'])

start = time.time()

def cut_and_scale(img):
    img = img[60:160, 0:320]
    img = cv2.resize(img,(160, 50), interpolation = cv2.INTER_CUBIC)
    return img
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def optimize_contrast(img):
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    res = clahe.apply(img)
    res = cv2.equalizeHist(img)
    return res
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
       
    draw_lines(line_img, lines)
    return line_img 
    
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
#    print("ok2")
#    cv2.imwrite('weighted_img-1.jpg',initial_img)
#    cv2.imwrite('weighted_img-2.jpg',img)
    return cv2.addWeighted(initial_img, α, img, β, λ)
         
def x_at_border(x_max, y_max, slope, intercept, default):
    try:
        if slope > 0 :
            if slope * x_max + intercept > y_max:
                return int((y_max - intercept) / slope) - 1 
            else:
                return x_max - 1
        else:
            if intercept < 0:
                return 1
            else:
                return int((y_max - intercept) / slope) - 1 
    except:
        return default
           
    
def draw_lines(img, lines, color=255, thickness=5):

    
    error_counter = 0
    
    #print(img.shape)
    width = img.shape[1]
    height = img.shape[0]
    
    # approach: weighted average of current image and average of last x images
    
    right_all = 0
    right_all_length = 0
    right_intercept_all = 0
    
    left_all = 0
    left_all_length = 0
    left_intercept_all = 0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2-x1) != 0 and (y2-y1) != 0: 
                try:
                    slope = (y2-y1)/(x2-x1)  
                    length = np.sqrt(np.exp2(y2-y1) + np.exp2(x2-x1))  # calcualte the length of the line for weighting
                    intercept = y2 - (slope * x2)
            
                    # we ignore slopes that are too flat or too steep
                    if slope > 0.3 or slope < -0.3:
                        # skopes maller zero belong to the left side of the lane
                        if slope < 0:
                            left_all += length * slope
                            left_all_length += length
                            left_intercept_all += intercept * length                
                        else:
                            right_all += length * slope
                            right_all_length += length
                            right_intercept_all += intercept * length
                except:
                    error_counter += 1
            
    
    left_slope = left_all / left_all_length
    left_intercept = left_intercept_all / left_all_length
    
    right_slope = right_all / right_all_length
    right_intercept = right_intercept_all / right_all_length
        
    #add the slope and intercept to an array to calculate also the average across several images
    # first reference the global variables
    global start_of_left
    global start_of_right
    global left_slopes
    global right_slopes
    
    # add data to array
    left_slopes = add_to_array(left_slopes, [left_slope, left_intercept], start_of_left)
    start_of_left = False
    right_slopes = add_to_array(right_slopes, [right_slope, right_intercept], start_of_right)
    start_of_right = False
    
    # get the average of the slopes array
    avg_left = get_average_line(left_slopes)
    left_slope = avg_left[0]
    left_intercept = avg_left[1]

    avg_right = get_average_line(right_slopes)
    right_slope = avg_right[0]
    right_intercept = avg_right[1]
    
       
    # tests revealed that try-catch is needed here. 
    # identify the x value of the points that are on the horizon
    try:
        horizon_x = int((right_intercept - left_intercept) / (left_slope - right_slope))
    except:
        horizon_x = int(width/2)
    # identify the y value of the points that are on the horizon    
    try:
        horizon_y = int(left_slope * horizon_x + left_intercept)
    except:
        horizon_y = int(height/2)
        
    
    # calculate where the line should start relative to the border of the picture
    
    left_start_x = x_at_border(width, height, left_slope, left_intercept, horizon_x)
    right_start_x = x_at_border(width, height, right_slope, right_intercept, horizon_x)
      
    left_p1_x = int(left_start_x)
    left_p1_y = int(left_start_x * left_slope + left_intercept)    
    left_p2_x = int(horizon_x)
    left_p2_y = int(horizon_x * left_slope + left_intercept) 

    right_p1_x = int(right_start_x)
    right_p1_y = int(right_start_x * right_slope + right_intercept)    
    right_p2_x = int(horizon_x)
    right_p2_y = int(horizon_x * right_slope + right_intercept) 

    # add the average and weigthed line to the array for drawing on current image
    lines = []       
    lines.append([left_p1_x, left_p1_y, left_p2_x, left_p2_y])      
    lines.append([right_p1_x, right_p1_y, right_p2_x, right_p2_y])
    
    #print(lines)
    
    for line in lines:
        #print(line)
        for x1,y1,x2,y2 in [line]:
            try:
                slope = ((y2-y1)/(x2-x1))
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            except:
                foo = 1
                #print()

                
def get_average_line(array):
    #return np.mean(array, axis=0, dtype=int).tolist()
    return np.mean(array, axis=0 ).tolist()
                
def add_to_array(arr, line, is_start):
    #print(line)
    #print(len(arr))
    if len(arr) > 20:
        arr = np.delete(arr, 1, axis=0)
    
           
    #arr = np.append(arr, [line], axis= 1)
 
    if is_start == True:
        arr = [line]
        is_start = False
    else:
        arr = np.append(arr, [line], axis= 0)
        
    #print("elements in array")
    #print(len(arr))
    #print("---")
    return arr

    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape), dtype=np.uint8)
       
    draw_lines(line_img, lines)
    return line_img
    
    
#test
#img = cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg',0)
#imgplot = plt.imshow(img, cmap='gray')
#imgplot = plt.imshow(cut_and_scale(img), cmap='gray')    

for i in range(0, x):
#    path = './data/data/' + df['center'][i]  #this was for original
    path = df['center'][i]  #this was for original
    
    img = cv2.imread(path,0)   
    #cv2.imwrite('test-data-cut_and_scale.jpg',cut_and_scale(img))
#    cv2.imwrite('1-image-raw.jpg',img)
#    img_gaussian = gaussian_blur(img, 5)
##    cv2.imwrite('2-image-blure.jpg',img)
#    img_canny = canny(img_gaussian, 50, 150)
##    cv2.imwrite('3-image-canny.jpg',img)
#
#    rho = 2 # distance resolution in pixels of the Hough grid
#    theta = np.pi/180 # angular resolution in radians of the Hough grid
#    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
#    min_line_len = 40 #minimum number of pixels making up a line
#    max_line_gap = 20    # maximum gap in pixels between connectable line segments
#    
#    image_lines = hough_lines(img_canny, rho, theta, threshold, min_line_len, max_line_gap)
##    print(image_lines.shape)
##    img = np.expand_dims(img, axis=3)
#    
##    print(img.shape)
#    img = weighted_img(image_lines, img, α=0.8, β=1., λ=0.)

    img = cut_and_scale(img)
    
    image_path = "./debug/" + str(i).zfill(6) + ".jpg"
    
    cv2.imwrite(image_path,img)
    
#    img = optimize_contrast(img)
    
    if i == 1:
        print("writing image")
        cv2.imwrite('test-camera-cut-and-scale-loading-2.jpg',img)
#        sys.exit(1)
    img = normalize_grayscale(img)
    
    arr.append(img)
    

end = time.time()

elapsed = end - start
print("elapsed time (sec):")
print(elapsed)
 
myarray = np.asarray(arr) 
print(myarray.shape)
#print("exit")
#sys.exit()

myarray = np.expand_dims(myarray, 3)    
#myarray = normalize_grayscale(myarray)    
print(myarray.shape)
print("--------------- SAVE -----------------") 
np.save("data-array.npy", myarray, allow_pickle=True, fix_imports=True)
print("--------------- ALL DONE -----------------") 

