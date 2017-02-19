import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from decimal import *
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import os
import sys

import math

image_counter = 0


left_lines = [[0,0,0,0]]  
right_lines = [[0,0,0,0]]
start_of_left = True
start_of_right = True

left_slopes = [[0,0]]  
right_slopes = [[0,0]]

#os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")
os.chdir("/home/alex/CODE/Udacity-Self-Driving-Car/Term-1/Project-3/")

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


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


#def optimize_contrast(img):
##    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
##    res = clahe.apply(img)
#    res = cv2.equalizeHist(img)
#    return res
    
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
    
  

          
     
def speed_control(speed):
    #print("speed:")
    #print(speed)
    throttle = 0.2
    if Decimal(speed) > Decimal("10"):
        throttle = 0.0
    return throttle

@sio.on('telemetry')
def telemetry(sid, data):
    #print("in telemetry")
    # The current steering angle of the car
    #steering_angle = data["steering_angle"]
    #print("steering_angle")
    #print(steering_angle)
    # The current throttle of the car
    throttle = data["throttle"]
    #print("throttle")
    #print(throttle)
    # The current speed of the car
    speed = data["speed"]
    #print("speed")
    #print(speed)
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    #image.show()
    image_array = np.asarray(image)
    
    cv2.imwrite('test-camera-raw.jpg',image_array)
    
    c = cv2.imread('test-camera-raw.jpg', 0)
    
    #cv2.imwrite('test-camera-raw2.jpg',c)
    
    img = c
#    
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
#
##    img = cut_and_scale(img)

    image_array = cut_and_scale(img)
    
#    image_array = optimize_contrast(image_array)

    global image_counter
    
    image_counter = image_counter + 1
    
    image_path = "./debug-camera/" + str(image_counter).zfill(6) + ".jpg"
#    image_path =  str(image_counter).zfill(6) + ".jpg"
    
    cv2.imwrite(image_path,image_array)
    
    image_array = normalize_grayscale(image_array)  
    
    #cv2.imwrite('test-camera-normalized.jpg',image_array)
    
#    print("image_array before")
#    print(image_array.shape)
#    
    image_array = np.expand_dims(image_array, 3) 
#    
#    print(image_array.shape)
    
    #image_array.resize((50, 160, 1))

    #imgplot = plt.imshow(image_array, cmap='gray')
    
    #exit(1)
     
#    imgplot = plt.imshow(cut_and_scale(img), cmap='gray') 
    
    #print("image_array after")
    #print(image_array.shape)
    
    #print("ok 2")
    transformed_image_array = image_array[None, :, :, :]
    #print("transformed_image_array")
    #print(transformed_image_array.shape)
    #transformed_image_array = list()
    
    #print("exit")
    #sys.exit()
    
    #print("ok 3")
     
    #transformed_image_array.append(image_array)
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #print("ok 4 - now prediction ---------------------------------")
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
   
    steering_angle = steering_angle * 1.0
    
    if steering_angle > 1:
        steering_angle = 0.99
    if steering_angle < -1:
        steering_angle = -0.99
        
        
    
    print(steering_angle)
    
    
    #print("prediction end ---------------------------------")
    
#    steering_angle = float(0)
    #print("steering_angle from prediction")
    #print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
#    throttle = 0.1
    throttle = speed_control(speed)
    
    #steering_angle = float(0.999)
    #print(steering_angle, throttle)
    send_control(steering_angle, throttle)
    


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    #print("in send_control")
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Remote Driving')
    #parser.add_argument('model', type=str,
    #help='Path to model definition json. Model weights should be on the same path.')
    #args = parser.parse_args()
    
    file_name = 'model.json'
    with open(file_name, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        print("reading model")
        model = model_from_json(jfile.read())


    print("compile model")
    model.compile("adam", "mse")
    weights_file = file_name.replace('json', 'h5')
    print("loading weights")
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)