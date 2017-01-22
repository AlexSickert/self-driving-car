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

os.chdir("F:\\CODE\\Udacity-Self-Driving-Car\\Term-1\\Project-3")

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
    
def speed_control(speed):
    #print("speed:")
    #print(speed)
    throttle = 0.2
    if Decimal(speed) > Decimal("5"):
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
    
    image_array = c
    
    #cv2.imshow("image", c);
    #cv2.waitKey();

    image_array = cut_and_scale(image_array)
    
    cv2.imwrite('test-camera-cut-and-scale.jpg',image_array)
    
    image_array = normalize_grayscale(image_array)  
    
    #cv2.imwrite('test-camera-normalized.jpg',image_array)
    
    
    #print("image_array before")
    #print(image_array.shape)
    
    image_array = np.expand_dims(image_array, 3) 
    
    #print(image_array.shape)
    
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