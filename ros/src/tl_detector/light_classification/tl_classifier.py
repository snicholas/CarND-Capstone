from styx_msgs.msg import TrafficLight
import numpy as np
import time
import cv2 
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import glob
# import rospy
import os 


class TLClassifier(object):
    folder = '/capstone/ros/imgs/'
    def __init__(self):
        self.model = None

        
    def recorddata(self, image, state):
        write = True
        if write:
            width, height = image.shape[1], image.shape[0]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fname = self.folder+'_'+str(state)+'_img_'+timestr+'.jpg'
            cv2.imwrite(fname, image)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if not self.model:
            self.model = load_model('../../lenetmod.h5')
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return np.argmax(self.model.predict(np.expand_dims(image, axis=0)))
