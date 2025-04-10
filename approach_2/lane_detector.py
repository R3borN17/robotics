#!/usr/bin/env python

import os
import cv2
import math
import numpy as np
import yaml
import rospy, rospkg

class LaneDetector(object):
    def __init__(self, color):
        self.target_color = color

        # Get the package path
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('lane_to_map')  # Replace with your package name

        # Construct the full path to the YAML file
        yaml_file_path = os.path.join(package_path, 'config', 'config.yaml')

        self.lab_data = self.get_yaml_data(yaml_file_path)

        self.hue_yellow_l = rospy.get_param("~detect/lane/yellow/hue_l", 27)
        self.hue_yellow_h = rospy.get_param("~detect/lane/yellow/hue_h", 41)
        self.saturation_yellow_l = rospy.get_param("~detect/lane/yellow/saturation_l", 130)
        self.saturation_yellow_h = rospy.get_param("~detect/lane/yellow/saturation_h", 255)
        self.lightness_yellow_l = rospy.get_param("~detect/lane/yellow/lightness_l", 160)
        self.lightness_yellow_h = rospy.get_param("~detect/lane/yellow/lightness_h", 255)


    def maskYellowLane(self, image):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_yellow_l
        Hue_h = self.hue_yellow_h
        Saturation_l = self.saturation_yellow_l
        Saturation_h = self.saturation_yellow_h
        Lightness_l = self.lightness_yellow_l
        Lightness_h = self.lightness_yellow_h

        # define range of yellow color in HSV
        lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(image, image, mask = mask)

        return mask

    def get_yaml_data(self, yaml_file):
        yaml_file = open(yaml_file, 'r', encoding='utf-8')
        file_data = yaml_file.read()
        yaml_file.close()

        data = yaml.load(file_data, Loader=yaml.FullLoader)

        return data
