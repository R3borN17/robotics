#!/usr/bin/env python
import rospy
import rosbag
import tf
import numpy as np
import math
import cv2
import os
import sys
import tty
import termios

from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray

from tf.transformations import quaternion_from_euler

class lab_2():

    def __init__(self):
        rospy.init_node('lab_2_task2', anonymous=True)

        self.save_path = "/home/developer/workspace/src/lab_2/screenshot"

        os.makedirs(self.save_path, exist_ok = True)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/robot_1/depth_cam/rgb/image_raw", Image, self.image_callback)

        self.latest_image = None

    def image_callback(self, msg): 
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def save_image(self):
        if self.latest_image is None:
            rospy.logwarn("No image")
            return
        
        count = len(os.listdir(self.save_path))
        filename = os.path.join(self.save_path, f"image_{count+1:03d}.jpg")

        cv2.imwrite(filename, self.latest_image)
        rospy.loginfo(f"Image saved: {filename}")

    def keyboard_listener(self):
        rospy.loginfo("Press 's' to save an image.")
        while not rospy.is_shutdown():
            key = self.get_key()
            if key == 's':
                self.save_image()
    
    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key= sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
if __name__ == '__main__':

    lab_2 = lab_2()

    try:
        lab_2.keyboard_listener()

    except KeyboardInterrupt:
        print ("shutting down")



