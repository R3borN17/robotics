#!/usr/bin/env python
import rospy
import rosbag
import tf
import numpy as np
import cv2
# import os

from sensor_msgs.msg import CompressedImage, PointCloud2, Image, LaserScan
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseWithCovariance
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from laser_geometry import LaserProjection
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray

from lane_detector import LaneDetector

class lab_1():

    def __init__(self):

        rospy.init_node('lane_to_map', anonymous=True)
        
        # lane detector
        self.lane_detector = LaneDetector('yellow')

        # CV bridge
        self.bridge = CvBridge()


        # image subscriber and publisher: TODO
        rospy.Subscriber("/rgbd_camera/rgb/image_raw", Image, self.image_callback)

        # Image publisher for lane detection output
        self.image_pub = rospy.Publisher("/rgbd_camera/rgb/lane", Image, queue_size=1)

        

    def image_callback(self, msg):

        # convert Image msg to CV2
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
        
        # use OpenCV to detect yellow lane
        yellow_mask_image = self.lane_detector.maskYellowLane(cv_image)

        # convert CV2 to Image msg
        mask_msg = self.bridge.cv2_to_imgmsg(yellow_mask_image, encoding="mono8")
        mask_msg.header.frame_id = 'hiwonder/rgbd_camera_rgb_optical_frame'

        # publish mask_msg
        self.image_pub.publish(mask_msg)

        pass



if __name__ == '__main__':

    proc = lab_1()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
