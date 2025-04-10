#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def compute_ipm(image, K):
    """
    Compute Inverse Perspective Mapping (IPM) to convert front-view image to top-down.
    """
    height, width = image.shape[:2]
    
    src_pts = np.array([[100, 480], [540, 480], [250, 320], [390, 320]], dtype=np.float32)
    dst_pts = np.array([[100, 480], [540, 480], [100, 0], [540, 0]], dtype=np.float32)

    
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    adjustment_matrix = np.array([
        [0.28, 0, 240],
        [0, .68, 145],
        [0, 0, 1]])
    M = adjustment_matrix @ M 

    # Apply transformation
    ipm_image = cv2.warpPerspective(image, M, (width, height))
    
    return ipm_image

def image_callback(msg):
    """ ROS callback function to process incoming images."""
    bridge = CvBridge()
    
    # Convert ROS Image to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Camera Intrinsic Matrix
    K = np.array([[ 577.3,           0.,         -97.5347032 ],
                  [   0.,          577.3,         -2.63607306],
                  [   0.,            0.,            0.        ]])
    
    # Apply IPM transformation
    top_view = compute_ipm(cv_image, K)
    
    # Convert back to ROS Image and publish
    pub = rospy.Publisher("/rgbd_camera/rgb/top_view", Image, queue_size=1)
    pub.publish(bridge.cv2_to_imgmsg(top_view, "bgr8"))

def main():
    rospy.init_node("ipm_transformer", anonymous=True)
    rospy.Subscriber("/rgbd_camera/rgb/lane", Image, image_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
