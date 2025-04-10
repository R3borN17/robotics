#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped

class LaneMapPublisher:
    def __init__(self):
        rospy.init_node('lane_map_publisher', anonymous=True)

        # Initialize TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Camera subscriber
        self.bridge = CvBridge()
        rospy.Subscriber("/rgbd_camera/rgb/top_view", Image, self.image_callback)
        rospy.Subscriber("/rgbd_camera/depth/camera_info", CameraInfo, self.camera_info_callback)

        # Lane map publisher
        self.map_pub = rospy.Publisher("/hiwonder/lane_map", OccupancyGrid, queue_size=10)

        # Camera Parameters (To be updated from /camera_info)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Map Parameters
        self.map_width = 200  # 200x200 grid cells
        self.map_height = 200
        self.resolution = 0.01  # Each cell represents 5cm
        self.origin_x = 0  # Map origin relative to camera
        self.origin_y = -self.map_width * self.resolution / 2

        # Occupancy Grid Map (Initialized as empty)
        self.lane_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

    def camera_info_callback(self, msg):
        """ Get camera intrinsic parameters from /camera_info topic """
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        rospy.loginfo(f"Camera Intrinsics Updated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def image_callback(self, msg):
        """ Convert detected lane image into a grid map """
        if self.fx is None or self.fy is None:
            rospy.logwarn("Camera intrinsics not available yet. Skipping frame.")
            return

        rospy.loginfo("Processing lane image...")

        # Convert ROS Image to OpenCV format
        lane_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")  # Assuming it's already a binary mask

        # Convert detected lane to occupancy grid
        self.update_lane_map(lane_mask)

        # Publish updated map
        self.publish_map()

        # Broadcast the TF between map frame and camera frame
        self.broadcast_tf()

    def update_lane_map(self, lane_mask):
        """ Convert lane mask to occupancy grid format and apply transformations """
        # Resize lane mask to match the grid size
        lane_resized = cv2.resize(lane_mask, (self.map_width, self.map_height), interpolation=cv2.INTER_NEAREST)

        # **Rotate 90 degrees counterclockwise and mirror**
        lane_rotated = np.rot90(lane_resized)  # Rotate 90 degrees counterclockwise
        lane_mirrored = np.fliplr(lane_rotated)  # Flip horizontally (mirror effect)

        # Convert to binary occupancy grid (0 = free, 100 = lane detected)
        self.lane_map[:] = 0  # Reset before updating
        self.lane_map[lane_mirrored > 0] = 100  # Assign lane markings


    def publish_map(self):
        """ Publish the lane map as an OccupancyGrid message """
        occupancy_msg = OccupancyGrid()
        occupancy_msg.header.stamp = rospy.Time.now()
        occupancy_msg.header.frame_id = "hiwonder/lane_map"

        occupancy_msg.info.resolution = self.resolution
        occupancy_msg.info.width = self.map_width
        occupancy_msg.info.height = self.map_height
        occupancy_msg.info.origin.position.x = self.origin_x
        occupancy_msg.info.origin.position.y = self.origin_y
        occupancy_msg.info.origin.position.z = 0.0

        occupancy_msg.data = self.lane_map.flatten().tolist()

        self.map_pub.publish(occupancy_msg)
        rospy.loginfo("Lane map published successfully.")

    def broadcast_tf(self):
        """ Broadcast TF between the lane_map frame and the camera frame """
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "hiwonder/base_link"
        transform.child_frame_id = "hiwonder/lane_map"

        # Set translation (position of the camera in the map frame)
        transform.transform.translation.x = 0.68
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0

        # Set rotation (identity quaternion)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform)
        rospy.loginfo("TF published between map and camera_link.")

if __name__ == "__main__":
    LaneMapPublisher()
    rospy.spin()
