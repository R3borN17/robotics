#!/usr/bin/env python3

import rospy
import struct
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


class YellowLaneExtractor:
    def __init__(self):
        """Initialize the Yellow Lane Extractor Node"""
        rospy.init_node("yellow_lane_extractor")

        # Create ROS publisher
        self.yellow_pcl_pub = rospy.Publisher("/rgbd_camera/depth_registered/points_lane", PointCloud2, queue_size=10)

        # Subscribe to the incoming point cloud topic
        rospy.Subscriber("/rgbd_camera/depth_registered/points", PointCloud2, self.process_point_cloud)

        rospy.loginfo("Yellow Lane Extractor Node Started")

    def pointcloud2_to_numpy(self, msg):
        """Convert PointCloud2 to a NumPy structured array for fast processing."""
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        cloud_array = np.array(list(gen), dtype=np.float32)  # Convert to NumPy array
        return cloud_array

    def extract_rgb_from_numpy(self, rgb_array):
        """Extract RGB values from a NumPy array of PointCloud2 data in a vectorized way."""
        rgb_int = rgb_array.view(np.uint32)  # Convert float to int representation
        b = (rgb_int >> 16) & 0xFF  # Extract Blue
        g = (rgb_int >> 8) & 0xFF   # Extract Green
        r = rgb_int & 0xFF          # Extract Red
        return np.vstack((r, g, b)).T  # Stack R, G, B as columns

    def is_yellow(self, rgb_values):
        """Vectorized function to filter yellow points in an entire array using NumPy."""
        hsv = cv2.cvtColor(rgb_values.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        mask = (15 <= hsv[:, 0]) & (hsv[:, 0] <= 50) & (hsv[:, 1] > 50) & (hsv[:, 2] > 50)
        return mask  # Boolean mask for yellow points

        # MIN_YELLOW = np.array([100, 100, 0],   dtype=np.uint8)  # [R, G, B]
        # MAX_YELLOW = np.array([255, 255, 80], dtype=np.uint8)

        # # Create a boolean mask for each channel, then combine
        # mask = np.all((rgb_values >= MIN_YELLOW) & (rgb_values <= MAX_YELLOW), axis=1)
        # return mask  # 1D boolean array

    def filter_yellow_lane(self, msg):
        """Fast filtering of yellow lane points using NumPy and vectorized operations."""

        cloud_array = self.pointcloud2_to_numpy(msg)

        if cloud_array.shape[0] == 0:
            rospy.logwarn("Received empty point cloud!")
            return

        xyz = cloud_array[:, :3]  # Extract XYZ values
        rgb_values = self.extract_rgb_from_numpy(cloud_array[:, 3])  # Extract RGB

        # Apply yellow lane filtering using vectorized NumPy operations
        yellow_mask = self.is_yellow(rgb_values)
        yellow_points = xyz[yellow_mask]  # Filter only yellow points

        # rospy.loginfo(f"Total points: {cloud_array.shape[0]}, Yellow points: {yellow_points.shape[0]}")

        if yellow_points.shape[0] == 0:
            rospy.logwarn("No yellow points found, skipping publish")
            return

        # Convert NumPy array back to PointCloud2 and publish
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        yellow_cloud_msg = pc2.create_cloud(msg.header, fields, yellow_points)

        # Publish the processed point cloud
        self.yellow_pcl_pub.publish(yellow_cloud_msg)
        # rospy.loginfo("Published filtered yellow lane points!")

    def process_point_cloud(self, msg):
        """Handles incoming point cloud messages and processes them."""
        self.filter_yellow_lane(msg)


if __name__ == "__main__":
    extractor = YellowLaneExtractor()  # Create an instance of the class
    rospy.spin()  # Keep the node running