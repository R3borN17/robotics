#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_smsg

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from ros_numpy import numpify

class LaneToMap(object):
    def __init__(self):
        rospy.init_node('lane_to_map', anonymous=True)

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        self.map_sub = rospy.Subscriber("/hiwonder/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.lane_sub = rospy.Subscriber("/rgbd_camera/depth_registered/points_lane", PointCloud2, self.lane_callback, queue_size=1)

        # Publisher for updated map
        self.map_pub = rospy.Publisher("/hiwonder/map_with_lanes", OccupancyGrid, queue_size=1, latch=True)

        # Internal storage
        self.current_map = None
        self.got_map = False

        self.target_frame = "hiwonder/map"

    def map_callback(self, msg):
        """
        Store the latest map. We copy it so we can modify it safely.
        """
        self.current_map = msg
        self.got_map = True

    def lane_callback(self, cloud_msg):
        """
        Whenever new lane points come in, transform them to 'map' frame 
        and mark cells in the OccupancyGrid as occupied.
        """
        if not self.got_map:
            return  # Wait until we have a map to modify

        # Get transform from the point cloud's frame to the map frame
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                cloud_msg.header.frame_id,  # e.g., "hiwonder/depth_cam_link"
                cloud_msg.header.stamp,     # Use the same timestamp
                rospy.Duration(1.0)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn("TF transform error: %s", ex)
            return

        # Transform the entire cloud to the map frame
        transformed_cloud = tf2_smsg.do_transform_cloud(cloud_msg, transform_stamped)

        # Convert to a structured numpy array
        pc_arr = numpify(transformed_cloud)  # shape: (num_points,)
        # pc_arr has named fields like ['x', 'y', 'z', ...]

        # Make a local copy of the map data we can modify
        updated_map = OccupancyGrid()
        updated_map.header = self.current_map.header
        updated_map.info = self.current_map.info
        updated_map.data = list(self.current_map.data)  # mutable copy of the occupancy data

        map_width = updated_map.info.width
        map_height = updated_map.info.height
        resolution = updated_map.info.resolution
        origin_x = updated_map.info.origin.position.x
        origin_y = updated_map.info.origin.position.y

        # For each point in the lane point cloud (now in map frame):
        for point in pc_arr:
            px = point['x']
            py = point['y']
            # pz = point['z']  # if you need the height

            # Compute grid cell indices
            mx = int((px - origin_x) / resolution)
            my = int((py - origin_y) / resolution)

            # Check bounds
            if mx < 0 or mx >= map_width or my < 0 or my >= map_height:
                continue

            idx = my * map_width + mx

            # Mark as occupied (100). Alternatively, you could pick a different cost.
            updated_map.data[idx] = 100

        # Publish the updated map
        self.map_pub.publish(updated_map)

        # Optionally, store this updated map internally (so we keep cumulative changes)
        self.current_map = updated_map

def main():
    node = LaneToMap()
    rospy.spin()

if __name__ == '__main__':
    main()
