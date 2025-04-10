#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2
import math
import time
import queue
import rospy
import signal
import threading
import numpy as np
import lane_detect
import hiwonder_sdk.pid as pid
import hiwonder_sdk.misc as misc
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import hiwonder_sdk.common as common
from hiwonder_app.common import Heart
from hiwonder_interfaces.msg import ObjectsInfo
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from hiwonder_servo_controllers.bus_servo_control import set_servos
from hiwonder_sdk.common import cv2_image2ros, colors, plot_one_box
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

class SelfDrivingNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.01, 0.0, 0.0)
        self.param_init()

        self.image_queue = queue.Queue(maxsize=2)
        # New class list as provided
        self.classes = ['green_light', 'no_left_turn', 'no_light', 'no_right_turn', 
                        'parking', 'red_light', 'speed_limit_5', 'speed_limit_lift', 
                        'stop', 'turn_right', 'yellow_light']

        self.lock = threading.RLock()
        self.colors = common.Colors()
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.acker_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', Twist, queue_size=1)  # Velocity control
        self.joints_pub = rospy.Publisher('servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)  # Servo control
        self.result_publisher = rospy.Publisher('~image_result', Image, queue_size=1)

        self.enter_srv = rospy.Service('~enter', Trigger, self.enter_srv_callback)
        self.exit_srv = rospy.Service('~exit', Trigger, self.exit_srv_callback)
        self.set_running_srv = rospy.Service('~set_running', SetBool, self.set_running_srv_callback)
        self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))

        if not rospy.get_param('~only_line_follow', False):
            while not rospy.is_shutdown():
                try:
                    if rospy.get_param('/yolov5/init_finish'):
                        break
                except:
                    rospy.sleep(0.1)
            rospy.ServiceProxy('/yolov5/start', Trigger)()
        while not rospy.is_shutdown():
            try:
                if rospy.get_param('/hiwonder_servo_manager/init_finish') and rospy.get_param('/joint_states_publisher/init_finish'):
                    break
            except:
                rospy.sleep(0.1)
        set_servos(self.joints_pub, 1, ((1, 500), ))  # Initial posture
        rospy.sleep(1)
        self.acker_pub.publish(Twist())
        self.dispaly = False
        if rospy.get_param('~start', True):
            self.dispaly = True
            self.enter_srv_callback(None)
            self.set_running_srv_callback(SetBoolRequest(data=True))
        self.park_action() 
        self.image_proc()

    def param_init(self):
        self.start = False
        self.enter = False

        self.speed_limit_detected = False
        # We'll use count_slow_down as a timestamp for the speed limit event.
        self.count_slow_down = 0

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # x pixel coordinate of the parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # Start turning

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # Right turning sign

        # New flag for no left turn sign.
        self.no_left_turn = False

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # Stop sign
        self.start_park = False  # Start parking sign

        self.count_crosswalk = 0
        # Use yellow_light to simulate zebra crossing distance
        self.crosswalk_distance = 0  
        self.crosswalk_length = 0.1 + 0.3  # Zebra crossing length + vehicle length

        self.start_slow_down = False  # Slowdown flag
        self.normal_speed = 0.15  # Normal forward speed
        self.slow_down_speed = 0.1  # Slow down speed

        self.traffic_signs_status = None  # Record traffic light status
        self.red_loss_count = 0

        self.stop_detected = False
        self.stop_start_time = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # New flag for triggering the 90° turn maneuver.
        self.do_turn_90 = False

    def enter_srv_callback(self, _):
        rospy.loginfo("self driving enter")
        with self.lock:
            self.start = False
            camera = rospy.get_param('/depth_camera_name', 'depth_cam')  # Get camera parameters
            self.image_sub = rospy.Subscriber('/%s/rgb/image_raw' % camera, Image, self.image_callback)
            self.object_sub = rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.get_object_callback)
            self.acker_pub.publish(Twist())
            self.enter = True
        return TriggerResponse(success=True)

    def exit_srv_callback(self, _):
        rospy.loginfo("self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                rospy.logerr(str(e))
            self.acker_pub.publish(Twist())
        self.param_init()
        return TriggerResponse(success=True)

    def set_running_srv_callback(self, req: SetBoolRequest):
        rospy.loginfo("set_running")
        with self.lock:
            self.start = req.data
            if not self.start:
                self.acker_pub.publish(Twist())
        return SetBoolResponse(success=req.data)

    def shutdown(self, signum, frame):  # Ctrl+C shutdown process
        self.is_running = False
        rospy.loginfo('shutdown')

    def image_callback(self, ros_image):  # Target check callback
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3),
                               dtype=np.uint8, buffer=ros_image.data)  # Original RGB image
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)
    
    # Parking handling (unchanged)
    def park_action(self):
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = twist.linear.x * math.tan(-0.6) / 0.213
        self.acker_pub.publish(twist)
        rospy.sleep(3)

        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = -twist.linear.x * math.tan(-0.6) / 0.213
        self.acker_pub.publish(twist)
        rospy.sleep(2)

        twist = Twist()
        twist.linear.x = -0.15
        twist.angular.z = twist.linear.x * math.tan(-0.6) / 0.213
        self.acker_pub.publish(twist)
        rospy.sleep(1.5)

        set_servos(self.joints_pub, 0.1, ((9, 500), ))
        self.acker_pub.publish(Twist())

    # New method: 90° Turn Maneuver
    def turn_90_maneuver(self):
        # Phase 1: Drive forward with a right turn.
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = twist.linear.x * math.tan(-0.30) / 0.213
        self.acker_pub.publish(twist)
        rospy.sleep(2.0)  # Adjust this duration as needed.

        # Phase 2: Backup while turning left.
        twist = Twist()
        twist.linear.x = -0.15
        # Reverse angular direction to turn left while backing up.
        twist.angular.z = -twist.linear.x * math.tan(-0.40) / 0.213
        self.acker_pub.publish(twist)
        rospy.sleep(1.5)  # Adjust this duration as needed.

        # Phase 3: Resume straight driving.
        self.acker_pub.publish(Twist())
        # Reset turning flags so that subsequent maneuvers can be triggered.
        self.do_turn_90 = False
        self.turn_right = False
        self.no_left_turn = False

    def image_proc(self):
        while self.is_running:
            time_start = time.time()
            image = self.image_queue.get(block=True)
            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]
                
                # Speed Limit Timeout Check
                current_time = rospy.get_time()
                if self.speed_limit_detected and (current_time - self.count_slow_down) > 3.0:
                    self.speed_limit_detected = False
                    self.start_slow_down = False

                if self.speed_limit_detected:
                    self.start_slow_down = True

                # Get the binary image of the lane line 
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()
                # Traffic Light Handling:
                if self.traffic_signs_status is not None:
                    if self.traffic_signs_status.class_name == 'red_light':
                        twist.linear.x = 0   # Stop
                        self.stop = True
                    elif self.traffic_signs_status.class_name == 'yellow_light':
                        twist.linear.x = self.slow_down_speed  # Slow down
                        self.stop = False
                    elif self.traffic_signs_status.class_name == 'green_light':
                        twist.linear.x = self.normal_speed  # Normal speed
                        self.stop = False
                else:
                    # Use speed limit logic if no traffic light detected.
                    if self.start_slow_down:
                        twist.linear.x = self.slow_down_speed
                        if rospy.get_time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                            self.start_slow_down = False
                    else:
                        twist.linear.x = self.normal_speed

                if self.stop_detected:
                    elapsed = rospy.get_time() - self.stop_start_time
                    if elapsed < 5.0:
                        self.acker_pub.publish(Twist())
                        continue
                    else:
                        self.stop_detected = False 
			  
                # Parking condition remains unchanged.
                if 0 < self.park_x and 180 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park and 235 < self.crosswalk_distance:
                        self.acker_pub.publish(Twist())
                        self.start_park = True
                        self.stop = True
                        threading.Thread(target=self.park_action).start()
                
                # Get lane detection result.
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                
                # Lane tracking processing:
                if lane_x >= 0 and not self.stop:
                    # Trigger the 90° turn maneuver if any of these conditions are met:
                    # (a) lane_x >= 175, (b) turn_right sign, or (c) no_left_turn sign.
                    if (lane_x >= 175) or self.turn_right or self.no_left_turn:
                        if not self.do_turn_90:
                            self.do_turn_90 = True
                            threading.Thread(target=self.turn_90_maneuver).start()
                        # Skip normal lane tracking during the maneuver.
                        continue
                    else:
                        # For minor deviations, use PID correction.
                        self.count_turn = 0
                        if rospy.get_time() - self.start_turn_time_stamp > 2 and self.start_turn:
                            self.start_turn = False
                        if not self.start_turn:
                            self.pid.SetPoint = 100
                            self.pid.update(lane_x)
                            twist.angular.z = twist.linear.x * math.tan(misc.set_range(self.pid.output, -0.1, 0.1)) / 0.213
                        else:
                            twist.angular.z = 0.15 * math.tan(-0.5) / 0.213
                        self.acker_pub.publish(twist)
                else:
                    self.pid.clear()

                # Draw detected objects.
                if self.objects_info != []:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )
            else:
                rospy.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.dispaly:
                cv2.imshow('result', bgr_image)
                key = cv2.waitKey(1)
                if key != -1:
                    self.is_running = False
            self.result_publisher.publish(cv2_image2ros(bgr_image))
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.acker_pub.publish(Twist())

    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
            # Reset turning flags if no objects detected.
            self.turn_right = False
            self.no_left_turn = False
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))
                if class_name == 'yellow_light':  
                    if center[1] > min_distance:
                        min_distance = center[1]
                elif class_name == 'speed_limit_5':
                    if not self.speed_limit_detected:
                        self.start_slow_down = True
                        self.count_slow_down = rospy.get_time()
                        self.speed_limit_detected = True
                    else:
                        self.count_slow_down = rospy.get_time()
                elif class_name == 'speed_limit_lift':
                    if self.speed_limit_detected:
                        self.start_slow_down = False
                        self.speed_limit_detected = False
                elif class_name == 'stop':
                    if not self.stop_detected:
                        self.stop_detected = True
                        self.stop_start_time = rospy.get_time()
                        rospy.loginfo("stopping for 5 seconds")

                elif class_name == 'turn_right':
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 10:
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'no_left_turn':
                    self.no_left_turn = True
                elif class_name == 'parking':
                    self.park_x = center[0]
                elif class_name in ['red_light', 'green_light', 'yellow_light']:
                    self.traffic_signs_status = i
            self.crosswalk_distance = min_distance

if __name__ == "__main__":
    SelfDrivingNode('self_driving')

