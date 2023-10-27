#!/usr/bin/env python

import math
import rospy
import tf
from april_detection.msg import AprilTagDetectionArray
from geometry_msgs.msg import Pose

class RobotPoseNode:
    def __init__(self):
        self.pub_pose = rospy.Publisher("/robot_pose", Pose, queue_size=1)
        self.apriltag_original_pose = {}
    
    def calc_pose(self, april_detection_array_msg):
        q = april_detection_array_msg.detections[0].pose.orientation
        p = april_detection_array_msg.detections[0].pose.position
        dist = math.sqrt(p.x**2 + p.z**2)
        quaternion = (q.x, q.y, q.z, q.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        print("dist: {}".format(dist))
        print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))
    
    def send_pose(self, pose):
        pose_msg = Pose()
        self.pub_pose.publish(pose_msg)


if __name__ == "__main__":
    robot_pose_node = RobotPoseNode()
    rospy.init_node('robot_pose_parser')
    detection_array_msg = rospy.wait_for_message("/apriltag_detection_array", AprilTagDetectionArray, timeout=1.0)
    # print(detection_array_msg)
    robot_pose_node.calc_pose(detection_array_msg)
