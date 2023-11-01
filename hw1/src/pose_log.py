import sys
import rospy
from hw1.msg import RobotInfo
from hw2.msg import RobotPose
import numpy as np
import time
import math

class Pose_Log_Node:
    def __init__(self):
        self.buffer = []

    def msg_callback(self,msg):
        self.buffer.append(msg)

    def write_to_log(self,):
        with open('log.txt', 'w') as f:
            for pose in self.buffer:

if __name__ == "__main__":
    pose_log_node = Pose_Log_Node()
    time.sleep(1)
    # robot_ctrl_node.rotate(1)
    # time.sleep(20)
    rospy.init_node('Pose_Log_Node')
    rospy.Subscriber('/combined_pose', RobotPose, pose_log_node.msg_callback, queue_size=1) 
    rospy.spin()


        


