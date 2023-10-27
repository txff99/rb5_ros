#!/usr/bin/env python

import os 
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

if __name__ == "__main__":
    directory = "/root/rb5_ws/src/rb5_ros/hw2/imgs"
    os.chdir(directory)
    bridge = CvBridge()
    rospy.init_node("image_reader")
    index = 0
    while True:
        text = raw_input("")
        if text == "":
            break
        else:
            raw_image_msg = rospy.wait_for_message("/camera_0", Image, timeout=1.0)
            cv_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='bgr8')
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3000)
            filename = str(index)+".png"
            index += 1
            cv2.imwrite(filename, cv_image)