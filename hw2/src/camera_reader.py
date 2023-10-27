#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

if __name__ == "__main__":
    bridge = CvBridge()
    rospy.init_node("image_reader")
    raw_image_msg = rospy.wait_for_message("/camera_0", Image, timeout=1.0)
    print(raw_image_msg.encoding)
    cv_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='bgr8')
    (rows,cols,channels) = cv_image.shape
    print("rows: {}, cols: {}, channels: {}".format(rows, cols, channels))
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(0)