#!/usr/bin/env python
"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
import rospy
from sensor_msgs.msg import Joy
import numpy as np
import time

# from key_parser import get_key, save_terminal_settings, restore_terminal_settings

class PathplanNode:
    def __init__(self):
        self.pub_joy = rospy.Publisher("/path_plan", Joy, queue_size=1)
        self.settings = save_terminal_settings()


    def run(self):

        # read waypoints
        wp = []
        with open("./waypoints.txt") as f:
            for i in f.readlines():
                wp.append([float(x) for x in i[:-1].split(',')])
        # print(wp)

        x = [x[0] for x in wp]
        y = [x[1] for x in wp]
        theta_t = [x[2] for x in wp]
        # theta_sin = [0.1*np.sin(x[2]) for x in wp]
        # plot a map
                # length_includes_head=True)
        targets = [x,y,theta_t]

        # def traverse(targets):

        # params
        timestamp = 0.1
        k_r = 6/30
        k_a = 8/30
        k_b = -0.5/30

        pts_x,pts_y,pts_theta = targets
        x_p = pts_x[0]
        y_p = pts_y[0]
        theta_p = pts_theta[0]

        target_length = len(pts_x)

        for i in range(1,target_length):
            #target point
            x_t = pts_x[i]
            y_t = pts_y[i]
            theta_t = pts_theta[i]
            # print(y_t)
            # print(y_p)
            #compute relative position
            alpha = np.arctan2(y_t-y_p,x_t-x_p)-theta_p
            beta = -np.arctan2(y_t-y_p,x_t-x_p)+theta_t
            r = np.sqrt((y_t-y_p)**2+(x_t-x_p)**2)
            # print(np.arctan2(y_t-y_p,x_t-x_p))
            # print(theta_p)
            # print(alpha)
            # print(beta)
            # print(theta_t)
            while(not ((x_p-x_t)**2+(y_p-y_t)**2<0.01)):
                # get new status
                alpha = np.arctan2(np.sin(alpha),np.cos(alpha))
                beta = np.arctan2(np.sin(beta),np.cos(beta))
                r1 = r - k_r*r*np.cos(alpha)*timestamp
                alpha1 = alpha + (k_r*np.sin(alpha)-k_a*alpha-k_b*beta)*timestamp 
                beta1 = beta - k_r*np.sin(alpha)*timestamp
                alpha = alpha1
                r = r1
                beta = beta1
                
                x_p = x_t - r*np.cos(theta_t-beta)
                y_p = y_t - r*np.sin(theta_t-beta)
                theta_p =  (theta_t - beta) - alpha

                # control flow
                v = k_r*r
                w = k_a*alpha+k_b*beta
                
                self.pub_joy.publish(v,w)
                time.sleep(0.1)

            while abs(theta_p-theta_t)>0.3:
                if w!=0 :
                    self.pub_joy.publish(0,w/abs(w))
                else: self.pub_joy.publish(0,1)
            # publish joy
    
        self.stop()

    def plan_msg(self, v, w):

                flag = True
                plan_msg = Joy()
                plan_msg.axes = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
                plan_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
                #normalize
                plan_msg.axes[1] = v/0.3
                plan_msg.axes[2] = w/0.785
                return plan_msg, flag
            

    def stop(self):
        restore_terminal_settings(self.settings)


if __name__ == "__main__":
    path_plan_node = PathplanNode()
    rospy.init_node("path_plan")
    path_plan_node.run()
