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
import math

# from key_parser import get_key, save_terminal_settings, restore_terminal_settings

class PID_plan_Node:
    def __init__(self,robot_vx_min=0.1, robot_vx_max=0.3, robot_vy_min=0.1, robot_vy_max=0.3, robot_w_min=0.7, robot_w_max=2.5):
        # pass
        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)
        self.timestamp = 0.1
        self.k_r = 6.0/30
        self.k_a = 8.0/30
        self.k_b = -0.5/30
        self.current = []
        self.target = []
        self.robot_v_min = robot_vy_min
        self.robot_v_max = robot_vy_max
        self.robot_w_max = robot_w_max
        self.robot_w_min = robot_w_min

        wp = []
        with open("/root/rb5_ws/src/rb5_ros/hw1/src/waypoints.txt") as f:
            for i in f.readlines():
                wp.append([float(x) for x in i[:-1].split(',')])
        self.wp = [[x[0] for x in wp], [x[1] for x in wp], [x[2] for x in wp]]
    
    def set_init_params(self):
        self.k_r = 6.0/30
        self.k_a = 8.0/30
        self.k_b = -0.5/30

    def clip(self, value, min_value, max_value):
        if value == 0.0:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * max(min(abs(value), max_value), min_value)

        
    def compute_next_status(self,alpha,beta,r):
        r = r - self.k_r*r*np.cos(alpha)*self.timestamp
        alpha = alpha + (self.k_r*np.sin(alpha)-self.k_a*alpha-self.k_b*beta)*self.timestamp 
        beta = beta - self.k_r*np.sin(alpha)*self.timestamp
        return alpha,beta,r
        
    def control(self,r,alpha,beta):
        # control flow
        v = self.k_r*r
        w = self.self.k_a*alpha+self.k_b*beta
        v = self.clip(v, self.robot_v_min, self.robot_v_max)
        w = self.clip(w, self.robot_w_min, self.robot_w_max)
        
        # adjust params if exceed threshold
        self.k_r = v/r
        if v==self.robot_v_min:
            self.k_a = (w-k_b*beta)/alpha
        
        return v,w

    def compute_angle(self):
        target_x,target_y,target_theta = self.target
        current_x,current_y,current_theta = self.current

        alpha = np.arctan2(target_y-current_y,target_x-current_x)-current_theta
        beta = -np.arctan2(target_y-current_y,target_x-current_x)+target_theta
        r = np.sqrt((target_y-current_y)**2+(target_x-current_x)**2)
        
        # constrain alpha and beta to be -pi to pi
        alpha = np.arctan2(np.sin(alpha),np.cos(alpha))
        beta = np.arctan2(np.sin(beta),np.cos(beta))
        return alpha,beta,r

    def compute_current_position(self,alpha,beta,r)
        target_theta = self.target[2]
        current_x = target_x - r*np.cos(target_theta-beta)
        current_y = target_y - r*np.sin(target_theta-beta)
        current_theta =  (target_theta - beta) - alpha
        self.current = [current_x,current_y,current_theta]

    def move_to_target(self):

        alpha, beta, r = self.compute_angle()
        # move to target
        while(not (r<0.01)):
            # get new status
            alpha,beta,r = self.compute_next_status(alpha,beta,r)
            self.compute_current_position(alpha,beta,r)
            v,w = self.control(r,alpha,beta)
            plan_msg = self.send_robot_info(0,v,w)
            print("v:{} w:{}".format(v,w))
            self.pub_robot_info.publish(plan_msg)
            time.sleep(0.1)
        print("position{} reached".format(i))

        #get correct pose
        current_theta = self.current[2]
        target_theta =self.target[2]
        while abs(current_theta-target_theta)>0.3:
            current_theta = np.arctan2(np.sin(current_theta),np.cos(current_theta))
            plan_msg = self.send_robot_info(0,0,w/abs(w))
            current_theta = current_theta + w/abs(w)*self.timestamp
            self.pub_robot_info.publish(plan_msg)
            time.sleep(0.1)
        print("pose{} reached".format(i))
        self.current[2] = current_theta
        self.set_init_params()

    def run(self):
        pts_x,pts_y,pts_theta = self.wp
        current_x = pts_x[0]
        current_y = pts_y[0]
        current_theta = pts_theta[0]

        target_length = len(pts_x)
        self.current = [current_x,current_y,current_theta]

        for i in range(1,target_length):
            self.target = [pts_x[i],pts_y[i],pts_theta[i]]
            self.move_to_target()
        print("finished")
        self.stop()

                
    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    # def plan(self, v, w):
    #     w = w*0.6/0.785
    #     if w < 0.15 and v < 0.1:w=0.15
    #     v = (4.634+100*v)/43.5*0.8
    #     plan_msg = Joy()
    #     plan_msg.axes = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
    #     plan_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
    #     #normalize
    #     plan_msg.axes[1] = v #y =0.435x-4.634
    #     plan_msg.axes[2] = w  # 0.55*w + 0.188 # y=0.0363x-0.342
    #     return plan_msg

    def stop():
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = 0.0
        robot_info_msg.vy = 0.0
        robot_info_msg.wz = 0.0
        self.pub_robot_info.publish(robot_info_msg)

if __name__ == "__main__":
    path_plan_node = PID_plan_Node()
    rospy.init_node("pid_planner")
    rospy.on_shutdown(path_plan_node.stop)
    path_plan_node.run()
