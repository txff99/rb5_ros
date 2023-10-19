#!/usr/bin/env python

import math
import time
import rospy
from sensor_msgs.msg import Joy

class PathplanNode:
    def __init__(self, interval=0.1, k_v=0.2, k_w=0.3, robot_vx_min=0.07, robot_vx_max=0.8, robot_vy_min=0.07, robot_vy_max=0.8, robot_w_min=0.2, robot_w_max=1.0):
        self.pub_joy = rospy.Publisher("/joy", Joy, queue_size=1)
        self.waypoints = [(0,0,0),(-1,0,0),(-1,1,1.57),(-2,1,0),(-2,2,-1.57),(-1,1,-0.78),(0,0,0)]
        self.interval = interval
        self.k_v = k_v
        self.k_w = k_w
        self.robot_vx_min = robot_vx_min
        self.robot_vx_max = robot_vx_max
        self.robot_vy_min = robot_vy_min
        self.robot_vy_max = robot_vy_max
        self.robot_w_min = robot_w_min
        self.robot_w_max = robot_w_max

    def dist(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def is_near_dest(self, now_x, now_y, now_theta, end_x, end_y, end_theta):
        return self.dist(now_x, now_y, end_x, end_y) < 0.01 and abs(end_theta-now_theta) < 0.01

    def get_global_v_w(self, now_x, now_y, now_theta, end_x, end_y, end_theta):
        vx = (end_x-now_x) * self.k_v
        vy = (end_y-now_y) * self.k_v
        w = (end_theta - now_theta) * self.k_w
        return vx, vy, w
    
    def calc_v_related_to_robot(self, vx, vy, now_theta):
        robot_vx = math.cos(-now_theta) * vx - math.sin(-now_theta) * vy
        robot_vy = math.sin(-now_theta) * vx + math.cos(-now_theta) * vy
        return robot_vx, robot_vy

    def calc_v_related_to_global(self, robot_vx, robot_vy, now_theta):
        vx = math.cos(now_theta) * robot_vx - math.sin(now_theta) * robot_vy
        vy = math.sin(now_theta) * robot_vx + math.cos(now_theta) * robot_vy
        return vx, vy
    
    def estimate_next_pose(self, now_x, now_y, now_theta, vx, vy, w):
        if w == 0:
            new_x = now_x + vx * self.interval
            new_y = now_y + vy * self.interval
            new_theta = now_theta
            return new_x, new_y, new_theta
        v = math.sqrt(vx**2 + vy**2)
        R = abs(v/w)
        nx = -vy if w>=0 else vy
        ny = vx if w>=0 else -vx
        norm_n = math.sqrt(nx**2 + ny**2)
        nx /= norm_n / R
        ny /= norm_n / R
        circle_x = now_x + nx
        circle_y = now_y + ny
        rad_in_interval = self.interval * w
        new_x = circle_x + math.cos(rad_in_interval) * (-nx) - math.sin(rad_in_interval) * (-ny)
        new_y = circle_y + math.sin(rad_in_interval) * (-nx) + math.cos(rad_in_interval) * (-ny)
        new_theta = now_theta + rad_in_interval
        return new_x, new_y, new_theta
    
    def clip(self, value, min_value, max_value):
        sign = 1.0 if value >= 0 else -1.0
        return sign * max(min(abs(value), max_value), min_value)

    def run(self):
        for idx in range(len(self.waypoints) - 1):
            start_waypoint = self.waypoints[idx]
            now_x, now_y, now_theta = start_waypoint
            dest_waypoint = self.waypoints[idx+1]
            end_x, end_y, end_theta = dest_waypoint
            while self.is_near_dest(now_x, now_y, now_theta, end_x, end_y, end_theta) == False:
                vx, vy, w = self.get_global_v_w(now_x, now_y, now_theta, end_x, end_y, end_theta)
                print("pose: ({}, {}, {}), v: ({}, {}), w:{}".format(now_x, now_y, now_theta, vx, vy, w))
                robot_vx, robot_vy = self.calc_v_related_to_robot(vx, vy, now_theta)
                robot_vx = self.clip(robot_vx, 0.08, 0.8)
                robot_vy = self.clip(robot_vy, 0.08, 0.8)
                vx, vy = self.calc_v_related_to_global(robot_vx, robot_vy, now_theta)
                now_x, now_y, now_theta = self.estimate_next_pose(now_x, now_y, now_theta, vx, vy, w)
                self.send_control_signal(robot_vx, robot_vy, w)
                time.sleep(self.interval)
            print("reach: {}".format(self.waypoints[idx+1]))
            self.stop()
            time.sleep(self.interval)

    def send_control_signal(self, robot_vx, robot_vy, w):
        plan_msg = Joy()
        plan_msg.axes = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
        plan_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
        # TODO
        plan_msg.axes[0] = 2.509 * robot_vy + 0.0595
        plan_msg.axes[1] = 2.509 * robot_vx + 0.0595
        plan_msg.axes[2] = w*0.6/0.785
        print("robot_vx: {}, robot_vy: {}, w: {}, axes: ({}, {}, {})".format(robot_vx, robot_vy, w, plan_msg.axes[0], plan_msg.axes[1], plan_msg.axes[2]))
        # self.pub_joy.publish(plan_msg)

    def stop(self):
        plan_msg = Joy()
        plan_msg.axes = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]
        plan_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.pub_joy.publish(plan_msg)

if __name__ == "__main__":
    path_plan_node = PathplanNode()
    # rospy.init_node("joy")
    # rospy.on_shutdown(path_plan_node.stop)
    path_plan_node.run()