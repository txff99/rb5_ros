#!/usr/bin/env python

import math
import time
import rospy
from hw1.msg import RobotInfo
from hw2.msg import RobotPose
from threading import Lock

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def clip(value, min_value, max_value):
    if value == 0.0:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * max(min(abs(value), max_value), min_value)

class PIDController:
    def __init__(self, k_p, k_i, k_d, interval, integral_limit):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.interval = interval
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.last_error = 0.0
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def calc_output(self, err):
        self.integral += err
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        if self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        diff = err - self.last_error
        self.last_error = err
        return self.k_p * err + (self.k_i * self.interval) * self.integral + (self.k_d / self.interval) * diff
    

class PathplannerNode:
    def __init__(self, interval=0.1, kp_v=0.5, ki_v=0.1, kd_v=0.0, kp_w=0.6, ki_w=0.1, kd_w=0.00, robot_vx_min=0.05, robot_vx_max=0.2, robot_vy_min=0.05, robot_vy_max=0.2, robot_w_min=0.6, robot_w_max=1.2):
        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)
        self.waypoints = [(0,0,0),(1,0,0),(1,2,3.14),(0,0,0)]
        # self.waypoints = self.waypoints[0:2]
        print("waypoints: {}".format(self.waypoints))

        self.real_pose_mutex = Lock()
        self.real_pose = (0.0, 0.0, 0.0)
        self.state = 0

        self.interval = interval
        self.v_controller = PIDController(kp_v, ki_v, kd_v, interval, 1.0)
        self.w_controller = PIDController(kp_w, ki_w, kd_w, interval, math.pi)

        self.robot_vx_min = robot_vx_min
        self.robot_vx_max = robot_vx_max
        self.robot_vy_min = robot_vy_min
        self.robot_vy_max = robot_vy_max
        self.robot_w_min = robot_w_min
        self.robot_w_max = robot_w_max

        self.now_x = 0.0
        self.now_y = 0.0
        self.now_theta = 0.0
        self.end_x = 0.0
        self.end_y = 0.0
        self.end_theta = 0.0

        self.r_err = 0.0
        self.theta_err = 0.0

        self.stable_cnt = 0

    def is_near_dest(self):
        return self.r_err < 0.05 and abs(self.theta_err) < 0.10

    def calc_pid_controller_output(self):
        v = self.v_controller.calc_output(self.r_err)
        w = self.w_controller.calc_output(self.theta_err)
        return v, w
    
    def calc_v_related_to_robot(self, vx, vy, now_theta):
        robot_vx = math.cos(-now_theta) * vx - math.sin(-now_theta) * vy
        robot_vy = math.sin(-now_theta) * vx + math.cos(-now_theta) * vy
        return robot_vx, robot_vy

    def calc_v_related_to_global(self, robot_vx, robot_vy, now_theta):
        vx = math.cos(now_theta) * robot_vx - math.sin(now_theta) * robot_vy
        vy = math.sin(now_theta) * robot_vx + math.cos(now_theta) * robot_vy
        return vx, vy
    
    def update_real_pose(self, pose_msg):
        if pose_msg.info == 'no tag detected':
            self.state = 0
            return
        x = pose_msg.x
        y = pose_msg.y
        theta = pose_msg.yaw

        with self.real_pose_mutex:
            if self.state == 0:
                self.state = 1
                self.real_pose = (x, y, theta)
            elif self.state == 1:
                if (abs(self.real_pose[0] - x) > 100 * self.interval * (self.robot_vx_max**2 + self.robot_vy_max**2) or \
                    abs(self.real_pose[1] - y) > 100 * self.interval * (self.robot_vx_max**2 + self.robot_vy_max**2)):
                    print("anamoly detected. x: {}, y: {}, theta: {}".format(x,y,theta))
                    self.state = -1
                else:
                    self.real_pose = (x, y, theta)
            elif self.state == -1:
                self.state = 1
                self.real_pose = (x, y, theta)

    
    def get_real_pose(self):
        with self.real_pose_mutex:
            detected = (self.state == 1)
            self.now_x, self.now_y, self.now_theta = self.real_pose
        while detected == False:
            self.send_robot_info(0.0, 0.0, 0.9)
            time.sleep(self.interval)
            with self.real_pose_mutex:
                detected = (self.state == 1)
                self.now_x, self.now_y, self.now_theta = self.real_pose
        print("pose: ({}, {}, {})".format(self.now_x, self.now_y, self.now_theta))

    def calc_vx_vy(self, now_x, now_y, end_x, end_y, v):
        rx = end_x - now_x
        ry = end_y - now_y
        r = math.sqrt(rx**2 + ry**2)
        vx = rx / r * v
        vy = ry / r * v
        return vx, vy

    def is_reached_waypoint(self, n_interval=2):
        if self.is_near_dest() == False:
            self.stable_cnt = 0
            return False
        else:
            self.stable_cnt += 1
            reached = self.stable_cnt >= n_interval
            if reached:
                self.stable_cnt = 0
            return reached

    def update_err(self):
        self.r_err = dist(self.now_x, self.now_y, self.end_x, self.end_y)
        self.theta_err = self.end_theta - self.now_theta
        if self.theta_err > math.pi:
            self.theta_err -= 2*math.pi
        elif self.theta_err < -math.pi:
            self.theta_err += 2*math.pi
        print("x_err: {}, y_err: {}, theta_err: {}".format(self.end_x-self.now_x, self.end_y-self.now_y, self.theta_err))

    def run(self):
        for idx in range(len(self.waypoints) - 1):
            self.end_x, self.end_y, self.end_theta = self.waypoints[idx+1]
            self.v_controller.reset()
            self.w_controller.reset()

            self.get_real_pose()
            self.update_err()
            while self.is_reached_waypoint() == False:
                v, w = self.calc_pid_controller_output()
                vx, vy = self.calc_vx_vy(self.now_x, self.now_y, self.end_x, self.end_y, v)
                robot_vx, robot_vy = self.calc_v_related_to_robot(vx, vy, self.now_theta)
                robot_vx = clip(robot_vx, self.robot_vx_min, self.robot_vx_max)
                robot_vy = clip(robot_vy, self.robot_vy_min, self.robot_vy_max)
                w = clip(w, self.robot_w_min, self.robot_w_max)
                print("v: ({}, {}), w:{}".format(robot_vx, robot_vy, w))
                self.send_robot_info(robot_vx, robot_vy, w)
                time.sleep(self.interval)

                self.get_real_pose()
                self.update_err()
            
            self.stop()
            print("reach: {}".format(self.waypoints[idx+1]))
            time.sleep(1)

    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    def stop(self):
        self.send_robot_info(0.0, 0.0, 0.0)

if __name__ == "__main__":
    path_planner_node = PathplannerNode()
    rospy.init_node("path_planner")
    rospy.Subscriber("/robot_pose", RobotPose, path_planner_node.update_real_pose, queue_size=1)
    rospy.on_shutdown(path_planner_node.stop)
    time.sleep(1)
    path_planner_node.run()
