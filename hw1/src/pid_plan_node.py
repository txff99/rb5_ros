#!/usr/bin/env python
import sys
import rospy
from hw1.msg import RobotInfo
from hw2.msg import RobotPose
import numpy as np
import time
import threading
import math

KR_DEFAULT = 6.0/30
KA_DEFAULT = 8.0/30
KB_DEFAULT = -0.5/30

class PID_plan_Node:
    def __init__(self,robot_vx_min=0.1, robot_vx_max=0.12, robot_vy_min=0.1, robot_vy_max=0.5, robot_w_min=0.5, robot_w_max=1.2):
        # pass
        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)
        self.pub_combined_pose = rospy.Publisher("/combined_pose", RobotPose, queue_size=1)
        self.lock = threading.Lock()
        self.timestamp = 0.1
        self.k_r = 6.0/30
        self.k_a = 15.0/10
        self.k_b = -0.5/10
        self.q1 = 0.9
         # weight of current pose
        self.current = []
        self.target = []
        self.robot_v_min = robot_vx_min
        self.robot_v_max = robot_vx_max
        self.robot_w_max = robot_w_max
        self.robot_w_min = robot_w_min
        self.msg_buffer = []

        wp = []
        with open("/root/rb5_ws/src/rb5_ros/hw4/src/waypoints.txt") as f:
            for i in f.readlines():
                wp.append([float(x) for x in i[:-1].split(',')])
        self.wp = [[x[0] for x in wp], [x[1] for x in wp], [x[2] for x in wp]]
    
    # def set_init_params(self):
    #     self.k_r = 6.0/10
    #     self.k_a = 8.0/10
    #     self.k_b = -0.5/10

    def clip(self, value, min_value, max_value):
        if value == 0.0:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * max(min(abs(value), max_value), min_value)

    def compute_next_status(self,v,w,alpha,beta,r):
        r = r - v*np.cos(alpha)*self.timestamp
        alpha = alpha + (self.k_r*np.sin(alpha)-w)*self.timestamp 
        beta = beta - self.k_r*np.sin(alpha)*self.timestamp
        return alpha,beta,r
        
    def control(self,r,alpha,beta):
        # control flow
        v = self.k_r*r
        w = self.k_a*alpha+self.k_b*beta
        v = self.clip(v, self.robot_v_min, self.robot_v_max)
        
        # adjust params if exceed threshold
        # self.k_r = v/r#min(v/r,KR_DEFAULT)
        if  w != 0.0:
            w = self.clip(w, self.robot_w_min, self.robot_w_max)  
            # if v==self.robot_v_min :
            # self.k_a = (w-self.k_b*beta)/alpha
        
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
        print("alpha:{} beta:{}".format(alpha,beta))
        return alpha,beta,r

    def compute_current_position(self,alpha,beta,r):
        target_x,target_y,target_theta = self.target
        current_x = target_x - r*np.cos(target_theta-beta)
        current_y = target_y - r*np.sin(target_theta-beta)
        current_theta =  (target_theta - beta) - alpha
        self.current = [current_x,current_y,current_theta]
        self.correct_pose()

    def correct_pose(self):
        self.lock.acquire()
        print(self.current)
        if len(self.msg_buffer):
            self.current = [self.q1*self.current[i]+(1-self.q1)*self.msg_buffer[i] for i in range(3)]
            self.msg_buffer = []
        self.lock.release()

    def dist(self):
        target_x,target_y,_ = self.target
        current_x,current_y,_ = self.current
        return np.sqrt((target_y-current_y)**2+(target_x-current_x)**2)

    def angle_diff(self):
        current_theta = self.current[2]
        current_theta = np.arctan2(np.sin(current_theta),np.cos(current_theta))
        target_theta = self.target[2]
        # print(current_theta-target_theta)
        return current_theta-target_theta
        # self.pub_combined_pose()

    def move_to_target(self,ith_target):
        # move to target
        while(not (self.dist()<0.01)):
            print("pose{} reached".format(ith_target-1))
            alpha, beta, r = self.compute_angle()
            v,w = self.control(r,alpha,beta)
            print("v:{},w:{}".format(v,w))
            # get new status
            alpha,beta,r = self.compute_next_status(v,w,alpha,beta,r)
            self.compute_current_position(alpha,beta,r)
            self.send_robot_info(v,0,w)
            # print("v:{} w:{}".format(v,w))
            time.sleep(0.1)

        #get correct pose
        # while abs(self.angle_diff())>0.1:
        #     print("position{} reached".format(ith_target))
        #     self.correct_pose()
        #     self.send_robot_info(0,0,1.256*w/abs(w))
        #     self.current[2] = self.current[2] + 1.2*w/abs(w)*self.timestamp
        #     time.sleep(0.1)
        time.sleep(1)
        # self.current[2] = current_theta

    def run(self):
        pts_x,pts_y,pts_theta = self.wp
        current_x = pts_x[0]
        current_y = pts_y[0]
        current_theta = pts_theta[0]

        target_length = len(pts_x)
        self.current = [current_x,current_y,current_theta]

        for i in range(1,target_length):
            self.target = [pts_x[i],pts_y[i],pts_theta[i]]
            self.move_to_target(i)
        print("finished")
        self.stop()

    def send_combined_pose(self):
        combined_pose_msg = RobotPose()
        combined_pose_msg.x =self.current[0]
        combined_pose_msg.y =self.current[1]
        combined_pose_msg.yaw =self.current[2]
        self.pub_combined_pose.publish(combined_pose_msg)

    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    def msg_callback(self, pose_msg):
        # translate to x,y,theta
        # print(pose_msg)
        self.lock.acquire()
        if pose_msg.info != 'no tag detected':
            x = pose_msg.x
            y = pose_msg.y
            z = pose_msg.z
            roll = pose_msg.roll
            pitch = pose_msg.pitch
            yaw = pose_msg.yaw
            self.msg_buffer = [x,y,yaw]
        else: self.msg_buffer = []
        self.lock.release()

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

    def stop(self):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = 0.0
        robot_info_msg.vy = 0.0
        robot_info_msg.wz = 0.0
        self.pub_robot_info.publish(robot_info_msg)

if __name__ == "__main__":
    path_plan_node = PID_plan_Node()
    rospy.init_node("pid_planner")
    rospy.Subscriber("/robot_pose", RobotPose, path_plan_node.msg_callback, queue_size=1) 
    
    rospy.on_shutdown(path_plan_node.stop)
    path_plan_node.run()
