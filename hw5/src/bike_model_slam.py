#!/usr/bin/env python
import sys
import numpy as np
import time
import threading
import math
from numpy import sin, cos, arctan2
import tf
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import rospy
from hw1.msg import RobotInfo
from slam_node import EKF, normalize_angle
from april_detection.msg import AprilTagDetectionArray


KR_DEFAULT = 6.0/30
KA_DEFAULT = 8.0/30
KB_DEFAULT = -0.5/30

class PID_plan_Node:
    def __init__(self,robot_vx_min=0.08, robot_vx_max=0.1, robot_vy_min=0.1, robot_vy_max=0.5, robot_w_min=0.4, robot_w_max=0.6):
        # pass
        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)
        self.mutex = threading.Lock()
        self.timestamp = 0.1
        self.k_r = 6.0/30
        self.k_a = 15.0/10
        self.k_b = -0.5/10
        self.robot_v_min = robot_vx_min
        self.robot_v_max = robot_vx_max
        self.robot_w_max = robot_w_max
        self.robot_w_min = robot_w_min

        self.known_landmark_ids = []
        self.landmark_id2state_vector_id = {}
        self.landmarks_detection_results = {}
        self.walls = []

        self.target = np.array([])
        self.current = np.array([0.0, 0.0, 0.0])
        self.landmark_poses = np.array([])
        P0 = np.eye(3, dtype=float)
        P0[0, 0] = 0.0
        P0[1, 1] = 0.0
        P0[2, 2] = 0.0
        Q = np.eye(3, dtype=float) * 0.01 
        R = np.eye(3, dtype=float) * 0.01 * 4
        self.ekf = EKF(self.current, P0, Q, R)

        self.interval = 0.1
        self.collect_history_data = True
        self.robot_history_poses = [self.current]
        self.landmarks_history_positions = []

    
        wp = []
        with open("/root/rb5_ws/src/rb5_ros/hw3/src/waypoints.txt") as f:
            for i in f.readlines():
                wp.append([float(x) for x in i[:-1].split(',')])
        self.wp = [[x[0] for x in wp], [x[1] for x in wp], [x[2] for x in wp]]

    def read_landmark_detection(self, april_detection_array_msg):
        with self.mutex:
            self.landmarks_detection_results = {}
            for detection in april_detection_array_msg.detections:
                landmark_id = detection.id
                if landmark_id>10:
                    continue
                position = detection.pose.position
                camera_x = position.x
                camera_z = position.z
                quaternion = detection.pose.orientation 
                _, pitch, _ = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
                self.landmarks_detection_results[landmark_id] = [camera_z, -camera_x, -normalize_angle(pitch)]

    def clip(self, value, min_value, max_value):
        if value == 0.0:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * max(min(abs(value), max_value), min_value)

    def get_lankmark_cov_mat_by_id(self, landmark_id):
        cov_matrix = self.ekf.get_cov_matrix()
        idx = self.landmark_id2state_vector_id[landmark_id]
        return cov_matrix[3+idx*3:3+(idx+1)*3, 3+idx*3:3+(idx+1)*3]
    # def estimate_next_pose(self,v,w,alpha,beta,r):
    #     r = r - v*np.cos(alpha)*self.timestamp
    #     alpha = alpha + (self.k_r*np.sin(alpha)-w)*self.timestamp 
    #     beta = beta - self.k_r*np.sin(alpha)*self.timestamp
    #     return alpha,beta,r
        
    def control(self,r,alpha,beta):
        # control flow
        v = self.k_r*r
        w = self.k_a*alpha+self.k_b*beta
        v = self.clip(v, self.robot_v_min, self.robot_v_max)
        
        # adjust params if exceed threshold
        # self.k_r = v/r#min(v/r,KR_DEFAULT)
        if  w != 0.0:
            w = self.clip(w, self.robot_w_min, self.robot_w_max)  
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
        # print("alpha:{} beta:{}".format(alpha,beta))
        return alpha,beta,r

    def get_lankmark_pose_by_id(self, landmark_id):
        landmark_poses = np.reshape(self.ekf.get_landmark_pose(), (-1, 3))
        return landmark_poses[self.landmark_id2state_vector_id[landmark_id]]

    def correct_pose(self,robot_input,ith_target):
        observations = np.array([])
        observed_landmark_ids = []
        with self.mutex:
            for landmark_id, landmark_pose in self.landmarks_detection_results.items():
                if landmark_id not in self.known_landmark_ids:
                    self.landmark_id2state_vector_id[landmark_id] = len(self.known_landmark_ids)
                    self.known_landmark_ids.append(landmark_id)
                    self.ekf.extend(1)
                observations = np.append(observations, np.array(landmark_pose))
                observed_landmark_ids.append(landmark_id)

        self.ekf.update(robot_input, observations, [self.landmark_id2state_vector_id[landmark_id] for landmark_id in observed_landmark_ids], self.interval)
        self.current = self.ekf.get_robot_pose()
        self.landmark_poses = np.reshape(self.ekf.get_landmark_pose(), (-1, 3))
        if self.collect_history_data:
            self.robot_history_poses.append(self.current)
            self.landmarks_history_positions.append(self.landmark_poses[:,:2])
        self.update_map()

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

    def update_map(self):
        threshold = 1
        wall_num = len(self.walls)
        # if len(self.wall)==0:
        #     for lm in self.landmark_poses:
        #         lm_x,lm_y,angle = lm
        #         for x,y,a in lines:
        #             if abs(a-angle)<threshold:
        #                 pass
        #             else:
        #                 lines.append((lm_x,lm_y,angle))
        #     self.walls = lines
        #     return 
        lines = [[] for _ in range(wall_num)]
        new_lines = [[]]
        # print(self.landmark_poses)
        for lm in self.landmark_poses:
            lm_x,lm_y,lm_a = lm
            # lm_a = angle
            flag = False
            for i in range(wall_num):
                # angle_w = np.arctan2(a,1)-np.pi/2
                x,y,theta = self.walls[i]
                if abs(theta-lm_a)>5/3*np.pi:
                    theta = abs(theta)
                    lm_a = abs(lm_a)
                
                if abs(theta-lm_a)<threshold:
                    flag=True
                    lines[i].append((lm_x,lm_y,lm_a))
                    break
            if flag == False:
                if len(new_lines[0])==0:
                    new_lines[0].append((lm_x,lm_y,lm_a))
                else:
                    for j in range(len(new_lines)):
                        line = new_lines[j][0]
                        l_x,l_y,l_angle = line
                        if abs(l_angle-lm_a)>5/3*np.pi:
                            l_angle = abs(l_angle)
                            lm_a = abs(lm_a)
                        if abs(l_angle-lm_a)<threshold:
                            new_lines[j].append((lm_x,lm_y,lm_a))
                            flag = True
                            break
                    if flag == False:
                        new_lines.append([(lm_x,lm_y,lm_a)])
        updated = []  
        for i in range(len(lines)):
            line = lines[i]
            if len(line)==0:
                updated.append(self.walls[i])
            else:
                line_x = np.mean([x for x,y,theta in line])
                line_y = np.mean([y for x,y,theta in line])
                line_theta = np.mean([theta for x,y,theta in line])
                updated.append((line_x,line_y,line_theta))
        for line in new_lines:
            if len(line)!=0:
                line_x = np.mean([x for x,y,theta in line])
                line_y = np.mean([y for x,y,theta in line])
                line_theta = np.mean([theta for x,y,theta in line])
                updated.append((line_x,line_y,line_theta))
        
        # print(new_lines)
        print(len(self.landmark_poses))
        self.walls = updated
        print(self.walls)



    def move_to_target(self,ith_target):
        # move to target
        while(not (self.dist()<0.1)):
            print("pose{} reached".format(ith_target-1))
            alpha, beta, r = self.compute_angle()
            v,w = self.control(r,alpha,beta)
            # print("current_pose:{}".format(self.current))
            # print("target_pose:{}".format(self.target))
            # print("v:{},w:{}".format(v,w))
            # get new status
            # alpha,beta,r = self.estimate_next_pose(v,w,alpha,beta,r)
            self.correct_pose([v,0,w],ith_target)
            self.send_robot_info(v,0,w)
            # print("v:{} w:{}".format(v,w))
            time.sleep(0.1)
        while abs(self.angle_diff())>0.1:
            print("position{} reached".format(ith_target))
            # print("current_pose:{}".format(self.current))
            # print("target_pose:{}".format(self.target))
            
            self.correct_pose([0,0,0.75],ith_target)
            self.send_robot_info(0,0,0.75)
            time.sleep(0.1)
        time.sleep(2)
        

    def run(self):
        timeout = time.time()+15
        while(True):
            self.correct_pose([0,0,0.75],0)
            self.send_robot_info(0,0,0.75)
            time.sleep(0.1)
            if time.time()>timeout:
                break
        # epoch_num = 3
        # for epoch in range(epoch_num):
        #     pts_x,pts_y,pts_theta = self.wp
        #     current_x = pts_x[0]
        #     current_y = pts_y[0]
        #     current_theta = pts_theta[0]

        #     target_length = len(pts_x)
        #     self.current = np.array([current_x,current_y,current_theta])

        #     for i in range(1,target_length):
        #         self.target = [pts_x[i],pts_y[i],pts_theta[i]]
        #         self.move_to_target(i)
        #     print("finished epoch {}".format(epoch))
        #     # self.plot(epoch=epoch)
        self.stop()


    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    def plot(self,epoch=0):
        plot_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan']
        if self.collect_history_data:
            plt.plot([p[0] for p in self.robot_history_poses], [p[1] for p in self.robot_history_poses], color='red')
            points = []
            for idx, landmark_pose in enumerate(self.landmark_poses):
                point = plt.scatter([p[idx][0] for p in self.landmarks_history_positions if idx < p.shape[0]], [p[idx][1] for p in self.landmarks_history_positions if idx < p.shape[0]], color=plot_colors[idx])
                points.append(point)
            plt.legend(points, tuple(self.known_landmark_ids), scatterpoints=1)
        else:
            plt.scatter(self.robot_pose[0], self.robot_pose[1], color='red')
            plt.arrow(self.robot_pose[0], self.robot_pose[1], 0.05*cos(self.robot_pose[2]), 0.05*sin(self.robot_pose[2]), color='black')
            points = []
            for idx, landmark_pose in enumerate(self.landmark_poses):
                point = plt.scatter(landmark_pose[0], landmark_pose[1], color=plot_colors[idx])
                points.append(point)
                plt.arrow(landmark_pose[0], landmark_pose[1], 0.05*cos(landmark_pose[2]), 0.05*sin(landmark_pose[2]), color='black')
            plt.legend(points, tuple(self.known_landmark_ids), scatterpoints=1)
        plt.savefig('epoch_'+str(epoch)+'.png')
        plt.close()
        self.robot_history_poses = [self.current]
        self.landmarks_history_positions=[]
        print("epoch_{}_cov_mat".format(epoch))
        print("ego_coordinate:")
        print(self.ekf.get_cov_matrix()[:3,:3])
        for i in self.known_landmark_ids:
            print("landmark_id:{}".format(i))
            print(self.get_lankmark_cov_mat_by_id(i))
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
    rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, path_plan_node.read_landmark_detection, queue_size=1)
    rospy.on_shutdown(path_plan_node.stop)
    path_plan_node.run()
