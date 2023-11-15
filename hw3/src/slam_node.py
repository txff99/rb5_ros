#!/usr/bin/env python

import math
import time
import numpy as np
from numpy import sin, cos, arctan2
from threading import Lock
import tf
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import rospy
from april_detection.msg import AprilTagDetectionArray
from hw1.msg import RobotInfo

def normalize_angle(angle):
    normalized_angle = np.mod(angle + np.pi, 2*np.pi) - np.pi
    return normalized_angle

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def clip(value, min_value, max_value):
    if value == 0.0:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * max(min(abs(value), max_value), min_value)

def estimate_next_pose(now_x, now_y, now_theta, vx, vy, w, interval=0.1):
    if w == 0:
        new_x = now_x + vx * interval
        new_y = now_y + vy * interval
        new_theta = now_theta
        return new_x, new_y, new_theta
    v = math.sqrt(vx**2 + vy**2)
    if v == 0:
        return now_x, now_y, now_theta + interval * w
    R = abs(v/w)
    nx = -vy if w>=0 else vy
    ny = vx if w>=0 else -vx
    norm_n = math.sqrt(nx**2 + ny**2)
    nx /= norm_n / R
    ny /= norm_n / R
    circle_x = now_x + nx
    circle_y = now_y + ny
    rad_in_interval = interval * w
    new_x = circle_x + math.cos(rad_in_interval) * (-nx) - math.sin(rad_in_interval) * (-ny)
    new_y = circle_y + math.sin(rad_in_interval) * (-nx) + math.cos(rad_in_interval) * (-ny)
    new_theta = now_theta + rad_in_interval
    return new_x, new_y, new_theta

class EKF:
    def __init__(self, x0, P0, Q, R):
        self.last_x = x0
        self.last_P = P0
        self.Q = Q
        self.R = R

    def extend(self, n_landmarks):
        if n_landmarks <= 0:
            return
        
        original_length = self.last_x.shape[0]

        init_state_est = np.zeros(n_landmarks * 3, dtype=float)
        self.last_x = np.append(self.last_x, init_state_est)

        init_covar_est = np.eye(n_landmarks * 3)
        for i in range(n_landmarks):
            init_covar_est[i*3, i*3] = 1.0
            init_covar_est[i*3+1, i*3+1] = 1.0
            init_covar_est[i*3+2, i*3+2] = 4.0
        self.last_P = np.block([[self.last_P, np.zeros((original_length, n_landmarks*3), dtype=float)], [np.zeros((n_landmarks*3, original_length), dtype=float), init_covar_est]])

        self.Q = np.pad(self.Q, ((0, n_landmarks*3), (0, n_landmarks*3)), mode='constant')

    def debug_print(self):
        print("x:\n{}".format(self.last_x))
        print("P:\n{}".format(self.last_P))

    def update(self, u, z, landmark_ids, interval=0.1):
        # prediction update
        x_prior = self.last_x
        x_prior[0], x_prior[1], x_prior[2] = estimate_next_pose(self.last_x[0], self.last_x[1], self.last_x[2], u[0], u[1], u[2], interval)
        x_prior[2] = normalize_angle(x_prior[2])
        P_prior = self.last_P + self.Q

        # if we could not measure anything, then just skip measurement update
        if z.shape[0] == 0:
            self.last_x = x_prior
            self.last_P = P_prior
            return

        # measurement update
        measurement_length = z.shape[0]
        state_vector_length = self.last_x.shape[0]
        H = np.zeros((measurement_length, state_vector_length), dtype=float)
        h = np.zeros(measurement_length, dtype=float)
        robot_x = x_prior[0]
        robot_y = x_prior[1]
        robot_theta = x_prior[2]
        for idx, landmark_id in enumerate(landmark_ids):
            landmark_x = x_prior[3+landmark_id*3]
            landmark_y = x_prior[3+landmark_id*3+1]
            landmark_theta = x_prior[3+landmark_id*3+2]

            h[idx*3+0] = cos(robot_theta) * landmark_x + sin(robot_theta) * landmark_y - cos(robot_theta) * robot_x - sin(robot_theta) * robot_y
            h[idx*3+1] = -sin(robot_theta) * landmark_x + cos(robot_theta) * landmark_y + sin(robot_theta) * robot_x - cos(robot_theta) * robot_y
            h[idx*3+2] = arctan2(-sin(robot_theta)*cos(landmark_theta) + cos(robot_theta)*sin(landmark_theta), cos(robot_theta)*cos(landmark_theta) + sin(robot_theta)*sin(landmark_theta))
            
            H[idx*3+0, 0] = -cos(robot_theta)
            H[idx*3+0, 1] = -sin(robot_theta)
            H[idx*3+0, 2] = -landmark_x*sin(robot_theta) + landmark_y*cos(robot_theta) + robot_x*sin(robot_theta) - robot_y*cos(robot_theta)
            H[idx*3+1, 0] = sin(robot_theta)
            H[idx*3+1, 1] = -cos(robot_theta)
            H[idx*3+1, 2] = -landmark_x*cos(robot_theta) - landmark_y*sin(robot_theta) + robot_x*cos(robot_theta) + robot_y*sin(robot_theta)
            H[idx*3+2, 0] = 0.0
            H[idx*3+2, 1] = 0.0
            H[idx*3+2, 2] = -1.0

            H[idx*3+0, 3+landmark_id*3+0] = cos(robot_theta)
            H[idx*3+0, 3+landmark_id*3+1] = sin(robot_theta)
            H[idx*3+0, 3+landmark_id*3+2] = 0.0
            H[idx*3+1, 3+landmark_id*3+0] = -sin(robot_theta)
            H[idx*3+1, 3+landmark_id*3+1] = cos(robot_theta)
            H[idx*3+1, 3+landmark_id*3+2] = 0.0
            H[idx*3+2, 3+landmark_id*3+0] = 0.0
            H[idx*3+2, 3+landmark_id*3+1] = 0.0
            H[idx*3+2, 3+landmark_id*3+2] = 1.0
        H_trans = np.transpose(H)
        R_stacked = np.eye(measurement_length, dtype=float)
        for i in range(len(landmark_ids)):
            R_stacked[i*3+0, i*3+0] = self.R[0, 0]
            R_stacked[i*3+1, i*3+1] = self.R[1, 1]
            R_stacked[i*3+2, i*3+2] = self.R[2, 2]
        K = np.matmul(np.matmul(P_prior, H_trans), np.linalg.inv(np.matmul(np.matmul(H, P_prior), H_trans) + R_stacked))
        tmp = z - h
        for j in range(len(landmark_ids)):
            tmp[j*3+2] = normalize_angle(tmp[j*3+2])
        self.last_x = x_prior + np.dot(K, tmp)
        for i in range(self.last_x.shape[0] // 3):
            self.last_x[i*3+2] = normalize_angle(self.last_x[i*3+2])
        self.last_P = np.matmul((np.eye(state_vector_length, dtype=float) - np.matmul(K, H)), P_prior)

    def get_landmark_size(self):
        return len(self.get_landmark_pose()) // 3

    def get_robot_pose(self):
        return self.last_x[:3]
    
    def get_landmark_pose(self):
        return self.last_x[3:]
    
    def get_cov_matrix(self):
        return self.last_P

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

class ControlNode:
    def __init__(self):
        self.known_landmark_ids = []
        self.landmark_id2state_vector_id = {}

        self.mutex = Lock()
        self.landmarks_detection_results = {}

        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.landmark_poses = np.array([])
        P0 = np.eye(3, dtype=float)
        P0[0, 0] = 0.0
        P0[1, 1] = 0.0
        P0[2, 2] = 0.0
        Q = np.eye(3, dtype=float)
        Q[0, 0] = 0.01
        Q[1, 1] = 0.01
        Q[2, 2] = 0.16
        R = np.eye(3, dtype=float)
        R[0, 0] = 0.16
        R[1, 1] = 0.16
        R[2, 2] = 1.44
        self.ekf = EKF(self.robot_pose, P0, Q, R)

        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)

        self.interval = 0.1
        self.v_controller = PIDController(0.5, 0.0, 0.03, self.interval, 1.0)
        self.w_controller = PIDController(0.6, 0.0, 0.03, self.interval, math.pi)

        self.collect_history_data = True
        self.robot_history_poses = [self.robot_pose]
        self.landmarks_history_positions = []

        self.not_detected_time_length = 0

    
    def read_landmark_detection(self, april_detection_array_msg):
        with self.mutex:
            self.landmarks_detection_results = {}
            for detection in april_detection_array_msg.detections:
                landmark_id = detection.id
                position = detection.pose.position
                camera_x = position.x
                camera_z = position.z
                quaternion = detection.pose.orientation 
                _, pitch, _ = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
                self.landmarks_detection_results[landmark_id] = [camera_z, -camera_x, -normalize_angle(pitch)]
    
    def get_lankmark_pose_by_id(self, landmark_id):
        landmark_poses = np.reshape(self.ekf.get_landmark_pose(), (-1, 3))
        return landmark_poses[self.landmark_id2state_vector_id[landmark_id]]
    
    def get_lankmark_cov_mat_by_id(self, landmark_id):
        cov_matrix = self.ekf.get_cov_matrix()
        idx = self.landmark_id2state_vector_id[landmark_id]
        return cov_matrix[3+idx*3:3+(idx+1)*3, 3+idx*3:3+(idx+1)*3]
    
    def is_near_waypoint(self, target_waypoint):
        r_err = math.sqrt((target_waypoint[0] - self.robot_pose[0]) ** 2 + (target_waypoint[1] - self.robot_pose[1]) ** 2)
        theta_err = normalize_angle(target_waypoint[2] - self.robot_pose[2])
        print("robot_pose: {}, target_pose: {}".format(self.robot_pose, target_waypoint))
        return r_err, theta_err, r_err < 0.1 and abs(theta_err) < 0.5

    def calc_vx_vy(self, now_x, now_y, end_x, end_y, v):
        rx = end_x - now_x
        ry = end_y - now_y
        r = math.sqrt(rx**2 + ry**2)
        vx = rx / r * v
        vy = ry / r * v
        return vx, vy
    
    def calc_v_related_to_robot(self, vx, vy, robot_theta):
        robot_vx = math.cos(-robot_theta) * vx - math.sin(-robot_theta) * vy
        robot_vy = math.sin(-robot_theta) * vx + math.cos(-robot_theta) * vy
        return robot_vx, robot_vy

    def update_state(self, robot_input):
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
        self.robot_pose = self.ekf.get_robot_pose()
        self.landmark_poses = np.reshape(self.ekf.get_landmark_pose(), (-1, 3))
        if self.collect_history_data:
            self.robot_history_poses.append(self.robot_pose)
            self.landmarks_history_positions.append(self.landmark_poses[:,:2])
    
    def step_once(self, destination_waypoint_pose):
        r_err, theta_err, close_enough = self.is_near_waypoint(destination_waypoint_pose)
        if close_enough == False:
            with self.mutex:
                detected = len(self.landmarks_detection_results) > 0
                if detected:
                    self.not_detected_time_length = 0
                else:
                    self.not_detected_time_length += 1
            
            if self.not_detected_time_length < 5:
                v = self.v_controller.calc_output(r_err)
                w = self.w_controller.calc_output(theta_err)
                v = clip(v, 0.07, 0.14)
                w = clip(w, 0.75, 1.1)
                vx, vy = self.calc_vx_vy(self.robot_pose[0], self.robot_pose[1], destination_waypoint_pose[0], destination_waypoint_pose[1], v)
                robot_vx, robot_vy = self.calc_v_related_to_robot(vx, vy, self.robot_pose[2])      
                robot_input = np.array([robot_vx, robot_vy, w])
            else:
                robot_input = np.array([0.0, 0.0, 0.75])
            self.send_robot_info(robot_input[0], robot_input[1], robot_input[2])
            time.sleep(self.interval)
            self.update_state(robot_input)
        return close_enough


    def goto_landmark(self, landmark_id):
        print("go to landmark {}".format(landmark_id))
        self.v_controller.reset()
        self.w_controller.reset()

        DIST_TO_TAG = 0.35

        landmark_pose = self.get_lankmark_pose_by_id(landmark_id)
        target_waypoint = [landmark_pose[0]-DIST_TO_TAG*cos(landmark_pose[2]), landmark_pose[1]-DIST_TO_TAG*sin(landmark_pose[2]), landmark_pose[2]]

        while self.step_once(target_waypoint) == False:
            landmark_pose = self.get_lankmark_pose_by_id(landmark_id)
            target_waypoint = [landmark_pose[0]-DIST_TO_TAG*cos(landmark_pose[2]), landmark_pose[1]-DIST_TO_TAG*sin(landmark_pose[2]), landmark_pose[2]]
        self.stop()

    def goto_waypoint(self, waypoint_pose):
        print("go to waypoint {}".format(waypoint_pose))
        self.v_controller.reset()
        self.w_controller.reset()

        while self.step_once(waypoint_pose) == False:
            pass
        self.stop()

    def run(self):
        start_time = time.time()
        elapsed_time = 0
        timeout = 10
        while elapsed_time < timeout:
            robot_input = [0.0, 0.0, 0.9]  # always spin
            self.send_robot_info(robot_input[0], robot_input[1], robot_input[2])
            time.sleep(self.interval)

            self.update_state(robot_input)
            
            print("robot_pose:\n{}".format(self.robot_pose))
            print("landmark_poses:")
            for idx, landmark_pose in enumerate(self.landmark_poses):
                print("{}: {}".format(self.known_landmark_ids[idx], landmark_pose))
            print("")
            elapsed_time = time.time() - start_time
        self.stop()
        print("landmark_ids: {}".format(self.known_landmark_ids))
        time.sleep(1)

        for loop in range(2):
            self.landmark_poses = np.reshape(self.ekf.get_landmark_pose(), (-1, 3))
            landmark_orientation_with_id = [(arctan2(pose[1], pose[0]), id) for id, pose in zip(self.known_landmark_ids, self.landmark_poses)]
            landmark_orientation_with_id.sort()
            for _, id in landmark_orientation_with_id:
                self.goto_landmark(id)
                time.sleep(1)
            self.plot("loop_"+str(loop+1)+".png")
            self.robot_history_poses = []
            self.landmarks_history_positions = []

        # waypoints = [[0.0, 0.6, 0.0]]# , [-0.5, 0.5, -1.57], [-0.5, -0.5, 0.0], [0.5, -0.5, 1.57]]
        # for waypoint in waypoints:
        #     self.goto_waypoint(waypoint)

    def plot(self, file_path="map.png"):
        plot_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan']
        if self.collect_history_data:
            plt.plot([p[0] for p in self.robot_history_poses], [p[1] for p in self.robot_history_poses], color='red')
            points = []
            for idx, landmark_pose in enumerate(self.landmark_poses):
                xx = []
                yy = []
                for p in self.landmarks_history_positions:
                    if idx < p.shape[0]:
                        xx.append(p[idx][0])
                        yy.append(p[idx][1])
                point = plt.scatter(xx, yy, color=plot_colors[idx])
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
        plt.savefig(file_path)
        plt.close()
        
    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    def stop(self):
        self.send_robot_info(0.0, 0.0, 0.0)
            

if __name__ == "__main__":
    robot_control_node = ControlNode()
    rospy.init_node("robot_control_node")
    rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, robot_control_node.read_landmark_detection, queue_size=1)
    rospy.on_shutdown(robot_control_node.stop)

    time.sleep(0.5)
    robot_control_node.run()