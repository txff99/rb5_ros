#!/usr/bin/env python

import math
import numpy as np
from numpy import sin, cos
import tf
import time
from threading import Lock

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
    def __init__(self, x0, landmark_poses, Q, R):
        self.last_x = x0
        self.landmark_poses = landmark_poses
        self.last_P = np.eye(self.last_x.shape[0])*0.01
        self.Q = Q
        self.R = R

    def debug_print(self):
        print("x:\n{}".format(self.last_x))
        print("P:\n{}".format(self.last_P))

    def update(self, u, z, landmark_ids, interval=0.1):
        # prediction update
        x_prior = self.last_x
        x_prior[0], x_prior[1], x_prior[2] = estimate_next_pose(self.last_x[0], self.last_x[1], self.last_x[2], u[0], u[1], u[2], interval)
        x_prior[2] = normalize_angle(x_prior[2])
        P_prior = self.last_P + self.Q
        # print("x_prior: {}".format(x_prior))

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
            landmark_x = self.landmark_poses[landmark_id][0]
            landmark_y = self.landmark_poses[landmark_id][1]
            landmark_theta = self.landmark_poses[landmark_id][2]

            h[idx*3+0] = cos(robot_theta) * landmark_x + sin(robot_theta) * landmark_y - cos(robot_theta) * robot_x - sin(robot_theta) * robot_y
            h[idx*3+1] = -sin(robot_theta) * landmark_x + cos(robot_theta) * landmark_y + sin(robot_theta) * robot_x - cos(robot_theta) * robot_y
            h[idx*3+2] = landmark_theta - robot_theta
            
            H[idx*3+0, 0] = -cos(robot_theta)
            H[idx*3+0, 1] = -sin(robot_theta)
            H[idx*3+0, 2] = -landmark_x*sin(robot_theta) + landmark_y*cos(robot_theta) + robot_x*sin(robot_theta) - robot_y*cos(robot_theta)
            H[idx*3+1, 0] = sin(robot_theta)
            H[idx*3+1, 1] = -cos(robot_theta)
            H[idx*3+1, 2] = -landmark_x*cos(robot_theta) - landmark_y*sin(robot_theta) + robot_x*cos(robot_theta) + robot_y*sin(robot_theta)
            H[idx*3+2, 0] = 0.0
            H[idx*3+2, 1] = 0.0
            H[idx*3+2, 2] = -1.0
        # print("h: {}".format(h))
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
        self.last_x[2] = normalize_angle(self.last_x[2])
        self.last_P = np.matmul((np.eye(state_vector_length, dtype=float) - np.matmul(K, H)), P_prior)

    def get_robot_pose(self):
        return self.last_x
    
    def get_landmark_poses(self):
        return self.landmark_poses
    
    def get_cov_matrix(self):
        return self.last_P

class PIDController:
    def __init__(self, k_p, k_i, k_d, integral_limit):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
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
        return self.k_p * err + self.k_i * self.integral + self.k_d * diff

class PlannerNode:
    def __init__(self, robot_init_pose, landmark_poses):
        self.robot_pose = np.array(robot_init_pose)
        self.landmark_poses = landmark_poses

        self.mutex = Lock()
        self.landmarks_detection_results = {}

        Q = np.eye(3, dtype=float)
        Q[0, 0] = 0.01
        Q[1, 1] = 0.01
        Q[2, 2] = 0.04
        R = np.eye(3, dtype=float)
        R[0, 0] = 0.02
        R[1, 1] = 0.02
        R[2, 2] = 0.08
        self.ekf = EKF(self.robot_pose, self.landmark_poses, Q, R)

        self.pub_robot_info = rospy.Publisher("/robot_info", RobotInfo, queue_size=1)

        self.interval = 0.1
        self.rate = rospy.Rate(1.0/self.interval)
        self.v_controller = PIDController(0.40, 0.0, 0.00, integral_limit=1.0)
        self.w_controller = PIDController(0.20, 0.0, 0.02, integral_limit=math.pi/2)

        self.not_detected_time_length = 0
    
    def read_landmark_detection(self, april_detection_array_msg):
        detection_results = {}
        for detection in april_detection_array_msg.detections:
            landmark_id = detection.id
            if landmark_id >= 7:
                continue
            position = detection.pose.position
            camera_x = position.x
            camera_z = position.z
            quaternion = detection.pose.orientation 
            _, pitch, _ = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
            detection_results[landmark_id] = [camera_z, -camera_x, -normalize_angle(pitch)]
        with self.mutex:
            for landmark_id in detection_results:
                if landmark_id in self.landmarks_detection_results:
                    r_new = math.sqrt(detection_results[landmark_id][0]**2 + detection_results[landmark_id][1]**2)
                    r_old = math.sqrt(self.landmarks_detection_results[landmark_id][0]**2 + self.landmarks_detection_results[landmark_id][1]**2)
                    if abs(r_new - r_old) > 1.0:
                        del detection_results[landmark_id]
            self.landmarks_detection_results = detection_results
            # print(self.landmarks_detection_results)
    
    def is_near_waypoint(self, target_waypoint):
        r_err = math.sqrt((target_waypoint[0] - self.robot_pose[0]) ** 2 + (target_waypoint[1] - self.robot_pose[1]) ** 2)
        theta_err = normalize_angle(target_waypoint[2] - self.robot_pose[2])
        print("robot_pose: {}, target_pose: {}".format(self.robot_pose, target_waypoint))
        return r_err, theta_err, r_err < 0.05 and abs(theta_err) < 1.0

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
                observations = np.append(observations, np.array(landmark_pose))
                observed_landmark_ids.append(landmark_id)
        self.ekf.update(robot_input, observations, observed_landmark_ids, self.interval)
        self.robot_pose = self.ekf.get_robot_pose()
    
    def step_once(self, destination_waypoint_pose):
        r_err, theta_err, close_enough = self.is_near_waypoint(destination_waypoint_pose)
        if close_enough == False:
            with self.mutex:
                detected = len(self.landmarks_detection_results) > 0
                if detected:
                    self.not_detected_time_length = 0
                else:
                    self.not_detected_time_length += 1
            
            if self.not_detected_time_length < 100:
                v = self.v_controller.calc_output(r_err)
                w = self.w_controller.calc_output(theta_err)
                v = clip(v, 0.07, 0.12)
                w = clip(w, 0.70, 1.00)
                vx, vy = self.calc_vx_vy(self.robot_pose[0], self.robot_pose[1], destination_waypoint_pose[0], destination_waypoint_pose[1], v)
                robot_vx, robot_vy = self.calc_v_related_to_robot(vx, vy, self.robot_pose[2])      
                robot_input = np.array([robot_vx, robot_vy, w])
            else:
                robot_input = np.array([0.0, 0.0, 0.85])
            self.send_robot_info(robot_input[0], robot_input[1], robot_input[2])
            self.rate.sleep()
            self.update_state(robot_input)
        return close_enough

    def goto_waypoint(self, waypoint_pose):
        print("go to waypoint {}".format(waypoint_pose))
        self.v_controller.reset()
        self.w_controller.reset()

        while self.step_once(waypoint_pose) == False:
            pass
        self.stop()
        
    def send_robot_info(self, robot_vx, robot_vy, wz):
        robot_info_msg = RobotInfo()
        robot_info_msg.vx = robot_vx
        robot_info_msg.vy = robot_vy
        robot_info_msg.wz = wz
        self.pub_robot_info.publish(robot_info_msg)

    def stop(self):
        self.send_robot_info(0.0, 0.0, 0.0)

if __name__ == "__main__":
    rospy.init_node("hw4_test_node")

    robot_start_pose = [0.0, 0.0, 0.78]
    landmark_poses = {
        0: [1.4, 0.0, 0.0],
        1: [1.0, 2.65, math.pi/2],
        2: [-0.47, 2.0, math.pi],
        3: [0.0, -0.56, -math.pi/2],
        4: [0.32, 1.00, 0.0],
        5: [0.48, 0.82, math.pi/2],
        6: [1.32, 0.93, 0.0],
    }
    test_node = PlannerNode(robot_start_pose, landmark_poses)
    
    rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, test_node.read_landmark_detection, queue_size=1)
    rospy.on_shutdown(test_node.stop)

    time.sleep(0.5)
    # waypoint_list = [
    #     [0.0,0.0],
    #     [0.22000000000000003,-0.009999999999999988],
    #     [0.32000000000000023,0.09000000000000001],
    #     [0.36645695364238423,0.13645695364238408],
    #     [0.42000000000000004,0.13141304347826077],
    #     [0.6958753551136362,0.14365855823863635],
    #     [0.9191525423728812,0.5808474576271188],
    #     [1.011111111111111,0.79],
    #     [0.9361111111111111,1.39],
    #     [0.8979999999999999,1.49],
    #     [0.7446666666666665,1.79],
    #     [0.841509433962264,1.89],
    # ]
    shortest_path = [
        [0.0,0.0],
        [0.8,0.8],
        [0.9,1.4],
    ]
    path = shortest_path
    for i in range(1, len(path)):
        orientation = math.atan2(path[i][1]-path[i-1][1], path[i][0]-path[i-1][0])
        test_node.goto_waypoint(path[i]+[orientation])
    test_node.goto_waypoint([1.0, 2.0, 1.57])
    