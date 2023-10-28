#!/usr/bin/env python

import math
import rospy
import tf
from april_detection.msg import AprilTagDetectionArray
# from geometry_msgs.msg import Pose
from hw2.msg import RobotPose
import numpy as np



class RobotPoseNode:
    def __init__(self):
        self.pub_pose = rospy.Publisher("/robot_pose", Pose, queue_size=1)
        self.apriltag_original_pose = {}
        self.apriltag_config = {'7':[[0.0,0.0,0.0],0.0,0.0,0.0],
                       '8':[[0.0,-0.2,0.0],0.0,0.0,0.0]}
        # april tag world pose
        # camera to robot
        self.edit_april_tag_pose()
        self.matrix_cr = self.compute_translation(p=[-2.7e-02,-2.5e-02,-17.5e-02],
                                roll=0.0,
                                pitch=0.0,
                                yaw=0.0,is_inv=False)

    def edit_april_tag_pose(self):
        for tag in self.apriltag_config:
            spec = self.apriltag_config[tag]
            self.apriltag_original_pose[tag] = self.compute_translation(p=spec[0],
                                    roll=spec[1],
                                    pitch=spec[2],
                                    yaw=spec[3],is_inv=False)
    
    def to_robot_axis(self,x,y,z):
        robot_z = -y 
        robot_y = -x
        robot_x = z
        return (robot_x,robot_y,robot_z)

    def decode(self,april_detection_array_msg):
        tags = []
        for detection in april_detection_array_msg.detections:
            # if tag.id in self.apriltag_original_pose:
            q = detection.pose.orientation
            p = detection.pose.position
            tags.append([str(detection.id),q,p])
        return tags

    def compute_pose_to_world(self,tag,matrix_ac):
        return np.matmul(np.matmul(self.apriltag_original_pose[tag],matrix_ac),self.matrix_cr)

    def compute_translation(self,p,roll,pitch,yaw,is_inv=False):
        # given coord c in axis a compute c->a
        R_yaw = [[np.cos(yaw),-np.sin(yaw),0],
                 [np.sin(yaw),np.cos(yaw),0],
                 [0,0,1]]
        R_pitch = [[np.cos(pitch),0,np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0,np.cos(pitch)]]
        R_roll = [[1,0,0],
                  [0,np.cos(roll),-np.sin(roll)],
                  [0,np.sin(roll),np.cos(roll)]]
        R = np.matmul(np.matmul(R_yaw,R_pitch),R_roll)
        t = np.array([p]).T
        if is_inv == True:
        # given the pose of c in axis a compute matrix a->c
            R = R.T
            t = -np.matmul(R,t)
        return np.concatenate((np.concatenate((R,t),axis=1),[[0,0,0,1]]),axis=0)

    def calc_pose(self,tag,q,p):
        dist = math.sqrt(p.x**2 + p.z**2)
        quaternion = (q.x, q.y, q.z, q.w)
        pitch,yaw,roll = tf.transformations.euler_from_quaternion(quaternion)
        # print("x:{} y:{} z:{}".format(p.x,p.y,p.z))
        # print("dist: {}".format(dist))
        # print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))
        p_transformed = self.to_robot_axis(p.x,p.y,p.z)
        matrix_ac = self.compute_translation(p_transformed,-roll,-pitch,-yaw,is_inv=True)
        pose = self.compute_pose_to_world(tag,matrix_ac)
        euler = np.array([tf.transformations.euler_from_matrix(pose[:3,:3])]).T
        t = pose[:3,3:]
        print("{}:euler:{}\nt:{}".format(tag,euler,t))
        return np.concatenate((euler,t),axis=1)

    def pose_adjustment(self,poses):
        return np.mean(poses,axis=0)
            
    def run(self,april_detection_array_msg):
        tags = self.decode(april_detection_array_msg)
        poses = []
        for tag in tags:
            name,q,p = tag
            poses.append(self.calc_pose(name,q,p))
        pose = self.pose_adjustment(poses)
        self.send_pose(pose)
        # print("avg:{}\n{}".format(euler,t))
        

    def send_pose(self, pose):
        print(pose)
        pose_msg = RobotPose()
        pose_msg.x=pose[0][0]
        pose_msg.y=pose[0][1]
        pose_msg.z=pose[0][2]
        pose_msg.roll=pose[1][0]
        pose_msg.pitch=pose[1][1]
        pose_msg.yaw=pose[1][2]
        self.pub_pose.publish(pose_msg)


if __name__ == "__main__":
    robot_pose_node = RobotPoseNode()
    rospy.init_node('robot_pose_parser')
    detection_array_msg = rospy.wait_for_message("/apriltag_detection_array", AprilTagDetectionArray, timeout=1.0)
    robot_pose_node.run(detection_array_msg)
        
