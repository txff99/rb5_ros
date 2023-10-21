#!/usr/bin/env python

import rospy
import time
from hw1.msg import RobotInfo
from mpi_control import MegaPiController

def sign(v):
    return 1.0 if v>=0 else -1.0

class RobotControllerNode:
    def __init__(self):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        self.r = 0.03
        self.lx = 0.067
        self.ly = 0.054
    
    def calc_wheel_w(self, vx, vy, wz):
        w_fl = (vx-vy-(self.lx+self.ly)*wz) / self.r
        w_fr = (vx+vy+(self.lx+self.ly)*wz) / self.r
        w_bl = (vx+vy-(self.lx+self.ly)*wz) / self.r
        w_br = (vx-vy+(self.lx+self.ly)*wz) / self.r
        return w_fl, w_fr, w_bl, w_br
    
    def calc_control_input(self, w_fl, w_fr, w_bl, w_br):
        
        v_fl = sign(w_fl) * (7.4738 * abs(w_fl) + 4.3118)
        v_fr = sign(w_fr) * (7.1953 * abs(w_fr) + 4.7908)
        v_bl = sign(w_bl) * (6.9881 * abs(w_bl) + 5.11)
        v_br = sign(w_br) * (6.9494 * abs(w_br) + 4.8995)
        return v_fl, v_fr, v_bl, v_br

    def control(self, vx, vy, wz):
        w_fl, w_fr, w_bl, w_br = self.calc_wheel_w(vx, vy, wz)
        v_fl, v_fr, v_bl, v_br = self.calc_control_input(w_fl, w_fr, w_bl, w_br)
        self.mpi_ctrl.setFourMotors(-v_fl, v_fr, -v_bl, v_br)

    def msg_callback(self, robot_info_msg):
        if robot_info_msg.vx == 0.0 and robot_info_msg.vy == 0.0 and robot_info_msg.wz == 0.0:
            self.stop()
        else:
            self.control(robot_info_msg.vx, robot_info_msg.vy, robot_info_msg.wz)

    def straight(self, vx):
        self.control(vx, 0, 0)
    
    def slide(self, vy):
        self.control(0, vy, 0)
    
    def rotate(self, wz):
        self.control(0, 0, wz)
    
    def stop(self):
        self.mpi_ctrl.setFourMotors(0, 0, 0, 0)
        

if __name__ == "__main__":
    robot_ctrl_node = RobotControllerNode()
    # time.sleep(1)
    # robot_ctrl_node.rotate(0.65)
    # time.sleep(1)
    # robot_ctrl_node.stop()
    rospy.init_node('robot_controller')
    rospy.Subscriber('/robot_info', RobotInfo, robot_ctrl_node.msg_callback, queue_size=1) 
    rospy.spin()
