import socket
import numpy as np
import struct
import time
from scipy.spatial.transform import Rotation as R


class robot_controller:
    def __init__(self):
        self.UDP_IP_IN = (
            "192.168.1.200"  # Ubuntu IP, should be the same as Matlab shows
        )
        self.UDP_PORT_IN = (
            57831  # Ubuntu receive port, should be the same as Matlab shows
        )
        self.UDP_IP_OUT = (
            "192.168.1.100"  # Target PC IP, should be the same as Matlab shows
        )
        self.UDP_PORT_OUT = 3826  # Robot 1 receive Port
        self.gripper_port = 3828  # Robot 1 receive Port

        # self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        # Receive TCP position (3*), TCP Rotation Matrix (9*), TCP Velcoity (6*), Force Torque (6*)
        self.unpacker = struct.Struct("12d 6d 6d")

        self.robot_pose, self.robot_vel, self.TCP_wrench = None, None, None

        
    def receive(self):
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s_in.bind((self.UDP_IP_IN, self.UDP_PORT_IN))
        data, _ = self.s_in.recvfrom(1024)
        unpacked_data = np.array(self.unpacker.unpack(data))
        self.robot_pose, self.robot_vel, self.TCP_wrench = (
            unpacked_data[0:12],
            unpacked_data[12:18],
            unpacked_data[18:24]
        )
        self.s_in.close()
        

    def send(self, udp_cmd):
        '''
        UDP command 1~6 TCP desired Position Rotation
        UDP desired vel 7~12 
        UDP Kp 13~18
        UDP Kd 19~24
        UDP Mass 25~27
        UDP Interial 28~30
        '''
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_cmd = udp_cmd.astype("d").tostring()
        self.s_out.sendto(udp_cmd, (self.UDP_IP_OUT, self.UDP_PORT_OUT))
        self.s_out.close()
    
    def get_current_pose(self):
        self.receive()
        robot_pos = self.robot_pose[0:3]
        robot_ori = self.robot_pose[3:12].reshape(3, 3).T
        robot_vel = self.robot_vel
        contact_force = self.TCP_wrench[0:6]
        return robot_pos, robot_ori, robot_vel, contact_force
    
    def gripper_move(self):
        one = np.array(1)
        zero = np.array(0)
        self.s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_out.sendto(one.astype("d").tobytes(), (self.UDP_IP_OUT, self.gripper_port))
        time.sleep(0.5)
        self.s_out.sendto(zero.astype("d").tobytes(), (self.UDP_IP_OUT, self.gripper_port))
        time.sleep(0.5)

    def move_to_point(self, waypoint, compliant=False, wait=10):
        if compliant:
            Mass = np.array([1,1,1])   # to determine
            Inertia = 1*np.array([2, 2, 0.1])   # to determine
            Kp = np.array([0,0,0,0,0,0])
            Kd = np.array([70,70,70,20,20,10])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        else:
            Mass = np.array([2,2,2])   # to determine
            Inertia = 1*np.array([2, 2, 2])   # to determine
            Kp = np.array([600,600,600,200,200,200])
            Kd = np.array([300,300,300,250,250,250])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        # send the command to robot until the robot reaches the waypoint
        dis = 100
        dis_ori = 100
        init_time = time.time()
        desired_ori = R.from_euler("ZYX", TCP_d_euler).as_matrix()
        while ((dis > 0.005 or dis_ori > 1/180*np.pi) and time.time()-init_time<wait):
            self.receive()
            dis = np.linalg.norm(self.robot_pose[0:3]-waypoint[:3])
            robot_ori = self.robot_pose[3:12].reshape(3,3).T
            dis_ori = np.arccos((np.trace(robot_ori@desired_ori.T)-1)/2)
            UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, TCP_d_vel, Kp, Kd, Mass, Inertia])
            self.send(UDP_cmd)
            # print(dis_ori)

    def move_to_point_step(self, waypoint, compliant=False, wait=10):
        if compliant:
            Mass = np.array([1,1,1])   # to determine
            Inertia = 1*np.array([2, 2, 0.1])   # to determine
            Kp = np.array([0,0,0,0,0,0])
            Kd = np.array([70,70,70,20,20,10])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        else:
            Mass = np.array([2,2,2])   # to determine
            Inertia = 1*np.array([2, 2, 2])   # to determine
            Kp = np.array([600,600,600,200,200,200])
            Kd = np.array([300,300,300,250,250,250])
            TCP_d_pos = waypoint[:3]
            TCP_d_euler = waypoint[3:]
            TCP_d_vel = np.zeros(6)
        # send the command to robot until the robot reaches the waypoint
        dis = 100
        dis_ori = 100
        init_time = time.time()
        desired_ori = R.from_euler("ZYX", TCP_d_euler).as_matrix()
        # while ((dis > 0.005 or dis_ori > 1/180*np.pi) and time.time()-init_time<wait):
        if True:
            self.receive()
            dis = np.linalg.norm(self.robot_pose[0:3]-waypoint[:3])
            robot_ori = self.robot_pose[3:12].reshape(3,3).T
            dis_ori = np.arccos((np.trace(robot_ori@desired_ori.T)-1)/2)
            UDP_cmd = np.hstack([TCP_d_pos, TCP_d_euler, TCP_d_vel, Kp, Kd, Mass, Inertia])
            self.send(UDP_cmd)
            # print(dis_ori)
        if dis <= 0.005 and dis_ori <= 1/180*np.pi:
            reached = True
        else:
            reached = False
        return reached

if __name__ == "__main__": 
    rc = robot_controller()
    robot_pos, robot_ori, robot_vel, contact_force = rc.get_current_pose()
    print(robot_pos)
    print(robot_ori)
    rc.gripper_move()