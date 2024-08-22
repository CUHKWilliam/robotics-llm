import numpy as np
from realsense import RealSense
from robot_controller import robot_controller
import pickle

class RealWorldEnv():
    def __init__(self, server=False):
        if not server:
            rs = RealSense()
            self.rs = rs
        else:
            self.rs = None
        self.robot_controller = robot_controller()
        self.desire_pos = None
        self.desire_euler = None
        self.data_file_path = "/media/msc-auto/HDD/wltang/tmp/communication.pkl"
        self.lock_file_path = "/media/msc-auto/HDD/wltang/tmp/lock.txt"
        self.server_name = "msc-auto@128.32.164.89"
        with open(self.lock_file_path, "w") as f:
            f.write('0')

    def get_obs(self, ):
        obs = {}
        color_image, depth_image = self.rs.capture()
        obs['color_image'] = color_image
        obs['depth_image'] = depth_image
        ## get position and pose of end-effector
        robot_pos, robot_ori, robot_vel, contact_force = self.robot_controller.get_current_pose()
        obs['ee_pos'] = robot_pos
        obs['ee_euler'] = robot_ori
        obs['ee_force'] = contact_force
        return obs

    def set_action(self, action):
        new_ee_pos = action['ee_pos']
        new_ee_euler = action['ee_euler']
        ## set end-effector position and pose
        self.desire_pos = new_ee_pos
        self.desire_euler = new_ee_euler
        return

    def step(self, ):
        reached = self.robot_controller.move_to_point_step(np.concatenate([self.desire_pos, self.desire_euler]))
        return reached
    
    def get_obs_remote(self):
        print("waiting for obs data... ")
        while not self.check_recv():
            continue
        print("obs get")
        with open(self.data_file_path, "rb") as f:
            recv = pickle.load(f)
        import ipdb;ipdb.set_trace()
        return recv
    
    def set_action_remote(self, action):
        new_ee_pos = action['ee_pos']
        new_ee_euler = action['ee_euler']
        ## set end-effector position and pose
        self.desire_pos = new_ee_pos
        self.desire_euler = new_ee_euler
        with open(self.data_file_path, 'wb') as f:
            pickle.dump({"desire_pos": self.desire_pos,  "desire_euler": self.desire_euler}, f)
        with open(self.lock_file_path, 'w') as f:
            f.write("0")

    def check_recv(self,):
        with open(self.lock_file_path, "r") as f:
            content = f.read().strip()
            if content  == "0":
                return False
            else:
                return True
