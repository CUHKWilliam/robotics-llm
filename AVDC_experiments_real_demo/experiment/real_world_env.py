import numpy as np
from realsense import RealSense
from robot_controller import robot_controller

class RealWorldEnv():
    def __init__(self, ):
        rs = RealSense()
        self.rs = rs
        self.robot_controller = robot_controller()
        self.desire_pos = None
        self.desire_euler = None

    def get_obs(self):
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

    def get_point_cloud(self, image, depth):
        pass