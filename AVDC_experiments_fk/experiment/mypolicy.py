from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.action import Action
import numpy as np
from metaworld_exp.utils import get_seg, get_cmat
import json
import cv2
from flowdiffusion.inference_utils import pred_video, pred_video_rgbd_fk
from myutils import pred_flow_frame, get_transforms, get_transformation_matrix
import torch
from PIL import Image
from torchvision import transforms as T
import torch
import time
import pickle
import random
import open3d as o3d

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def log_time(time_vid, time_flow, time_action, n_replan, log_dir="logs"):
    with open(f"{log_dir}/time_vid_{n_replan}.txt", "a") as f:
        f.write(f"{time_vid}\n")
    with open(f"{log_dir}/time_flow_{n_replan}.txt", "a") as f:
        f.write(f"{time_flow}\n")
    with open(f"{log_dir}/time_action_{n_replan}.txt", "a") as f:
        f.write(f"{time_action}\n")

def log_time_execution(time_execution, n_replan, log_dir="logs"):
    with open(f"{log_dir}/time_execution_{n_replan}.txt", "a") as f:
        f.write(f"{time_execution}\n")

class ProxyPolicy(Policy):
    def __init__(self, env, proxy_model, camera, task, resolution=(320, 240)):
        self.env = env
        self.proxy_model = proxy_model
        self.camera = camera
        self.task = task
        self.last_pos = np.array([0, 0, 0])
        self.grasped = False
        with open("../text_embeds.pkl", "rb") as f:
            self.task2embed = pickle.load(f)
        with open("name2mode.json", "r") as f:
            name2mode = json.load(f)
        self.mode = name2mode[task]
        self.resolution = resolution
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])
        self.seg_ids = name2maskid[task]

        grasp, transforms = self.calculate_next_plan()

        subgoals = self.calc_subgoals(grasp, transforms)

        subgoals_np = np.array(subgoals)
        if self.mode == "push":
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()

    def calc_subgoals(self, grasp, transforms):
        print("Calculating subgoals...")
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        x = self.transform(Image.fromarray(image*np.expand_dims(seg, axis=2))).unsqueeze(0)
        # substract "-v2-goal-observable" from task string without rstip
        
        
        task_embed = torch.tensor(self.task2embed[self.task.split("-v2-goal-observable")[0]]).unsqueeze(0)
        flow = self.proxy_model(x, task_embed).squeeze(0).cpu().numpy()

        # make flow back to (320, 240), paste the (128, 128) flow to the center
        blank = np.zeros((2, 240, 320))
        blank[:, 56:184, 96:224] = flow
        flow = blank * 133.5560760498047 ## flow_abs_max=133.5560760498047
        flow = [flow.transpose(1, 2, 0)]
        

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

        return grasp[0], transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        self.last_pos = o_d['hand_pos']

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # place end effector above object
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # replan
        else:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp
            self.subgoals = self.calc_subgoals(grasp, transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8

class DiffusionPolicy(Policy):
    def __init__(self, env, policy_model, camera, task, resolution=(320, 240), obs_cache_size=2, min_action_cache_size=8):
        self.env = env
        self.policy_model = policy_model
        self.camera = camera
        self.task = task
        self.resolution = resolution
        self.obs_cache_size = obs_cache_size # To
        self.min_action_cache_size = min_action_cache_size # Tp - Ta
        assert self.obs_cache_size > 0 and self.min_action_cache_size >= 0

        self.obs_cache = []
        self.action_cache = []

    def reset(self):
        self.obs_cache = []
        self.action_cache = []

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }

    def get_stack_obs(self):
        return np.stack(self.obs_cache, axis=0)
    
    def update_obs_cache(self, obs):
        while len(self.obs_cache) < self.obs_cache_size:
            self.obs_cache.append(obs)
        self.obs_cache.append(obs)
        self.obs_cache.pop(0)
        assert len(self.obs_cache) == self.obs_cache_size
    
    def replan(self):
        stack_obs = self.get_stack_obs()
        self.action_cache = [a for a in self.policy_model(stack_obs, self.task)]
    
    def get_action(self, obs):
        obs, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        self.update_obs_cache(obs)
        
        if len(self.action_cache) <= self.min_action_cache_size:
            self.replan()
        
        return self.action_cache.pop(0)

class IDPolicy(Policy):
    def __init__(self, env, ID_model, video_model, camera, task, resolution=(320, 240), max_replans=5):
        self.env = env
        self.remain_replans = max_replans + 1
        self.vid_plan = []
        self.ID_model = ID_model
        self.ID_model.cuda()
        self.subgoal_idx = 0
        self.video_model = video_model
        self.resolution = resolution
        self.task = task
        with open("ID_exp/all_cams.json", "r") as f:
            all_cams = json.load(f)
        cam2vec = {cam: torch.eye(len(all_cams))[i] for i, cam in enumerate(all_cams)}
        self.camera = camera
        self.cam_vec = cam2vec[camera]

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.replan()

    def replan(self):
        image, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        self.subgoal_idx = 0
        self.vid_plan = []
        self.vid_plan = pred_video(self.video_model, image, self.task)
        self.remain_replans -= 1

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }

    def get_action(self, obs):
        obs, _ = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        obs = self.transform(Image.fromarray(obs))
        subgoal = self.vid_plan[self.subgoal_idx].transpose(1, 2, 0)
        subgoal = self.transform(Image.fromarray(subgoal))
        cam = self.cam_vec

        with torch.no_grad():
            action, is_last = self.ID_model(obs.unsqueeze(0).cuda(), subgoal.unsqueeze(0).cuda(), cam.unsqueeze(0).cuda())
            action = action.squeeze().cpu().numpy()
            is_last = is_last.squeeze().cpu().numpy() > 0
        
        if is_last:
            if self.subgoal_idx < len(self.vid_plan) - 1:
                self.subgoal_idx += 1
            elif self.remain_replans > 0:
                self.replan()
        
        return action

class MyPolicy(Policy):
    def __init__(self, grasp, transforms):
        subgoals = []
        grasp = grasp[0]
        subgoals.append(grasp)
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
            subgoals = [s + np.array([0, 0, 0.03]) for s in subgoals[:-1]] + [subgoals[-1]]
        else:
            self.mode = "push"
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  
        
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # place end effector above object
        if not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp + np.array([0., 0., 0.03])
        # grab object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the next subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > 0.02:
            return self.subgoals[0]
        else:
            self.grasped=False
            return self.subgoals[0]
        
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']
        
        if self.phase_grasp and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        elif not self.phase_grasp:
            return 0.8
        else:
            return -0.8
        # if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
        #     return 0.8
        # else:
        #     return -0.8

class MyPolicy_CL(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(320, 240), plan_timeout=15, max_replans=0, log=False):

        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log

        grasp, transforms = self.calculate_next_plan()

        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        # measure time for vidgen
        start = time.time()
        images = pred_video(self.video_model, image, self.task)
        time_vid = time.time() - start

        # measure time for flow
        start = time.time()
        image1, image2, color, flow, flow_b = pred_flow_frame(self.flow_model, images)
        time_flow = time.time() - start

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)

        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        t = len(transform_mats)
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8

class MyPolicy_CL_rgbd_v0(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(320, 240), plan_timeout=15, max_replans=0, log=False):

        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log

        grasp, transforms = self.calculate_next_plan()

        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        depth2 = depth.copy()
        low, high = -8., -1.5
        depth[depth < low] = low
        depth[depth > high] = high
        depth -= low
        depth /= (high - low)
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        segm = np.zeros((image.shape[0], image.shape[1]))
        segm[data[:, :, -1] == 31] = 1
        segm[data[:, :, -1] == 33] = 2
        segm1 = (segm == 1).astype(depth.dtype)
        segm2 = (segm == 2).astype(depth.dtype)
        image_depth_segm = np.concatenate([image, depth[..., None], segm1[..., None], segm2[..., None]], axis=-1)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)
        # measure time for vidgen
        start = time.time()
        images, depths, segms1, segms2 = pred_video_rgbd(self.video_model, image_depth_segm, self.task)
        time_vid = time.time() - start

        # measure time for flow
        start = time.time()
        image1, image2, color, flow, flow_b = pred_flow_frame(self.flow_model, images)
        time_flow = time.time() - start

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth2, cmat, flow)

        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        t = len(transform_mats)
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8


def sample_with_binear(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = int(kp[0]), int(kp[1])
    x1, y1 = x0+1, y0+1
    x, y = kp[0]-x0, kp[1]-y0
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def to_3d(points, depth, cmat):
    points = points.reshape(-1, 2)
    # depths = np.array([[sample_with_binear(depth, kp)] for kp in points])
    depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_3d2(points, depths, cmat):
    points = points.reshape(-1, 2)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths[:, None]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

from sklearn.cluster import KMeans
import argparse
import sys
sys.path.append("/data/wltang/robotics-llm/AVDC_experiments/experiment/CamLiFlow")
scene_flow_model_config = {'name': 'camliraft', 'batch_size': 8, 'freeze_bn': False, 'backbone': {'depth': 50, 'pretrained': 'pretrain/resnet50-11ad3fa6.pth'}, 'n_iters_train': 10, 'n_iters_eval': 20, 'fuse_fnet': True, 'fuse_cnet': True, 'fuse_corr': True, 'fuse_motion': True, 'fuse_hidden': False, 'loss2d': {'gamma': 0.8, 'order': 'l2-norm'}, 'loss3d': {'gamma': 0.8, 'order': 'l2-norm'}}
scene_flow_model_config = argparse.Namespace(**scene_flow_model_config)
scene_flow_model_config.backbone = argparse.Namespace(**{"depth": 50, "pretrained": './CamLiFlow/pretrain/resnet50-11ad3fa6.pth'})
scene_flow_model_config.loss2d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})
scene_flow_model_config.loss3d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})

DEBUG = False
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").cuda()
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import contact_detection
import json
from metaworld_exp.inverse_kinematics import calculate_ik
import mujoco_py
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import torchvision
def draw_arrow(pcd, start, end):
    pts = np.linspace(start, end, num=50)
    pcs = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    pcs = np.concatenate([pcs, pts], axis=0)
    cols = np.concatenate([cols, np.ones((len(pts), 3)) * np.array([1, 0, 0])])
    pcd.points = o3d.utility.Vector3dVector(pcs)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


class MyPolicy_CL_rgbd(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(640, 480), plan_timeout=20, max_replans=5, log=False, return_qpos=False):
        self.depth_low = -8
        self.depth_high = -1.5
        plan_timeout = 100
        self.env = env
        self.seg_ids = name2maskid[task]
        self.full_task_name = task
        if 'v2' in  task:
            self.task = " ".join(task.split('-')[:-3])
        elif "v3" in task:
            self.task = task.split('-v3')[0].replace("_", " ").replace("sdoor", "sliding door").replace("ldoor", 'opening door').replace('micro', 'microwave') 
        
        with open('task2description.json', 'r') as f:
            self.task2description = json.load(f)
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.return_qpos = return_qpos
        self.last_pos = np.array([0, 0, 0]) if not self.return_qpos else np.array([0.] * 29)
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log
        self.phase_grasp = True


        # self.scene_flow_model = model_factory(
        #     scene_flow_model_config,
        # )
        # scene_flow_model_ckpt = torch.load("./CamLiFlow/camliraft_things150e.pt")
        # self.scene_flow_model.load_state_dict(scene_flow_model_ckpt['state_dict'], strict=True)
        # self.scene_flow_model.to("cuda:0")
        # self.scene_flow_model.eval()

        ## TODO:
        # self.tasks = self.plan_tasks(task)
        # self.tasks = self.plan_tasks_LISA(task)
        self.tasks = ['locate', 'grasp', self.task]
        subgoals = []
        for i in range(len(self.tasks)):
            self.tasks[i] = self.tasks[i].strip()
        self.tasks_all = self.tasks
        self.is_traj = False
        subgoals, subgoals_rot = self.calculate_next_plan()
        self.mode = 'push'
        self.subgoals = subgoals 
        self.subgoals_rot = subgoals_rot
        self.subgoals_2d = None
        self.is_grasp = False
        self.grasp_cnt = 0
        self.grasp_lock = False
        self.need_refine = False
        self.cnt_wait = 0
        self.grasping = False

    def init_grasp(self):
        self.grasped = False

    def plan_tasks(self, task):
        template_path = './prompt_decompose_task.txt'
        with open(template_path, 'r') as f:
            template = f.read()
        query = template.replace('{}', task)
        from openai import OpenAI

        client = OpenAI(
        api_key = 'sk-wlsfmHh2a8QKxzQZbiR100l994fxc4aHUR6FMOoFtNTHQvAC',
        base_url = 'https://api.chatanywhere.tech/v1'
        )

        # openai.api_key = 'sk-wlsfmHh2a8QKxzQZbiR100l994fxc4aHUR6FMOoFtNTHQvAC'

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": query}
            ]
        )

        print(completion)
        tasks = completion.choices[0].message.content.replace('[', '').replace(']', '').split(',')
        return tasks


    def get_ee_pos(self):
        ee_id = self.env.sim.model.site_name2id('end_effector')
        ee_pos1 = self.env.sim.data.site_xpos[ee_id]
        ee_id = self.env.sim.model.site_name2id('end_effector')
        ee_pos2 = self.env.sim.data.site_xpos[ee_id]
        return (ee_pos1 + ee_pos2) / 2.
    
    def get_ee_quat(self):
        ee_id = self.env.sim.model.site_name2id('end_effector')
        ee_quat = np.array([0, 0, 0, 0]).astype(np.float64)
        mujoco_py.functions.mju_mat2Quat(ee_quat, self.env.sim.data.site_xmat[ee_id])
        return ee_quat

    def calculate_next_plan(self, first=False, pos_curr=None):
        self.task = self.tasks[0]
        
        if self.task.split(' ')[0] == 'locate':
            self.is_traj = False
            # subgoals = self.calculate_locate_predict(task = self.task)
            subgoals = self.calculate_locate(task = self.task)
            subgoals_rot = [None for _ in range(len(subgoals))]
        elif self.task.split(' ')[0] == 'grasp':
            self.is_traj = False
            self.is_grasp = True
            self.grasp_lock = False
            self.grasp_cnt = 0
            # subgoals = self.calculate_grasp_predict()
            subgoals = self.calculate_grasp(name = self.task)
            subgoals_rot = [None for _ in range(len(subgoals))]
        else:
            self.grasping = False
            safe_travel = False
            self.is_traj = True
            subgoals, subgoals_rot = self.calculate_traj()
        
        # if not self.is_traj:
        #     subgoals = self.safe_subgoals(subgoals, self.previous_image, self.previous_depth, pos_curr)
        
        if self.return_qpos:
            # subgoal = self.get_ee_pos()

            # subgoals2 = []
            # for i in range(0, len(subgoals)):
            #     subgoal2 = subgoals[i]
            #     length = np.linalg.norm(subgoal - subgoal2)
            #     if length > 0.1:
            #         subgoals2 += list(np.linspace(subgoal, subgoal2, 5))
            #     else:
            #         subgoals2 += [subgoal, subgoal2]
            #     subgoal = subgoal2
            #     subgoals2.append(subgoal)
            # subgoals = subgoals2

            qposes = []
            xyzs = []
            # self.vis_subgoals(subgoals)
        return subgoals, subgoals_rot
        
    def vis_image(self):
        image, depth = self.env.get_image_depth(body_invisible=True)
        cv2.imwrite('debug.png', image)
        # import ipdb;ipdb.set_trace()

    def vis_subgoals(self, subgoals):
        for i in range(len(subgoals)):
            if i > 8:
                break
            subgoal = subgoals[i]
            debug_ball_id = self.env.sim.model.body_name2id("debug{}".format(i))
            self.env.sim.model.body_pos[debug_ball_id][:] = subgoal.copy()
            
            mujoco_py.functions.mj_forward(self.env.sim.model, self.env.sim.data) 
        self.vis_image()
        if not self.is_traj:
            for i in range(len(subgoals)):
                if i > 8:
                    break
                subgoal = subgoals[i]
                debug_ball_id = self.env.sim.model.body_name2id("debug{}".format(i))
                self.env.sim.model.body_pos[debug_ball_id][:] = np.array([-10, -0., -0])
                mujoco_py.functions.mj_forward(self.env.sim.model, self.env.sim.data) 
        return

        
    def xyz_to_qpos(self, xyz, split_path=False):
        model = self.env.sim.model
        data = self.env.sim.data
        qpos0 = data.qpos.copy()
        qpos_targets = []
        xyzs = [xyz]
        start_xyz = self.get_ee_pos()
        xyzs_ret = [xyz]
        cnt = 0
        while len(xyzs) > 0:
            a_xyz = xyzs[0]
            qpos1 = data.qpos.copy()
            qpos_target, success = calculate_ik(model, data, a_xyz, self.env)

            split_path = False
            if success or not split_path:
                qpos_targets.append(qpos_target)
                start_xyz = xyzs[0]
                xyzs_ret.append(xyzs[0])
                xyzs = xyzs[1:]
            else:
                inter_xyz = (start_xyz + a_xyz) / 2.
                xyzs = [inter_xyz] + xyzs
                xyzs_ret = [inter_xyz] + xyzs
                data.qpos[:] = qpos1.copy()
            cnt += 1
            if cnt > 5:
                break
        if not split_path:
            qpos_targets = qpos_targets[0]
        self.env.sim.data.qpos[:] = qpos0.copy()
        mujoco_py.functions.mj_forward(self.env.sim.model, self.env.sim.data) 
        return qpos_targets, xyzs_ret

    def calculate_locate(self, task=None):
        if self.env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = self.env.get_image_depth(body_invisible=True)
        else:
            image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)

        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        if isinstance(self.seg_ids, dict):
            seg_ids = [self.seg_ids[name] + 20]
        else:
            seg_ids = list(np.array(self.seg_ids) + 20)
        
        if self.env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = self.env.get_image_depth(body_invisible=True)
            seg = self.env.get_segmentation(body_invisible=True)[:, :, -1] == seg_ids[0]
        else:
            image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
            seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=seg_ids)
        seg = cv2.erode(seg.astype(np.uint8) * 255, np.ones((2, 2), np.uint8) ).astype(np.bool_)
        self.previous_seg = seg.copy()

        seg_aligned_image = torch.from_numpy(seg[::-1, :].copy()).float()
        seg = torch.from_numpy(seg).float()
        width, height = image.shape[1], image.shape[0]
        pts_2d_all = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        image[seg.detach().cpu().numpy()[::-1, :] ==1] = np.array([255, 0, 0])
        cv2.imwrite('debug.png', image)
        # import ipdb;ipdb.set_trace()

        # pts_2d = pts_2d[seg > -1]
        # pts_2d = np.array([[127, 127]])
        # pts_3d = to_3d(pts_2d, depth, cmat)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_3d)
        # pcd.colors = o3d.utility.Vector3dVector(image[::-1, :, :].reshape(-1, 3) / 255.)
        # o3d.io.write_point_cloud("debug.ply", pcd)
        # import ipdb;ipdb.set_trace()

        
        pts_2d = pts_2d_all[seg > 0]
        if (seg > -0).sum() == 0:
            return None

        pts_3d = to_3d(pts_2d, depth, cmat)
        # TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy()
        cam_pos = self.env.sim.model.cam_pos[self.env.sim.model.camera_name2id(self.camera)]
        cam_vec =( pts_3d - cam_pos) / np.linalg.norm(pts_3d - cam_pos)
        pts_3d = pts_3d + cam_vec * 0.02 # + np.array([0, -0.02, 0])

        self.grasp_3d = pts_3d
        self.grasp_2ds = pts_2d.detach().cpu().numpy()
        self.grasp_2ds_aligned_image = pts_2d_all[seg_aligned_image > 0]
        # subgoals = [pts_3d + np.array([0, -0.1, 0.])]
        subgoals = [self.grasp_3d +  np.array([0, -0.3, 0.])]
        self.subgoals_2d = subgoals
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals

    def calculate_locate_predict(self, task=None):
        if self.env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = self.env.get_image_depth(body_invisible=True)
        else:
            image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        full_task_name = self.full_task_name
        description = self.task2description[full_task_name]
        prompt = "Where should I grasp if I need to conduct task {} ? Please output segmentation mask.".format(description)
        seg = contact_detection.predict(image, prompt)
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        
        pos_curr_2d_1 = np.stack(np.where(data[:, :, -1] == 51), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d_2 = np.stack(np.where(data[:, :, -1] == 53), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d = (pos_curr_2d_1 + pos_curr_2d_2) / 2.
        self.pos_curr_2d = pos_curr_2d

        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)

        width, height = image.shape[1], image.shape[0]
        pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d[seg > 0]
        pts_3d = to_3d(pts_2d, depth, cmat)
        self.grasp_2ds = pts_2d.detach().cpu().numpy()
        # TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy() + np.array([0, 0, 0.0])
        pts_3d = pts_3d
        self.grasp_3d = pts_3d
        subgoals = [pts_3d + np.array([0, 0, 0.2])]
        self.subgoals_2d = subgoals
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        return subgoals

    def refine_subgoal(self, subgoal, subgoal_2d, tgt_depth, tgt_image):
        prev_image, prev_depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        prev_depth[prev_depth < self.depth_low] = self.depth_low
        prev_depth[prev_depth > self.depth_high] = self.depth_high
        prev_depth -= self.depth_low
        prev_depth /= (self.depth_high - self.depth_low)
        import ipdb;ipdb.set_trace()
        prev_image, tgt_image, query=subgoal_2d

        return refined_subgoal

    def grasp_pose_estimation(self, pcs, pcs_obj):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcs_mean = pcs.mean(0)
        pcs -= pcs.mean(0)
        pcs_obj -= pcs_mean
        pcd.points = o3d.utility.Vector3dVector(pcs)
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcs), 3)))
        o3d.io.write_point_cloud("tmp.pcd", pcd)
        import subprocess
        grasp_cfg_path = "/data/wltang/robotic-llm/gpd/cfg/eigen_params.cfg"
        grasp_bin_path = "detect_grasps"
        output = subprocess.check_output(['{}'.format(grasp_bin_path), '{}'.format(grasp_cfg_path), "tmp.pcd"])
        app_strs = str(output).split("Approach")[1:]
        approaches = []
        for app_str in app_strs:
            app_str = app_str.strip().split(':')[1].strip()
            app_vec =  app_str.split("\\n")
            app_vec = np.array([float(app_vec[0]), float(app_vec[1]), float(app_vec[2])])
            approaches.append(app_vec)
        approaches = np.stack(approaches, axis=0)
        pos_str = app_strs[-1]
        pos_strs = pos_str.split("Position")[1:]
        positions = []
        for pos_str in pos_strs:
            pos_str = pos_str.strip().split(':')[1].strip()
            pos_vec =  pos_str.split("\\n")
            pos_vec = np.array([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2])])
            positions.append(pos_vec)
        positions = np.stack(positions, axis=0)
        dist = np.min(np.linalg.norm(positions[:, None] - pcs_obj[None, :], axis=-1), axis=-1)
        selected = dist < 2
        approaches = approaches[selected]
        positions = positions[selected]
        ## TODO: visualization
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcs)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcs), 3)))
        # starts = positions
        # ends = positions + approaches / 5.
        # for i in range(len(starts)):
        #     pcd = draw_arrow(pcd, starts[i], ends[i])
        # o3d.io.write_point_cloud('debug.ply', pcd)
        # import ipdb;ipdb.set_trace()
        ## TODO: end visualization

        def rotation_matrix_from_vectors(vec1, vec2):
            """ Find the rotation matrix that aligns vec1 to vec2
            :param vec1: A 3d "source" vector
            :param vec2: A 3d "destination" vector
            :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
            """
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix
        approach = approaches[0]
        approach = approach / np.linalg.norm(approach)
        rot_mat = rotation_matrix_from_vectors(np.array([0., 1., 0.]), approach)
        return R.from_matrix(rot_mat).as_quat()


    def calculate_grasp(self, name=None):
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        pts_3d = self.grasp_3d + np.array([-0.04, 0.07, 0])
        ## TODO: grasp pose estimation
        subgoals = [pts_3d]
        seg = self.previous_seg
        depth =self.previous_depth
        seg_dilate = cv2.dilate(seg.copy().astype(np.uint8), np.ones((30, 30))) > 0

        width, height = seg_dilate.shape[1], seg_dilate.shape[0]
        pts_2d_ = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d_[seg_dilate > 0]
        pts_2d_obj = pts_2d_[seg > 0]
        pts_3ds = to_3d(pts_2d, depth, cmat)
        pts_3ds_obj = to_3d(pts_2d_obj, depth , cmat)
        # rot_quat = self.grasp_pose_estimation(pts_3ds, pts_3ds_obj)

        ## TODO: for visualization
        '''
        pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width-1), np.arange(height-1)), axis=-1))
        pts_2d = pts_2d.reshape(-1, 2)
        pts = to_3d(pts_2d, depth, cmat)
        cols = np.ones((height, width, 3))
        cols = image[:image.shape[0]-1, :image.shape[1]-1, :] / 256.
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        pcd_sphere.translate(pts_3d.detach().cpu().numpy())
        pts_sphere = np.asarray(pcd_sphere.sample_points_uniformly(500).points)
        pts = np.concatenate([pts, pts_sphere], axis=0)
        cols = cols.reshape(-1, 3)
        cols = np.concatenate([cols, np.array([[1, 1, 0]] * len(pts_sphere))], axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud('debug.ply', pcd)
        import ipdb;ipdb.set_trace()
        '''
        ## TODO: end TODO
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals

    def calculate_grasp_predict(self):
        if self.env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = self.env.get_image_depth(body_invisible=True)
        else:
            image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        
        pos_curr_2d_1 = np.stack(np.where(data[:, :, -1] == 51), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d_2 = np.stack(np.where(data[:, :, -1] == 53), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d = (pos_curr_2d_1 + pos_curr_2d_2) / 2.
        self.pos_curr_2d = pos_curr_2d

        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        pts_3d = self.grasp_3d        
        self.grasp_3d = pts_3d
        subgoals = [pts_3d + np.array([0, 0, -0.1])]
        self.subgoals_2d = subgoals
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals
    
    def visualize_ply(self, rgb, depth, cmat, file_name):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        width, height = rgb.shape[1], rgb.shape[0]
        pts_2d = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
        mask = np.logical_and(depth > -3, depth < -1.5)
        pts_2d = pts_2d[mask]
        colors = rgb[mask]
        pts = to_3d(pts_2d, depth, cmat)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
        o3d.io.write_point_cloud(file_name, pcd)
        import ipdb;ipdb.set_trace()
        

    def calculate_quat(self, sgs, prev_sgs):
        sgs -= sgs.mean(0)
        prev_sgs -= prev_sgs.mean(0)

        # import fpsample
        # fps_samples_idx = fpsample.fps_sampling(sgs, 6)
        # sgs = sgs[fps_samples_idx]
        # prev_sgs = prev_sgs[fps_samples_idx]

        mat = cv2.estimateAffine3D(prev_sgs, sgs)[1]
        rot_mat = mat[:3, :3]
        rot_mat = rot_mat / np.linalg.norm(rot_mat, axis=-1, keepdims=True)
        rotvec = R.from_matrix(rot_mat).as_rotvec()
        # rotvec = rotvec[[0, 2, 1]]
        rotvec[rotvec > 50] = 0
        rotvec *= np.array([0, 1, 0])
        rot_quat = R.from_rotvec(rotvec).as_quat()
        return rot_quat, rot_mat

    # def calculate_quat(self, sgs, prev_sgs):
    #     sgs -= sgs.mean(0)
    #     prev_sgs -= prev_sgs.mean(0)
    #     import open3d as o3d
    #     src_pcd = o3d.geometry.PointCloud()
    #     src_pcd.points = o3d.utility.Vector3dVector(prev_sgs)
    #     tgt_pcd = o3d.geometry.PointCloud()
    #     tgt_pcd.points = o3d.utility.Vector3dVector(sgs)
    #     threshold = 0.002  # Distance threshold
    #     trans_init = np.identity(4)  # Initial guess (identity matrix)
    #     trans_init[:3, :3]  = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 4, np.pi / 4, np.pi / 4))  # We set the initial rotation to the known rotation
    #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #         source=src_pcd, target=tgt_pcd, max_correspondence_distance=threshold,
    #         init=trans_init
    #     )

    #     # Extract the rotation matrix from the transformation matrix
    #     estimated_rotation_matrix = reg_p2p.transformation[:3, :3]
    #     rotation_matrix = reg_p2p.transformation[:3, :3]
    #     rot_quat = R.from_matrix(rotation_matrix.copy()).as_quat()
    #     return rot_quat, rotation_matrix

    def calculate_traj(self):
        self.mode = 'push'
        self.previous_direction = None
        if self.env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = self.env.get_image_depth(body_invisible=True)
        else:
            image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        depth -= self.depth_low
        depth /= (self.depth_high - self.depth_low)
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()

        segm = np.zeros((image.shape[0], image.shape[1]))

        segm1 = (segm == 1).astype(depth.dtype)
        segm1[self.grasp_2ds[:, 1], self.grasp_2ds[:, 0]] = 1.
        segm2 = (segm == 2).astype(depth.dtype)

        depth_orig_size = depth.shape
        image_depth_segm = np.concatenate([image, depth[::-1, :, None], segm1[::-1, :, None], segm2[..., None]], axis=-1)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        
        # measure time for vidgen
        start = time.time()
        images, depths, segms1, segms2 = pred_video_rgbd_fk(self.video_model, image_depth_segm, self.task)

      
        images = F.interpolate(torch.from_numpy(images), depth_orig_size, ).detach().cpu().numpy()
        video = torch.tensor(images)[None].float().cuda()
        ## TODO:
        # self.grasp_2ds = np.expand_dims(self.grasp_2d, axis=0)
        grasp_2d_aligned = (self.grasp_2ds_aligned_image).detach().cpu().numpy().astype(np.int64)
        
        center_crop = False
        if center_crop:
            grasp_2d_aligned = np.round((grasp_2d_aligned - depth_orig_size[0] / 2.) * (depth_orig_size[0] / 120.) + depth_orig_size[0] / 2.)
        
        queries = torch.cat([torch.tensor([0] * grasp_2d_aligned.shape[0]).unsqueeze(-1), torch.from_numpy(grasp_2d_aligned)], dim=-1).cuda().float()
        pred_tracks, pred_visibility = cotracker(video, queries=queries[None])
        subgoals_2d = pred_tracks[0, :, :, :].detach().cpu().numpy()
        ## TODO: vis
        for idx in range(len(images)):
            image_debug = images[idx].copy().transpose(1, 2, 0).astype(np.uint8).copy()
            for i in range(len(subgoals_2d[idx])):
                image_debug = cv2.circle(image_debug, [int(subgoals_2d[idx, i, 0]), int(subgoals_2d[idx, i, 1])], radius=1, color=[3 * i, 0, 0], thickness=-1)
            cv2.imwrite('debug_{}.png'.format(idx), image_debug)

        # import ipdb;ipdb.set_trace()
        # _, _, flow_debug, flow, flow_b = pred_flow_frame(self.flow_model, images)
        # self.visualize_ply(self.pred_image_depth_segms[0, :, :, :3], self.pred_image_depth_segms[0, :, :, 3] * (self.depth_high - self.depth_low) + self.depth_low, cmat, "debug1.ply")
        # self.visualize_ply(self.pred_image_depth_segms[1, :, :, :3], self.pred_image_depth_segms[1, :, :, 3] * (self.depth_high - self.depth_low) + self.depth_low, cmat, "debug2.ply")
        subgoals = []
        subgoals_rot = []
        
        if center_crop:
            depths = F.interpolate(depths, (120, 120))
            depths = torchvision.transforms.Pad(int(depth_orig_size[0] - 120) // 2)(depths)

        depths = F.interpolate(depths, depth_orig_size)
        ## TODO:
        # depths[0] = torch.from_numpy(depth).unsqueeze(0)
        height = depths.shape[-2]
        prev_sgs = None
        sgs_seq = []
        for i in range(len(depths)):
            subgoals_2d[i][:, 1] = height - subgoals_2d[i][:, 1]
            subgoals_2d[i] = np.clip(subgoals_2d[i], a_min=0, a_max=255)
            if center_crop:
                subgoals_2d[i] = np.round((subgoals_2d[i] - depth_orig_size[0] / 2.) * (120. / depth_orig_size[0]) + depth_orig_size[0] / 2.)

            sgs = to_3d(subgoals_2d[i], (torch.flip(depths[i][0], dims=[0]) * (self.depth_high - self.depth_low)) + self.depth_low, cmat)
            sgs_seq.append(sgs)
            sg_pos = np.median(sgs, axis=0)
            if prev_sgs is not None and 'grasp' in self.tasks_all:
                sg_quat, _ = self.calculate_quat(sgs, prev_sgs)
                prev_sgs = sgs
                subgoals.append(sg_pos)
                subgoals_rot.append(sg_quat)
                
            else:
                prev_sgs = sgs
                subgoals.append(sg_pos)
                subgoals_rot.append(None)

        # if 'grasp' in self.tasks_all:
        #     for i in range(len(subgoals_rot)):
        #         sgs_seq[3] -= sgs_seq[3].mean(0)
        #         sgs_seq[1] -= sgs_seq[1].mean(0)
        #         sg_quat, mat = self.calculate_quat(sgs_seq[3], sgs_seq[1])
        #         import open3d as o3d
        #         pcs_trans = sgs_seq[1] @ mat.T
        #         pcs = np.concatenate([sgs_seq[1], sgs_seq[3], pcs_trans], axis=0)
        #         cols = np.concatenate([
        #             np.repeat(np.expand_dims(np.array([1, 0, 0]), axis=0), repeats=len(sgs_seq[1]), axis=0), # * np.arange(len(sgs_seq[0]))[:, None], 
        #             np.repeat(np.expand_dims(np.array([0, 1, 0]), axis=0), repeats=len(sgs_seq[3]), axis=0), # * np.arange(len(sgs_seq[0]))[:, None],
        #             np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=0), repeats=len(pcs_trans), axis=0), # * np.arange(len(sgs_seq[0]))[:, None]
        #         ], axis=0)

        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(pcs)
        #         pcd.colors = o3d.utility.Vector3dVector(cols)
        #         o3d.io.write_point_cloud("debug.ply", pcd)
        #         import ipdb;ipdb.set_trace()
        #         rotvec = R.from_quat(sg_quat).as_rotvec()
        #         rotvec *= np.array([0, 1, 0])
        #         sg_quat = R.from_rotvec(rotvec).as_quat()
        #         subgoals_rot[i] = sg_quat
        ## TODO:
        
        if 'grasp' in self.tasks_all:
            self.grasp_3d = self.get_ee_pos() # + np.array([-0.04, 0.07, 0])
            subgoals = [subgoal  + (self.grasp_3d - subgoals[0]) for subgoal in subgoals]

            direction = (subgoals[-1] - subgoals[-2])
            direction /= np.linalg.norm(direction)
            subgoals = subgoals + [subgoals[-1] + 0.3 * direction]
            subgoals_rot = subgoals_rot + [None]
            
        if 'grasp' not in self.tasks_all:
            direction = (self.grasp_3d - subgoals[-1]) / np.linalg.norm(self.grasp_3d - subgoals[-1])
            subgoals = [self.grasp_3d + 0.2 * direction] + [self.grasp_3d] + subgoals
            subgoals_rot = [None, None] + subgoals_rot
            direction = (subgoals[-1] - subgoals[0])
            direction /= np.linalg.norm(direction)
            subgoals = subgoals + [subgoals[-1] + 0.2 * direction]
            subgoals_rot = subgoals_rot + [None]
        subgoals = [x + np.array([0, -0., 0]) for x in subgoals]
        time_vid = time.time() - start

        # measure time for action planning
        time_action = time.time() - start
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals, subgoals_rot
    ''' 
    def get_contact_point(self, pos_curr, image, depth):
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        width, height = image.shape[1], image.shape[0]
        pts_2d_map = torch.from_numpy(np.stack(np.meshgrid(np.arange(width - 1), np.arange(height - 1)), axis=-1))
        pts_2d = pts_2d_map.reshape(-1, 2)
        mask = segm
        pts_2d = pts_2d[mask.reshape(-1)]
        avoid_pts = to_3d(pts_2d, depth, cmat)
    '''

    def safe_subgoals(self, subgoals, image, depth, pos_curr):
        if len(self.subgoals_2d) < 2:
            return subgoals
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        width, height = image.shape[1], image.shape[0]
        pts_2d_map = torch.from_numpy(np.stack(np.meshgrid(np.arange(width - 1), np.arange(height - 1)), axis=-1))
        pts_2d = pts_2d_map.reshape(-1, 2)
        mask = np.zeros((height - 1, width - 1)).astype(np.bool_)
        g1 = self.pos_curr_2d
        g2 = self.subgoals_2d[1]
        x_min = np.min([max(int(g1[0]) - 30, 0), min(int(g2[0]) + 30, width - 1)])
        y_min = np.min([max(int(g1[1]) - 30, 0), min(int(g2[1]) + 30, height - 1)])
        x_max = np.max([max(np.ceil(g1[0]) - 30, 0), min(np.ceil(g2[0]) + 30, width - 1)])
        y_max = np.max([max(np.ceil(g1[1]) - 30, 0), min(np.ceil(g2[1]) + 30, height - 1)])
        mask[int(y_min): int(y_max), int(x_min): int(x_max)] = True
        # mask[depth[:height - 1, :width - 1] < -5.] = False
        if not mask.any():
            return subgoals
        if len(subgoals) <= 1:
            return subgoals
        pts_2d = pts_2d[mask.reshape(-1)]
        avoid_pts = to_3d(pts_2d, depth, cmat)
        
        subgoals2 = [subgoals[0]]
        g1 = pos_curr
        ## TODO:
        g2 = subgoals[1]
        for step_idx in range(5):
            force_pull = g2 - g1
            force_pull_mag = np.clip(np.linalg.norm(force_pull) ** 2, a_max=1, a_min=0.02)
            force_pull = force_pull / np.linalg.norm(force_pull) * force_pull_mag
            mask2 = np.linalg.norm(avoid_pts - g1, axis=-1) < 0.2
            if mask2.any():
                force_push = (avoid_pts[mask2] - g1).mean(0)
                force_push_mag = np.clip(1. / (np.linalg.norm(force_push) + 1e-5) ** 3, a_min=0., a_max=3.0) * 0 + 0.005
                force_push = force_push / np.linalg.norm(force_push) * force_push_mag
                force_overall = force_pull - force_push
            else:
                force_overall = force_pull
            g3 = g1 + force_overall / np.linalg.norm(force_overall ) * 0.05
            g1 = g3
            subgoals2.append(g3)
        ## TODO: vis
        '''
        import matplotlib.pyplot as plt
        sphere_g1 = pos_curr + np.random.rand(100, 3) * 0.02
        sphere_g2 = subgoals[1] + np.random.rand(100, 3) * 0.02
        colors = image[:height - 1, :width - 1, :][mask]
        colors_g1 = np.ones((100, 3)) * np.array([[255, 0, 0]])
        colors_g2 = np.ones((100, 3)) * np.array([[0, 255, 0]])
        sphere_subgoals = []
        colors_subgoals = []
        for subgoal in subgoals2:
            colors_sub = np.ones((100, 3)) * np.array([255, 255, 0])
            sphere_sub = subgoal + np.random.rand(100, 3) * 0.02
            sphere_subgoals.append(sphere_sub)
            colors_subgoals.append(colors_sub)
        sphere_subgoals = np.concatenate(sphere_subgoals, axis=0)
        colors_subgoals = np.concatenate(colors_subgoals, axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        pts = np.concatenate([avoid_pts, sphere_g1, sphere_g2, sphere_subgoals], axis=0)
        col = np.concatenate([colors, colors_g1, colors_g2, colors_subgoals], axis=0) / 256.
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col)
        plt.show()
        import ipdb;ipdb.set_trace()
        '''
        ## TODO: end
        subgoals2 = subgoals2 + subgoals[1:]
        return subgoals2

    def _parse_obs(self, obs):
        index = 29 if self.return_qpos else 3
        return {
            'hand_pos': obs[:index],
            'unused_info': obs[index:],
        }
        
    def get_action(self, obs, return_hand=False, qpos_control=False):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1
        action = {
            'grab_effort': 3,
        }
        desire_pos, desire_rot = self._desired_pos(o_d)
        # desire_qpos, _ = self.xyz_to_qpos(desire_pos, split_path=False)
        if self.return_qpos:
            action['delta_pos'] = move(
                self.get_ee_pos(), 
                to_xyz=desire_pos, 
                p=1
            )

            if desire_rot is None:
                action['rot'] = self.get_ee_quat()
                action['delta_rot'] = np.zeros(3,)
                action['delta_rot_value'] = None
            else:
                prev_quat =  self.get_ee_quat()
                action['delta_rot'] = move(
                    prev_quat,
                    to_xyz=desire_rot,
                    p=3,
                )
                action['delta_rot'] = np.clip(action['delta_rot'], a_min=-3, a_max=3)
                action['rot'] =  prev_quat + action['delta_rot']
                action['delta_rot_value'] = R.from_quat(action['delta_rot']).as_rotvec()[1]
           
            # direction = (desire_pos - self.get_ee_pos()) / np.linalg.norm(desire_pos - self.get_ee_pos())
            # from scipy.spatial.transform import Rotation as R
            # mocap_rotvec = R.from_quat(self.env.sim.data.mocap_quat).as_rotvec()[0]
            # mocap_rotvec[2] = np.pi / 2.
            # self.env.sim.data.set_mocap_quat('ee_mocap', R.from_rotvec(mocap_rotvec).as_quat())
            
        else:
            action['delta_pos'] = move(o_d['hand_pos'], 
                to_xyz=desire_pos, 
                p=20.
            )
        
        grab_effort = self._grab_effort(o_d)
        print('grab_effort:', grab_effort)
        print("delta_pos:", action["delta_pos"])
        action['grab_effort'] = grab_effort
        if return_hand:
            return action, o_d['hand_pos']
        else:
            return action['delta_pos']

    def _desired_pos(self, o_d):
        
        if self.cnt_wait < 0:
            self.cnt_wait += 1
            return self.previous_subgoal, self.previous_subgoal_rot

        if self.grasp_cnt < 0:
            self.grasp_cnt += 1
            print("cnt_wait:", self.grasp_cnt)
            return self.grasp_3d, None
        if not self.return_qpos:
            pos_curr = o_d['hand_pos']
        else:
            pos_curr = self.get_ee_pos()
        self.pos_curr = pos_curr
        self.rot_curr = rot_curr = self.env.sim.data.get_mocap_quat("ee_mocap")

        rot_precision = 0.05
        if self.is_traj:
            move_precision = 0.1
        else:
            move_precision = 0.1
        
        # if stucked/stopped(all subgoals reached), replan
        
        ## TODO: need refine
        # if self.need_refine:
        #     print("need refine, replan")
        #     self.subgoals = self.calculate_next_plan(first=True, pos_curr=pos_curr)
        #     self.subgoals, self.need_refine = self.process_flow(self.subgoals)
        #     return self.subgoals[0]


        if (self.replan_countdown <= 0 and self.replans > 0):
            print("replan")
            self.is_grasp = False
            self.grasp_lock = False
            self.grasp_cnt = 0
            self.tasks = self.tasks_all
            self.subgoals, self.subgoals_rot = self.calculate_next_plan(first=True, pos_curr=pos_curr)
            return self.subgoals[0], self.subgoals_rot[0]
        print("error:", np.linalg.norm((pos_curr - self.subgoals[0]) ))
        # if (np.linalg.norm((pos_curr - self.subgoals[0]) ) > move_precision and self.subgoals_rot[0] is None)\
        #     or (np.linalg.norm((pos_curr - self.subgoals[0]) ) > move_precision and self.subgoals_rot[0] is not None and np.linalg.norm((rot_curr - self.subgoals_rot[0]) ) > rot_precision):
        if np.linalg.norm((pos_curr - self.subgoals[0]) ) > move_precision:
            return self.subgoals[0], self.subgoals_rot[0]
        else:
            if len(self.subgoals) == 1:
                if len(self.tasks) > 1:
                    self.tasks = self.tasks[1:]
                    print('next task:', self.tasks[0])
                    self.subgoals, self.subgoals_rot = self.calculate_next_plan(pos_curr=pos_curr)
                    return self.subgoals[0], self.subgoals_rot[0]
                else:
                    ## TODO:
                    # if self.is_traj:
                    #     # previous_subgoal = self.subgoals[0]
                    #     direction = self.subgoals[0] - self.previous_subgoal
                    #     direction /= np.linalg.norm(direction)
                    #     self.subgoals = [
                    #         # self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 2,
                    #         # self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 20,
                    #         # self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 35
                    #     ]
                    #     self.previous_subgoal = previous_subgoal
                    return self.subgoals[0], self.subgoals_rot[0]

            self.previous_subgoal = self.subgoals[0]
            self.subgoals = self.subgoals[1:]
            self.previous_subgoal_rot = self.subgoals_rot[0]
            self.subgoals_rot = self.subgoals_rot[1:]
            
            ## TODO: start from previous grasp
            # if self.is_traj:
            #     self.subgoals = [subgoal + (self.get_ee_pos() + np.array([-0.04, 0.07, 0.]) - self.previous_subgoal) for subgoal in self.subgoals]
                # self.vis_subgoals(self.subgoals)
            # self.cnt_wait = -50

            # self.subgoals[0] = self.refine_subgoal(self.subgoals[0])
            # self.pred_image_depth_segms = self.pred_image_depth_segms[1:]
            return self.subgoals[0], self.subgoals_rot[0]
    '''
    def refine_2d(self, uv2, pt1, depth2, depth1, cmat):
        uv2 = uv2.astype(np.int64)
        pt2 = to_3d(uv2, depth2, cmat)
        current_direction = pt2 - pt1
        traj_len = np.linalg.norm(current_direction)
        LEN_THRESH = 0.1
        if self.previous_direction is None or traj_len < LEN_THRESH:
            if traj_len > LEN_THRESH:
                self.previous_direction = current_direction
            return uv2, False
        THRESH = 0.4
        angle = (current_direction * self.previous_direction).sum() / np.linalg.norm(self.previous_direction) / np.linalg.norm(current_direction)
        if angle > THRESH:
            self.previous_direction = current_direction
            return uv2, False
        else:
            DEBUG = True
            uvs = torch.stack(torch.meshgrid((torch.arange(uv2[1] - 10, uv2[1] + 10), torch.arange(uv2[0] - 5, uv2[0] + 5))), dim=-1).view(-1, 2).detach().cpu().numpy()
            pts = to_3d(uvs, depth2, cmat)
            current_directions = pts - pt1
            angles = (current_directions * self.previous_direction).sum(-1) / np.linalg.norm(self.previous_direction) / np.linalg.norm(current_directions, axis=-1)
            disp_2d = np.linalg.norm(uvs - uv2, axis=-1)
            uv2 = uvs[angles > THRESH]
            disp_2d = disp_2d[angles > THRESH]
            uv2 = uv2[np.argsort(disp_2d)[0]]
            pt2 = to_3d(uv2, depth2, cmat)
            current_direction = pt2 - pt1
            self.previous_direction = current_direction
            return uv2, True
    '''

    def refine_2d(self, uv2s, uv1s, depth2, depth1):
        uv2s = uv2s.astype(np.int64)
        uv1s = uv1s.astype(np.int64)
        THRESH = 0.1
        d1s = depth1[uv1s[:, 1], uv1s[:, 0]]
        d2s = depth2[uv2s[:, 1], uv2s[:, 0]]
        depth_changes = np.abs(d1s - d2s)
        uv2s = uv2s[depth_changes < THRESH]
        return uv2s, False

    def process_flow(self, subgoals):
        pred_image_depth_segm = self.pred_image_depth_segms[0]
        depth = pred_image_depth_segm[..., 3]
        depth *= (self.depth_high - self.depth_low)
        depth += self.depth_low
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        segm = np.zeros((depth.shape[0], depth.shape[1]))
        segm[data[:, :, -1] == 51] = 1
        segm[data[:, :, -1] == 53] = 2
        segm1 = (segm == 1).astype(depth.dtype)
        segm2 = (segm == 2).astype(depth.dtype)
        flow = subgoals[0]
        grasp_2ds = self.grasp_2ds
        subgoal_2ds = grasp_2ds + flow[grasp_2ds[:, 1].astype(np.int64), grasp_2ds[:, 0].astype(np.int64)]
        subgoal_2d_coarse = np.median(subgoal_2ds, axis=0)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        subgoal_2ds, need_refine = self.refine_2d(subgoal_2ds, grasp_2ds, depth, self.previous_depth)
        self.grasp_2ds = subgoal_2ds
        subgoal_2d = np.median(subgoal_2ds, axis=0)
        subgoal_2d = subgoal_2d_coarse
        self.grasp_2d = subgoal_2d
        
        # ## TODO: visualization
        image = pred_image_depth_segm[..., :3]
        import cv2
        # image = cv2.circle(image.astype(np.uint8).copy(), [int(subgoal_2d_coarse[0]), int(subgoal_2d_coarse[1])], radius=3, color=[255, 255, 0], thickness=-1)
        # image = cv2.circle(image.astype(np.uint8).copy(), [int(subgoal_2d[0]), int(subgoal_2d[1])], radius=3, color=[255, 0, 0], thickness=-1)
        for i in range(len(subgoal_2ds)):
            s2d = subgoal_2ds[i]
            image = cv2.circle(image.astype(np.uint8).copy(), [int(s2d[0]), int(s2d[1])], radius=1, color=[255, 0, 0], thickness=-1)

        cv2.imwrite("debug.png", image)
        # if DEBUG:
        # import ipdb;ipdb.set_trace()

        subgoal_3ds = to_3d(subgoal_2ds, depth, cmat)
        subgoal_3d = np.median(subgoal_3ds ,axis=0)

        ## TODO: xyz flow
        # rgb2 = torch.from_numpy(pred_image_depth_segm[..., :3]).cuda()
        # pts_3d2 = torch.from_numpy(np.stack(np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0])), axis=-1)).cuda()
        # pts_3d2 = torch.cat([pts_3d2, torch.from_numpy(depth).cuda().unsqueeze(-1)], dim=-1)
        # pts_3d2 = pts_3d2[torch.logical_and(pts_3d2[..., -1] > -2.5, pts_3d2[..., -1] < -1.5)]
        # pts_3d2 = pts_3d2.detach().cpu().numpy()
        # indices = np.random.choice(pts_3d2.shape[0], size=min(8000, pts_3d2.shape[0]), replace=False)
        # pts_3d2 = torch.from_numpy(pts_3d2[indices]).cuda()
        # rgb1 = torch.from_numpy(self.previous_image).cuda()
        # pts_3d1 = torch.from_numpy(np.stack(np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0])), axis=-1)).cuda()
        # pts_3d1 = torch.cat([pts_3d1, torch.from_numpy(self.previous_depth).cuda().unsqueeze(-1)], dim=-1)
        # pts_3d1 = pts_3d1[torch.logical_and(pts_3d1[..., -1] > -2.5, pts_3d1[..., -1] < 1.5)]
        # indices = np.random.choice(pts_3d1.shape[0], size=min(8000, pts_3d1.shape[0]), replace=False)
        # pts_3d1 = pts_3d1.detach().cpu().numpy()
        # pts_3d1 = torch.from_numpy(pts_3d1[indices]).cuda()
        # rgbs = torch.cat([rgb1, rgb2], dim=-1).unsqueeze(0).permute(0, 3, 1, 2)
        # # pts = torch.cat([pts_3d1, pts_3d2] ,dim=-1).unsqueeze(0).permute(0, 2, 1)
        # inputs = {'images': rgbs, "pcs": pts}
        # outputs = self.scene_flow_model(inputs)
        # flow_3d = outputs['flow_3d'][0].detach().cpu().numpy().transpose()
        # pts_3d3 = pts_3d1.detach().cpu().numpy() + flow_3d
        # pts_3d3 = to_3d2(pts_3d3[:, :2], pts_3d3[..., -1], cmat)
        # pts_3d1 = pts_3d1.detach().cpu().numpy()
        # pts_3d1 = to_3d2(pts_3d1[:, :2], pts_3d1[..., -1], cmat)
        # pts_3d2 = pts_3d2.detach().cpu().numpy()
        # pts_3d2 = to_3d2(pts_3d2[:, :2], pts_3d2[..., -1], cmat)
        # ## TODO: visualization
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_3d1)
        # o3d.io.write_point_cloud('debug1.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(pts_3d2)
        # o3d.io.write_point_cloud('debug2.ply', pcd)
        # pcd.points = o3d.utility.Vector3dVector(pts_3d3)
        # o3d.io.write_point_cloud('debug3.ply', pcd)
        # import ipdb;ipdb.set_trace()

        ## TODO:
        self.subgoal_2d = subgoal_2d
        if not isinstance(subgoals, list):
            subgoals = [subgoal for subgoal in subgoals]
        subgoals[0] = subgoal_3d
        self.previous_depth = depth
        self.previous_image = pred_image_depth_segm[..., :3]
        return subgoals, need_refine

    def _grab_effort(self, o_d):
        if self.return_qpos:
            pos_curr = self.get_ee_pos()
        else:
            pos_curr = o_d['hand_pos']

        if self.grasp_lock and self.is_traj and not self.grasping:
            self.grasp_cnt = -30
            self.grasping = True
            return 0.8
        if self.is_traj and self.grasping:
            return 0.8
        if self.is_grasp:
            print('to grasp dist:', np.linalg.norm(pos_curr - self.grasp_3d))
            if (np.linalg.norm(pos_curr[2] - self.grasp_3d[2]) < 0.08 and not self.return_qpos)\
                or (np.linalg.norm(pos_curr - self.grasp_3d) < 0.2 and self.return_qpos):
                self.grasp_lock = True
                return -0.8
            else:
                return -0.8
        else:
            if 'grasp' in self.tasks_all:
                return -0.8
            else:
                return 0.8



class MyPolicy_CL_seg(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, seg_model, resolution=(320, 240), plan_timeout=15, max_replans=0, log=False):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.seg_model = seg_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.max_replans = max_replans
        self.replans = max_replans + 1
        self.time_from_last_plan = 0
        self.log = log

        grasp, transforms = self.calculate_next_plan()
        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def get_seg(self, resolution):
        image, _ = self.env.render(depth=True, camera_name=self.camera)
        image = Image.fromarray(image)
        with open("seg_text.json", "r") as f:
            seg_text = json.load(f)
            text_prompt = seg_text[self.task]

        with torch.no_grad():
            masks, *_ = self.seg_model.predict(image, text_prompt)
            mask = masks[0].cpu().numpy()
        # resize to resolution
        mask = cv2.resize(mask.astype('uint8') * 255, resolution)
        # convert to binary
        mask = (mask > 0)
        return mask


    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        # seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)
        seg = self.get_seg(self.resolution)

        # measure time for vidgen
        start = time.time()
        images = pred_video(self.video_model, image, self.task)
        time_vid = time.time() - start

        # measure time for flow
        start = time.time()
        image1, image2, color, flow, flow_b = pred_flow_frame(self.flow_model, images, device="cuda:0")
        time_flow = time.time() - start

        # measure time for action planning
        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
        time_action = time.time() - start

        t = len(transform_mats)
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)

        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            return
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        self.time_from_last_plan += 1

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        if self.return_qpos:
            action['delta_pos'] = (self._desired_pos(o_d) - o_d['hand_pos']) * 0.02
            # action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20)
        else:
            action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']
        
        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8



class MyPolicy_Flow(Policy):
    def __init__(self, env, task, camera, video_flow_model, resolution=(320, 240), plan_timeout=15, max_replans=0):
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_flow_model = video_flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.replans = max_replans + 1

        grasp, transforms = self.calculate_next_plan()
        grasp = grasp[0]

        subgoals = self.calc_subgoals(grasp, transforms)
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        if max_deltaz > 0.1:
            self.mode = "grasp"
        else:
            self.mode = "push"
            # move the gripper down a bit for robustness
            # (Attempt to use the gripper wrist for pushing, not fingers)
            subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        self.grasp = grasp
        self.subgoals = subgoals
        self.init_grasp()  

    def calc_subgoals(self, grasp, transforms):
        subgoals = [grasp]
        for transforms in transforms:
            grasp_ext = np.concatenate([subgoals[-1], [1]])
            next_subgoal = (transforms @ grasp_ext)[:3]
            subgoals.append(next_subgoal)
        return subgoals

    def calculate_next_plan(self):
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        image, depth = self.env.render(resolution=self.resolution, depth=True, camera_name=self.camera)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)

        flows = pred_video(self.video_flow_model, image, self.task, flow=True)

        grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flows)

        transform_mats = [get_transformation_matrix(*transform) for transform in transforms]

        return grasp, transform_mats

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
    def init_grasp(self):
        self.grasped = False
        if self.mode == "push":
            for subgoal in self.subgoals:
                norm = np.linalg.norm(subgoal[:2] - self.grasp[:2])
                direction = subgoal[:2] - self.grasp[:2]
                direction = direction / norm
                if norm > 0.1:
                    break
            self.grasp[:2] = self.grasp[:2] - direction * 0.08

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        # if stucked (not moving), step the countdown
        if np.linalg.norm(o_d['hand_pos'] - self.last_pos) < 0.001:
            self.replan_countdown -= 1
        self.last_pos = o_d['hand_pos']

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=20.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        move_precision = 0.12 if self.mode == "push" else 0.04

        # if stucked/stopped(all subgoals reached), replan
        if self.replan_countdown <= 0 and self.replans > 0:
            grasp, transforms = self.calculate_next_plan()
            self.grasp = grasp[0]
            self.subgoals = self.calc_subgoals(grasp[0], transforms)
            if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        # place end effector above object
        elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
            return self.grasp + np.array([0., 0., 0.2])
        # drop end effector down on top of object
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
            return self.grasp
        # grab object (if in grasp mode)
        elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
            self.grasped = True
            return self.grasp
        # move end effector to the current subgoal
        elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        # if close enough to the current subgoal, move to the next subgoal
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            return self.subgoals[0]
        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
            
        
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']

        if self.grasped or self.mode == "push" or not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        else:
            return -0.8
