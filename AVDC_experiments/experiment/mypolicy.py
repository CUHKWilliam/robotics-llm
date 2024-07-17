from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.action import Action
import numpy as np
from metaworld_exp.utils import get_seg, get_cmat
import json
import cv2
from flowdiffusion.inference_utils import pred_video, pred_video_rgbd
from myutils import pred_flow_frame, get_transforms, get_transformation_matrix
import torch
from PIL import Image
from torchvision import transforms as T
import torch
import time
import pickle
import random

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
    depths = np.array([[sample_with_binear(depth, kp)] for kp in points])
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
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
sys.path.append("/media/msc-auto/HDD/wltang/robotics-llm/AVDC_experiments/experiment/CamLiFlow")
from factory import model_factory
scene_flow_model_config = {'name': 'camliraft', 'batch_size': 8, 'freeze_bn': False, 'backbone': {'depth': 50, 'pretrained': 'pretrain/resnet50-11ad3fa6.pth'}, 'n_iters_train': 10, 'n_iters_eval': 20, 'fuse_fnet': True, 'fuse_cnet': True, 'fuse_corr': True, 'fuse_motion': True, 'fuse_hidden': False, 'loss2d': {'gamma': 0.8, 'order': 'l2-norm'}, 'loss3d': {'gamma': 0.8, 'order': 'l2-norm'}}
scene_flow_model_config = argparse.Namespace(**scene_flow_model_config)
scene_flow_model_config.backbone = argparse.Namespace(**{"depth": 50, "pretrained": './CamLiFlow/pretrain/resnet50-11ad3fa6.pth'})
scene_flow_model_config.loss2d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})
scene_flow_model_config.loss3d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})

DEBUG = False
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").cuda()
from cotracker.utils.visualizer import Visualizer, read_video_from_path
import contact_detection
class MyPolicy_CL_rgbd(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(640, 480), plan_timeout=20, max_replans=5, log=False):
        self.depth_low = -8
        self.depth_high = -1.5
        plan_timeout = 40
        self.env = env
        self.seg_ids = name2maskid[task]
        self.task = " ".join(task.split('-')[:-3])
        self.camera = camera
        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.last_pos = np.array([0, 0, 0])
        self.last_pos2 = None
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
        # self.tasks = ['locate', task]
        self.tasks = ['locate', 'grasp', task]
        subgoals = []
        for i in range(len(self.tasks)):
            self.tasks[i] = self.tasks[i].strip()
        self.tasks_all = self.tasks
        self.is_traj = False
        subgoals = self.calculate_next_plan()
        self.mode = 'push'
        subgoals = [x+np.array([0,0,0.0]) for x in subgoals]
        self.subgoals = subgoals 
        self.subgoals_2d = None
        self.is_grasp = False
        self.grasp_cnt = 0
        self.grasp_lock = False
        self.need_refine = False
        self.cnt_wait = 0
    
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


    def calculate_next_plan(self, first=False, pos_curr=None):
        self.task = self.tasks[0]
        
        if self.task.split(' ')[0] == 'locate':
            self.is_traj = False
            subgoals = self.calculate_locate_predict(task = self.task)
        elif self.task.split(' ')[0] == 'grasp':
            self.is_traj = False
            self.is_grasp = True
            self.grasp_lock = False
            self.grasp_cnt = 0
            subgoals = self.calculate_grasp_predict()
        else:
            safe_travel = False
            self.is_traj = True
            subgoals = self.calculate_traj()
        
        # if not self.is_traj:
        #     subgoals = self.safe_subgoals(subgoals, self.previous_image, self.previous_depth, pos_curr)
        
        return subgoals

    def calculate_locate(self, name=None):
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
        if isinstance(self.seg_ids, dict):
            seg_ids = [self.seg_ids[name] + 20]
        else:
            seg_ids = list(np.array(self.seg_ids) + 20)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=seg_ids)
        seg = torch.from_numpy(seg).float()
        
        width, height = image.shape[1], image.shape[0]
        pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d[seg > 0]
        if (seg > 0).sum() == 0:
            return None
        pts_3d = to_3d(pts_2d, depth, cmat)
        # TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy() + np.array([0, 0, 0.0])
        self.grasp_3d = pts_3d
        subgoals = [pts_3d + np.array([0, 0, 0.2])]
        self.subgoals_2d = subgoals
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0

        return subgoals

    def calculate_locate_predict(self, task=None):
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        prompt = "Where should I grasp if I need to conduct task {} ? Please output segmentation mask.".format(task)
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

    def calculate_grasp(self, name=None):
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        low, high = -8., -1.5
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        
        pos_curr_2d_1 = np.stack(np.where(data[:, :, -1] == 51), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d_2 = np.stack(np.where(data[:, :, -1] == 53), axis=-1)[:, ::-1].mean(0)
        pos_curr_2d = (pos_curr_2d_1 + pos_curr_2d_2) / 2.
        self.pos_curr_2d = pos_curr_2d

        # segm = np.zeros((image.shape[0], image.shape[1]))
        # segm[data[:, :, -1] == 51] = 1
        # segm[data[:, :, -1] == 53] = 2
        # segm1 = (segm == 1).astype(depth.dtype)
        # segm2 = (segm == 2).astype(depth.dtype)
        # image_depth_segm = np.concatenate([image, depth[..., None], segm1[..., None], segm2[..., None]], axis=-1)
        
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        if isinstance(self.seg_ids, dict):
            seg_ids = [self.seg_ids[name]+ 20]
        else:
            seg_ids = list(np.array(self.seg_ids) + 20)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=seg_ids)
        seg = torch.from_numpy(seg).float()
        
        width, height = image.shape[1], image.shape[0]
        pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d[seg > 0]
        if (seg > 0).sum() == 0:
            return None
        pts_3d = to_3d(pts_2d, depth, cmat)
        ## TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy() + np.array([0, 0, 0.0])
        self.grasp_3d = pts_3d
        self.grasp_2ds = pts_2d.detach().cpu().numpy()
        self.grasp_2d = torch.median(pts_2d, dim=0)[0].detach().cpu().numpy()
        subgoals = [pts_3d + np.array([0, 0, 0.])]
        self.subgoals_2d = subgoals
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
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        low, high = -8., -1.5
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
        subgoals = [pts_3d + np.array([0, 0, -0.])]
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
        

    def calculate_traj(self):
        self.mode = 'push'
        self.previous_direction = None
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        depth -= self.depth_low
        depth /= (self.depth_high - self.depth_low)
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        segm = np.zeros((image.shape[0], image.shape[1]))
        segm[data[:, :, -1] == 51] = 1
        segm[data[:, :, -1] == 53] = 2
        segm1 = (segm == 1).astype(depth.dtype)
        segm2 = (segm == 2).astype(depth.dtype)

        image_depth_segm = np.concatenate([image, depth[..., None], segm1[..., None], segm2[..., None]], axis=-1)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        
        # measure time for vidgen
        start = time.time()
        images, depths, segms1, segms2 = pred_video_rgbd(self.video_model, image_depth_segm, self.task)
        video = torch.tensor(images)[None].float().cuda()
        ## TODO:
        # self.grasp_2ds = np.expand_dims(self.grasp_2d, axis=0)
        queries = torch.cat([torch.tensor([0] * self.grasp_2ds.shape[0]).unsqueeze(-1), torch.from_numpy(self.grasp_2ds)], dim=-1).cuda().float()
        pred_tracks, pred_visibility = cotracker(video, queries=queries[None])
        subgoals_2d = pred_tracks[0, :, :, :].detach().cpu().numpy()
        
        ## TODO: vis
        idx = -1
        image_debug = images[idx].copy().transpose(1, 2, 0).astype(np.uint8).copy()
        for i in range(len(subgoals_2d[idx])):
            image_debug = cv2.circle(image_debug, [int(subgoals_2d[idx, i, 0]), int(subgoals_2d[idx, i, 1])], radius=3, color=[255, 0, 0], thickness=-1)
        cv2.imwrite('debug.png', image_debug)
        import ipdb;ipdb.set_trace()

        # _, _, flow_debug, flow, flow_b = pred_flow_frame(self.flow_model, images)
        # self.visualize_ply(self.pred_image_depth_segms[0, :, :, :3], self.pred_image_depth_segms[0, :, :, 3] * (self.depth_high - self.depth_low) + self.depth_low, cmat, "debug1.ply")
        # self.visualize_ply(self.pred_image_depth_segms[1, :, :, :3], self.pred_image_depth_segms[1, :, :, 3] * (self.depth_high - self.depth_low) + self.depth_low, cmat, "debug2.ply")
        subgoals = []
        for i in range(len(depths)):
            sgs = to_3d(subgoals_2d[i], (depths[i][0] * (self.depth_high - self.depth_low)) + self.depth_low, cmat)
            sg = np.median(sgs, axis=0)
            subgoals.append(sg)
        direction = (self.grasp_3d - subgoals[4]) / np.linalg.norm(self.grasp_3d - subgoals[4])
        subgoals = [self.grasp_3d + 0.04 * direction] + subgoals
        time_vid = time.time() - start

        # measure time for action planning
        time_action = time.time() - start
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals
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

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_info': obs[3:],
        }
        
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
        desire_pos = self._desired_pos(o_d)
        action['delta_pos'] = move(o_d['hand_pos'], 
            to_xyz=self._desired_pos(o_d), 
            p=20.
        )
        grab_effort = self._grab_effort(o_d)
        action['grab_effort'] = grab_effort
        print('desire_pos:', desire_pos)
        print('grab_effort:', grab_effort)
        print("delta_pos:", action["delta_pos"])
        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        self.pos_curr = pos_curr
        if self.cnt_wait < 30:
            self.cnt_wait += 1
            return self.pos_curr
        if self.is_traj:
            move_precision = 0.08
        else:
            move_precision = 0.04
        
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
            self.subgoals = self.calculate_next_plan(first=True, pos_curr=pos_curr)
            return self.subgoals[0]

        # print(np.linalg.norm((pos_curr - self.subgoals[0])))
        if np.linalg.norm((pos_curr - self.subgoals[0]) ) > move_precision:
            return self.subgoals[0]
        else:
            if self.is_grasp:
                if self.grasp_cnt < 50:
                    self.grasp_cnt += 1
                    print("grasping")
                    return self.subgoals[0]
                else:
                    self.is_grasp = False
            if len(self.subgoals) == 1:
                if len(self.tasks) > 1:
                    self.tasks = self.tasks[1:]
                    print('next task:', self.tasks[0])
                    self.subgoals = self.calculate_next_plan(pos_curr=pos_curr)
                    return self.subgoals[0]
                else:
                    if self.is_traj:
                        previous_subgoal = self.subgoals[0]
                        self.subgoals = [
                            self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 10,
                            self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 20,
                            self.subgoals[0] + (self.subgoals[0] - self.previous_subgoal) * 35
                        ]
                        self.previous_subgoal = previous_subgoal
                    return self.subgoals[0]
            self.previous_subgoal = self.subgoals[0]
            self.subgoals = self.subgoals[1:]
            # self.subgoals[0] = self.refine_subgoal(self.subgoals[0])
            # self.pred_image_depth_segms = self.pred_image_depth_segms[1:]
            return self.subgoals[0]
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
        pos_curr = o_d['hand_pos']
        if self.grasp_lock:
            return 0.8
        if self.is_grasp:
            if np.linalg.norm(pos_curr[2] - self.grasp_3d[2]) < 0.08:
                self.grasp_lock = True
                return 0.8
            else:
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
