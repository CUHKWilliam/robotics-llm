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

from sklearn.cluster import KMeans
class MyPolicy_CL_rgbd(Policy):
    def __init__(self, env, task, camera, video_model, flow_model, resolution=(640, 480), plan_timeout=20, max_replans=5, log=False):
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
        self.phase_grasp = True

        _, subgoals = self.calculate_next_plan()

        # self.grasp = grasp
        ### for stablity, extrapolate the last subgoal
        # next_subgoal = subgoals[-1] + (subgoals[-1] - subgoals[-2])
        # subgoals.append(next_subgoal)
        subgoals_np = np.array(subgoals)
        # print(subgoals_np)
        self.mode = 'grasp'
        # try:
        #     max_deltaz = abs(subgoals_np[1:-2, 2] - subgoals_np[2:-1, 2]).max()
        # except:
        #     import ipdb;ipdb.set_trace()
        # if max_deltaz > 0.1:
        #     self.mode = "grasp"
        # else:
        #     self.mode = "push"
        #     # move the gripper down a bit for robustness
        #     # (Attempt to use the gripper wrist for pushing, not fingers)
        #     subgoals = [s - np.array([0, 0, 0.03]) for s in subgoals]
        
        subgoals = [x+np.array([0,0,0.0]) for x in subgoals]
        self.subgoals = subgoals 
        # self.subgoals = self.subgoals[3:6]

        self.init_grasp()  

    def calculate_next_plan(self):
        '''
        if self.phase_grasp:
            subgoals = self.calculate_grasp()
            if subgoals is None:
                self.phase_grasp = False
                subgoals = self.calculate_traj()
            else:
                if self.mode=='push':
                    subgoals = subgoals[:-1]
        else:
            subgoals = self.calculate_traj()
        '''
        subgoals = self.calculate_traj()
        return subgoals[0], subgoals
    
    def calculate_grasp(self):
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        low, high = -8., -1.5
        depth[depth < low] = low
        depth[depth > high] = high
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        # segm = np.zeros((image.shape[0], image.shape[1]))
        # segm[data[:, :, -1] == 51] = 1
        # segm[data[:, :, -1] == 53] = 2
        # segm1 = (segm == 1).astype(depth.dtype)
        # segm2 = (segm == 2).astype(depth.dtype)
        # image_depth_segm = np.concatenate([image, depth[..., None], segm1[..., None], segm2[..., None]], axis=-1)
        
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=list(np.array(self.seg_ids) + 20))
        seg = torch.from_numpy(seg).float()
        
        width, height = image.shape[1], image.shape[0]
        pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d[seg > 0]
        if (seg > 0).sum() == 0:
            return None
        pts_3d = to_3d(pts_2d, depth, cmat)
        ## TODO:
        length = (np.max(pts_3d, axis=0) - np.min(pts_3d, axis=0)).max()
        if length > 0.1:
            self.mode = 'push'
        else:
            self.mode = 'grasp'
        self.mode = 'push'
        ## TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy() + np.array([0, 0, 0.0])
        self.grasp = pts_3d
        subgoals = [pts_3d + np.array([0, 0, 0.2]), pts_3d]
        

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


    def calculate_traj(self):
        image, depth = self.env.render(depth=True, camera_name=self.camera, body_invisible=True, resolution=self.resolution)
        low, high = -8., -1.5
        depth[depth < low] = low
        depth[depth > high] = high
        depth -= low
        depth /= (high - low)
        data = self.env.render(camera_name=self.camera, depth=True, body_invisible=True, segmentation=True, resolution=self.resolution)
        segm = np.zeros((image.shape[0], image.shape[1]))
        segm[data[:, :, -1] == 51] = 1
        segm[data[:, :, -1] == 53] = 2
        segm1 = (segm == 1).astype(depth.dtype)
        segm2 = (segm == 2).astype(depth.dtype)
        image_depth_segm = np.concatenate([image, depth[..., None], segm1[..., None], segm2[..., None]], axis=-1)
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        seg = get_seg(self.env, resolution=self.resolution, camera=self.camera, seg_ids=self.seg_ids)
        
        
        # measure time for vidgen
        start = time.time()
        images, depths, segms1, segms2 = pred_video_rgbd(self.video_model, image_depth_segm, self.task)
        segms = np.zeros_like(segms1)
        thresh = 0.1
        segms[segms1 > thresh] = 1.
        segms[segms2 > thresh] = 2.
        self.pred_image_depth_segm = np.concatenate([images, depths, segms], axis=1).transpose(0, 2, 3, 1)

        time_vid = time.time() - start
        subgoals = []
        for i in range(len(segms1)):
            height, width = images.shape[-2] - 1, images.shape[-1] - 1
            
            segm1, segm2 = segms1[i][0][:height, :width], segms2[i][0][:height, :width]
            mask1, mask2 = segm1 > thresh, segm2 > thresh
            
            if not mask1.any() or not mask2.any():
                continue
            mask1, mask2 = mask1.view(-1), mask2.view(-1)
            pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
            pts_2d = pts_2d.reshape(-1, 2)
            pts_2d_1 = pts_2d[mask1]
            pts_2d_2 = pts_2d[mask2]
            low, high = -8., -1.5
            depth = depths[i][0]
            depth *= (high - low)
            depth += low
            pts1 = to_3d(pts_2d_1, depth, cmat)
            pts2 = to_3d(pts_2d_2, depth, cmat)
            kmeans = KMeans(n_clusters=min(5, len(pts1))).fit(pts1)
            label = np.argmax(np.bincount(kmeans.labels_))
            pts1 = pts1[kmeans.labels_ == label]
            kmeans = KMeans(n_clusters=min(5, len(pts2))).fit(pts2)
            label = np.argmax(np.bincount(kmeans.labels_))
            pts2 = pts2[kmeans.labels_ == label]
            pos1, pos2 = np.mean(pts1, axis=0), np.mean(pts2, axis=0)
            subgoal = (pos1 + pos2) / 2.
            subgoals.append(subgoal)
            
            ## TODO: for visualization
            '''
            segm1, segm2 = segms1[i][0], segms2[i][0]
            pts_2d = torch.from_numpy(np.stack(np.meshgrid(np.arange(images.shape[-1]-1), np.arange(images.shape[-2]-1)), axis=-1))
            pts_2d = pts_2d.reshape(-1, 2)
            image = images[i].transpose((1, 2, 0))
            pts = to_3d(pts_2d, depth, cmat)
            segm1 = segm1[:images.shape[-2]-1, :images.shape[-1]-1].detach().cpu().numpy()
            segm2 = segm2[:images.shape[-2]-1, :images.shape[-1]-1].detach().cpu().numpy()
            cols = np.ones((segm1.shape[0], segm1.shape[1], 3))
            cols = image[:image.shape[0]-1, :image.shape[1]-1, :] / 256.
            mask1, mask2 = segm1 > 0.1, segm2 > 0.1
            cols[mask1] = np.array([1, 0, 0])
            cols[mask2] = np.array([0, 1, 0])
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            
            pcd_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            pcd_sphere.translate(pos1)
            pts_sphere = np.asarray(pcd_sphere.sample_points_uniformly(500).points)
            pts = np.concatenate([pts, pts_sphere], axis=0)
            cols = cols.reshape(-1, 3)
            cols = np.concatenate([cols, np.array([[1, 1, 0]] * len(pts_sphere))], axis=0)
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            o3d.io.write_point_cloud('debug{}.ply'.format(i), pcd)
            import ipdb;ipdb.set_trace()
            '''
            ## TODO: end TODO

        # measure time for action planning
        time_action = time.time() - start
        if self.log: log_time(time_vid/t, time_flow/t, time_action/t, self.max_replans-self.replans+1)
        if self.log and (self.time_from_last_plan!=0): log_time_execution(self.time_from_last_plan*0.1/t, self.max_replans-self.replans)
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        if len(subgoals) == 0:
            subgoals = [np.array([0,0,0])]
        # elif len(subgoals) > 2:
        #     subgoals[-1] = subgoals[-2] + 5 * (subgoals[-2]-subgoals[-3])

        # elif len(subgoals) == 2:
        #     subgoals.append(subgoals[-1] + 5 * (subgoals[-1]-subgoals[-2]))
        # if len(subgoals) > 1:
        #     subgoals.append(subgoals[-1] + 5 * (subgoals[-1]-subgoals[-2]))
        # subgoals = self.safe_subgoals(subgoals, image, depth)
        
        ## TODO:
        if not hasattr(self, 'mode') or self.mode == 'push':
            vec = self.grasp - subgoals[0]
            vec = 0.2 * vec / np.linalg.norm(vec)
            subgoal = [self.grasp + vec] + subgoals
        return subgoals
    
    def safe_subgoals(self, subgoals, image, depth):
        cmat = get_cmat(self.env, self.camera, resolution=self.resolution)
        width, height = image.shape[1], image.shape[0]
        pts_2d_map = torch.from_numpy(np.stack(np.meshgrid(np.arange(width - 1), np.arange(height - 1)), axis=-1))
        pts_2d = pts_2d_map.reshape(-1, 2)
        pts_2d = pts_2d[depth[pts_2d[:, 1], pts_2d[:, 0]] > -8.]
        avoid_pts = to_3d(pts_2d, depth, cmat)
        avoid_pts_map = torch.zeros([height, width, 3]) - 10000.
        avoid_pts_map[pts_2d[:, 1], pts_2d[:, 0], :] = torch.from_numpy(avoid_pts).float()
        
        subgoals2 = []
        for i in range(len(subgoals) - 1):
            subgoals2.append(subgoals[i])
            g1 = subgoals[i]
            g2 = subgoals[i + 1]
            import ipdb;ipdb.set_trace()
            route = torch.stack([torch.linspace(g1[0], g2[0], 10), torch.linspace(g1[1], g2[1], 10)],dim=-1)
            pts_in_route =  avoid_pts_map[route[:, 1].long(), route[:, 0].long(), :]
            z_max = pts_in_route[:, -1].max()
            xy_max = pts_in_route[pts_in_route[:, -1].argmax()]
            if g1 < z_max or g2 < z_max:
                subgolas2.append(torch.cat([xy_max, z_max]))
        subgoals2.append(subgoals[-1])
        return subgoals2

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
        move_precision = 0.04
        
        # if stucked/stopped(all subgoals reached), replan
        
        if self.replan_countdown <= 0 and self.replans > 0:
            if self.phase_grasp and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.1:
                self.phase_grasp = False
                print('grasp to move')
            else:
                print("replan")
                self.phase_grasp = True
            self.grasp, self.subgoals = self.calculate_next_plan()
            # if self.mode == "push": self.init_grasp()
            return self.subgoals[0]
        
        # # place end effector above object
        # elif not self.grasped and np.linalg.norm(pos_curr[:2] - self.grasp[:2]) > 0.02:
        #     return self.grasp + np.array([0., 0., 0.2])
        # # drop end effector down on top of object
        # elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) > 0.04:
        #     return self.grasp
        # # grab object (if in grasp mode)
        # elif not self.grasped and np.linalg.norm(pos_curr[2] - self.grasp[2]) <= 0.04:
        #     self.grasped = True
        #     return self.grasp
        # # move end effector to the current subgoal
        # elif np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
        #     return self.subgoals[0]
        # # if close enough to the current subgoal, move to the next subgoal
        # elif len(self.subgoals) > 1:
        #     ## TODO:
        #     self.subgoals.pop(0)
        #     return self.subgoals[0]
        
        if np.linalg.norm(pos_curr - self.subgoals[0]) > move_precision:
            return self.subgoals[0]
        elif len(self.subgoals) > 1:
            self.subgoals.pop(0)
            print("len:{}".format(len(self.subgoals)))
            return self.subgoals[0]

        # move to the last subgoal
        # ideally the gripper will stop at the last subgoal and the countdown will run out quickly
        # and then the next plan (next set of subgoals) will be calculated
        else:
            return self.subgoals[0]
               
    def _grab_effort(self, o_d):
        pos_curr = o_d['hand_pos']
        if self.phase_grasp and np.linalg.norm(pos_curr[2] - self.grasp[2]) < 0.08:
            return 0.8
        elif not self.phase_grasp:
            return 0.8
        else:
            return -0.8



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
