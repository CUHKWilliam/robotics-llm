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

import subprocess
class MyPolicy_CL_rgbd(Policy):
    def __init__(self, env, task, video_model, flow_model, dataset, resolution=(640, 480), plan_timeout=20, log=False, return_qpos=False):
        self.dataset = dataset
        self.depth_max, self.depth_min = self.dataset.depth_max, self.dataset.depth_min
        plan_timeout = 100
        self.env = env
        self.full_task_name = task

        self.video_model = video_model
        self.flow_model = flow_model
        self.resolution = resolution
        self.plan_timeout = plan_timeout
        self.return_qpos = return_qpos
        self.last_pos = np.array([0, 0, 0, 0, 0, 0, 0])
        self.log = log
        self.phase_grasp = True

        self.task = task
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
        self.time_from_last_plan = 0

    def _parse_obs(self,obs):
        return None
    
    def init_grasp(self):
        self.grasped = False

    def get_ee_pos(self):
        return self.env.get_obs_remote()['ee_pos']
        
    def get_ee_quat(self):
        return R.from_euler("ZYX", self.env.get_obs_remote()['ee_euler']).as_quat()

    def calculate_next_plan(self, ):
        self.task = self.tasks[0]
        
        if self.task.split(' ')[0] == 'locate':
            self.is_traj = False
            subgoals = self.calculate_locate(task = self.task)
            subgoals_rot = [None for _ in range(len(subgoals))]
        elif self.task.split(' ')[0] == 'grasp':
            self.is_traj = False
            self.is_grasp = True
            self.grasp_lock = False
            self.grasp_cnt = 0
            # subgoals = self.calculate_grasp_predict()
            subgoals, subgoals_rot = self.calculate_grasp(name = self.task)
        else:
            self.grasping = False
            self.is_traj = True
            subgoals, subgoals_rot = self.calculate_traj()
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


    def calculate_locate(self, task=None):
        obs =  self.env.get_obs_remote()
        image, depth = obs['color_image'], obs["depth_image"]

        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high
        self.previous_image = image.copy()
        self.previous_depth = depth.copy()
        
        seg = self.predict_segm(image)
        seg = cv2.erode(seg.astype(np.uint8) * 255, np.ones((2, 2), np.uint8) ).astype(np.bool_)
        self.previous_seg = seg.copy()

        seg = torch.from_numpy(seg).float()
        width, height = image.shape[1], image.shape[0]
        pts_2d_all = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        image[seg.detach().cpu().numpy() ==1] = np.array([255, 0, 0])
        cv2.imwrite('debug.png', image)
        
        pts_2d = pts_2d_all[seg > 0]
        if (seg > -0).sum() == 0:
            return None

        pts_3d = self.env.get_point_cloud(pts_2d, depth)
        # TODO:
        pts_3d = torch.median(torch.from_numpy(pts_3d), dim=0)[0].detach().cpu().numpy()
        self.grasp_3d = pts_3d
        self.grasp_2ds = pts_2d.detach().cpu().numpy()
        subgoals = [self.grasp_3d +  np.array([0, 0., 0.])]
        self.subgoals_2d = subgoals
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals

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


    def calculate_grasp(self,):
        pts_3d = self.grasp_3d + np.array([0., 0., 0])
        ## TODO: grasp pose estimation
        subgoals = [pts_3d]
        seg = self.previous_seg
        depth =self.previous_depth
        seg_dilate = cv2.dilate(seg.copy().astype(np.uint8), np.ones((30, 30))) > 0

        width, height = seg_dilate.shape[1], seg_dilate.shape[0]
        pts_2d_ = torch.from_numpy(np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1))
        pts_2d = pts_2d_[seg_dilate > 0]
        pts_2d_obj = pts_2d_[seg > 0]
        pts_3ds = self.env.get_point_cloud(pts_2d, depth, )
        pts_3ds_obj = self.env.get_point_cloud(pts_2d_obj, depth , )
        rot_quat = self.grasp_pose_estimation(pts_3ds, pts_3ds_obj)

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
        return subgoals, [rot_quat]
    
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
        rotvec[rotvec > 50] = 0
        rot_quat = R.from_rotvec(rotvec).as_quat()
        return rot_quat, rot_mat

    def calculate_traj(self):
        obs = self.env.get_obs_remote()
        image, depth = obs['color_image'], obs['depth_image']
        depth[depth < self.depth_low] = self.depth_low
        depth[depth > self.depth_high] = self.depth_high

        self.previous_image = image.copy()
        self.previous_depth = depth.copy()

        self.dataset.normalize_depth(depth)
        

        segm = np.zeros((image.shape[0], image.shape[1]))

        segm1 = (segm == 1).astype(depth.dtype)
        segm1[self.grasp_2ds[:, 1], self.grasp_2ds[:, 0]] = 1.
        segm2 = (segm == 2).astype(depth.dtype) * 0.

        depth_orig_size = depth.shape
        image_depth_segm = np.concatenate([image, depth[::-1, :, None], segm1[::-1, :, None], segm2[..., None]], axis=-1)
        
        # measure time for vidgen
        start = time.time()
        images, depths, segms1, segms2 = pred_video_rgbd_fk(self.video_model, image_depth_segm, self.task)

      
        images = F.interpolate(torch.from_numpy(images), depth_orig_size, ).detach().cpu().numpy()
        video = torch.tensor(images)[None].float().cuda()
        ## TODO:
        # self.grasp_2ds = np.expand_dims(self.grasp_2d, axis=0)
        image = (self.image).detach().cpu().numpy().astype(np.int64)
        
        queries = torch.cat([torch.tensor([0] * image.shape[0]).unsqueeze(-1), torch.from_numpy(image)], dim=-1).cuda().float()
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
        depths = F.interpolate(depths, depth_orig_size)
        ## TODO:
        # depths[0] = torch.from_numpy(depth).unsqueeze(0)
        height = depths.shape[-2]
        prev_sgs = None
        sgs_seq = []
        for i in range(len(depths)):
            subgoals_2d[i][:, 1] = height - subgoals_2d[i][:, 1]
            subgoals_2d[i] = np.clip(subgoals_2d[i], a_min=0, a_max=255)
            sgs = self.env.get_point_cloud(subgoals_2d[i], (self.dataset.unnormalize(depths[i][0])))
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
        #     self.grasp_3d = self.get_ee_pos() # + np.array([-0.04, 0.07, 0])
        #     subgoals = [subgoal  + (self.grasp_3d - subgoals[0]) for subgoal in subgoals]

        if 'grasp' not in self.tasks_all:
            direction = (self.grasp_3d - subgoals[-1]) / np.linalg.norm(self.grasp_3d - subgoals[-1])
            subgoals = [self.grasp_3d + 0.2 * direction] + [self.grasp_3d] + subgoals
            subgoals_rot = [None, None] + subgoals_rot

        time_vid = time.time() - start

        # measure time for action planning
        time_action = time.time() - start
        self.replans -= 1
        self.replan_countdown = self.plan_timeout
        self.time_from_last_plan = 0
        return subgoals, subgoals_rot

    def get_action_remote(self, obs):
        self.server_name = self.env.server_name
        self.data_file_path = self.env.data_file_path
        self.lock_file_path = self.env.lock_file_path
        with open("tmp.pkl", "wb") as f:
            pickle.dump(obs, f)
        subprocess.Popen('scp -r -P 22 {} {}:{}'.format("tmp.pkl", self.server_name, self.data_file_path), shell=True)
        with open("tmp.txt", 'w') as f:
            f.write("1")
        subprocess.Popen('scp -r -P 22 {} {}:{}'.format("tmp.txt", self.server_name, self.lock_file_path), shell=True)
        time.sleep(0.5)
        while True:
            subprocess.Popen('scp -r -P 22 {}:{} {}'.format(self.server_name, self.lock_file_path, "tmp.txt"), shell=True)
            with open("tmp.txt", 'r') as f:
                content = f.read().strip()
            if content == "0":
                break
        subprocess.Popen('scp -r -P 22 {}:{} {}'.format(self.server_name, self.data_file_path, "tmp.pkl"), shell=True)
        with open("tmp.pkl", 'rb') as f:
            action = pickle.load(f)
        return action
    
    def get_action(self, obs):
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
        action['pos'] = desire_pos
        if desire_rot is None:
            action['rot'] = self.get_ee_quat()
        else:
            action['rot'] = desire_rot

        grab_effort = self._grab_effort(o_d)
        print('grab_effort:', grab_effort)
        action['grab_effort'] = grab_effort
        return action

    def _desired_pos(self, o_d):

        if len(self.subgoals) == 1:
            if len(self.tasks) > 1:
                self.tasks = self.tasks[1:]
                print('next task:', self.tasks[0])
                self.subgoals, self.subgoals_rot = self.calculate_next_plan()
                return self.subgoals[0], self.subgoals_rot[0]
            else:
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


