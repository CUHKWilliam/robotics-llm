from mypolicy import MyPolicy_CL_rgbd
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames, collect_video_rgbd
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import gym
from r3meval.utils.gym_env import GymEnv
from r3meval.utils.obs_wrappers import MuJoCoPixelObs, StateEmbedding
import mj_envs
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model_rgbd, pred_video
import random
import torch
from argparse import ArgumentParser
from real_world_env import RealWorldEnv
from datasets import SequentialDatasetv2_rgbd_with_segm_rw

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def run(args):
    env_name = args.env_name
    print(env_name)

    video_model = get_video_model_rgbd(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()
    return_qpos = True
    dummy_dataset = SequentialDatasetv2_rgbd_with_segm_rw(
        sample_per_seq=6, 
        path="../../AVDC/datasets/{}".format("realworld"), 
        target_size=(128, 128),
        randomcrop=True
    )
    policy = MyPolicy_CL_rgbd(env, env_name, video_model, flow_model, dataset=dummy_dataset, return_qpos=return_qpos)
    
    env = RealWorldEnv()
    obs = env.get_obs()

    while True:
        action = policy.get_action_remote(obs)
        env.set_action(action)
        reached = False
        while not reached:
            reached = env.step()
        obs = env.get_obs()
        color_image = obs['color_image']
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        cv2.imshow('demo', color_image)
        ch = cv2.waitKey(25)  
        if ch & 0xFF == ord('q') or ch == 27:
            cv2.destroyAllWindows()
            break
        while True:
            ch = cv2.waitKey(25)
            if ch & 0xFF == ord('c'):
                break

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld") 
    parser.add_argument("--milestone", default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()
    try:
        args.milestone = int(args.milestone)
    except:
        args.milestone = str(args.milestone)

    run(args)
        
