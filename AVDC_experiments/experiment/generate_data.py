from mypolicy import MyPolicy_CL
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model, pred_video
import random
import torch
from argparse import ArgumentParser

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def get_policy(env_name):
    name = "".join(" ".join(get_task_text(env_name)).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def run(args):
    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    resolution = (320, 240)
    cameras = ['corner',]

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    env_name = args.env_name
    print(env_name)
    seg_ids = name2maskid[env_name]
    benchmark_env = env_dict[env_name]
    for camera in cameras:
        cnt = 0
        for seed in tqdm(range(n_exps)):
            try:
                env = benchmark_env(seed=seed)
                
                obs = env.reset()
                policy = MyPolicy_CL(env, env_name, camera, video_model, flow_model, max_replans=0)

                images, depths, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
                
                if len(images) <= 500:
                    print("success")
                    SAVE_PATH = '/data/wltang/robotic-llm/AVDC/datasets/metaworld/metaworld_dataset_2'
                    save_path = SAVE_PATH
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, camera)
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, "{:03}".format(cnt))
                    os.makedirs(save_path, exist_ok=True)
                    for i in range(len(images)):
                        image, depth = images[i], depths[i]
                        depth = np.expand_dims(depth, axis=-1)
                        image_depth = np.concatenate([image, depth], axis=-1)
                        np.save(os.path.join(save_path, "{:03}.npy".format(i)), image_depth)
                    cnt += 1
            except:
                print('error')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()


    run(args)
        