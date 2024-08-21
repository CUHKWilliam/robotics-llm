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
import pickle

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import json
with open('name2maskid.json', 'r') as f:
    name2maskid = json.load(f)


def get_policy(env_name):
    name = "".join(get_task_text(env_name).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    policy = getattr(policies, policy_name)()
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

DATA_ROOT = "/media/msc-auto/HDD/wltang/robotics-llm/AVDC/datasets"
SAVE_PATH = '/media/msc-auto/HDD/wltang/robotics-llm/AVDC/datasets/franka-kitchen/franka-kitchen_dataset'

def run(args):

    n_exps = args.n_exps
    resolution = (320, 240)
    cameras = ['right_cap2']

    env_name = args.env_name
    print(env_name)
    for camera in cameras:
        for seed in tqdm(range(n_exps)):
            pickle_path = os.path.join(DATA_ROOT, "franka-kitchen", camera, "{}.pickle".format(env_name))
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            import ipdb;ipdb.set_trace()
            save_path = SAVE_PATH
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, env_name)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, camera)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "{:03}".format(cnt))
            os.makedirs(save_path, exist_ok=True)
            for i in range(len(images)):
                # image, depth, segm = images[i], depths[i], segms[i]
                image, depth, image_segm = images[i], depths[i], image_segms[i]
                depth = np.expand_dims(depth, axis=-1)
                image_segm = np.expand_dims(image_segm, axis=-1)
                image_depth = np.concatenate([image, depth, image_segm], axis=-1)
                np.save(os.path.join(save_path, "{:03}.npy".format(i)), image_depth)
        
                
                cnt += 1
                # import ipdb;ipdb.set_trace()
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="kitchen_knob1_on-v3")
    parser.add_argument("--n_exps", type=int, default=200)
    args = parser.parse_args()
    run(args)
        
