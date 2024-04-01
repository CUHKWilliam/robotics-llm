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
    name = "".join(get_task_text(env_name).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    policy = getattr(policies, policy_name)()
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

def run(args):
    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    resolution = (320, 240)
    # cameras = ['corner', 'corner2', 'corner3']
    cameras = ['corner']

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    env_name = args.env_name
    print(env_name)
    seg_ids = name2maskid[env_name]
    benchmark_env = env_dict[env_name]
    for camera in cameras:
        cnt = 0
        for seed in tqdm(range(n_exps)):
            # try:
            #     env = benchmark_env(seed=seed)
                
            #     obs = env.reset()
            #     # policy = MyPolicy_CL(env, env_name, camera, video_model, flow_model, max_replans=5)
            #     policy = get_policy(env_name)

            #     # images, depths, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
            #     next_obs, _, _, _, info = env.step()

            #     if len(images) <= 500:
            #         print("success")
            #         SAVE_PATH = '/data/wltang/robotic-llm/AVDC/datasets/metaworld/metaworld_dataset_2'
            #         save_path = SAVE_PATH
            #         os.makedirs(save_path, exist_ok=True)
            #         save_path = os.path.join(save_path, camera)
            #         os.makedirs(save_path, exist_ok=True)
            #         save_path = os.path.join(save_path, "{:03}".format(cnt))
            #         os.makedirs(save_path, exist_ok=True)
            #         for i in range(len(images)):
            #             image, depth = images[i], depths[i]
            #             depth = np.expand_dims(depth, axis=-1)
            #             image_depth = np.concatenate([image, depth], axis=-1)
            #             np.save(os.path.join(save_path, "{:03}.npy".format(i)), image_depth)
            #         cnt += 1
            # except:
            #     print('error')


            env = benchmark_env(seed=seed)
            obs = env.reset()
            policy = get_policy(env_name)
            cnt2 = 0
            done = False
            images = []
            depths = []
            segms = []
            while cnt2 < 500 and not done:
                action = policy.get_action(obs)
                next_obs, _, _, info = env.step(action)
                obs = next_obs
                if int(info['success']) == 1:
                    done = True
                data = env.render(camera_name=camera, depth=True, body_invisible=True, segmentation=True)                
                img, depth = env.render(camera_name=camera, depth=True, body_invisible=True,)
                img = np.stack(img, axis=0)
                seg_img = np.zeros((img.shape[0], img.shape[1]))
                seg_img[data[:, :, -1] == 31] = 1
                seg_img[data[:, :, -1] == 33] = 2
                segms.append(seg_img)
                # colors = np.random.randint(low=0, high=255, size=(100, 3))
                # seg_img = colors[data[:, :, -1]]
                images.append(img)
                depths.append(depth)

            if len(images) <= 500:
                print("success")
                SAVE_PATH = '/data/wltang/robotic-llm/AVDC/datasets/metaworld/metaworld_dataset_2_test'
                save_path = SAVE_PATH
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, env_name.split("-v2")[0])
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, camera)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "{:03}".format(cnt))
                os.makedirs(save_path, exist_ok=True)
                for i in range(len(images)):
                    image, depth, segm = images[i], depths[i], segms[i]
                    depth = np.expand_dims(depth, axis=-1)
                    segm = np.expand_dims(segm, axis=-1)
                    image_depth_segm = np.concatenate([image, depth, segm], axis=-1)
                    np.save(os.path.join(save_path, "{:03}.npy".format(i)), image_depth_segm)
                cnt += 1
                # import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=200)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()


    run(args)
        