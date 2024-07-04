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

import json
with open('name2maskid.json', 'r') as f:
    name2maskid = json.load(f)


def get_key_indices(segm_images, task):
    indices = []
    mask_ids = name2maskid[task]
    mask_ids = [mask_id + 20 for mask_id in mask_ids]
    last_poses = None
    for i in range(len(segm_images)):
        segm_image = segm_images[i]
        poses = []
        flag_continue = False
        for mask_id in mask_ids:
            if (segm_image == mask_id).sum() == 0:
                flag_continue = True
                break
            pos = np.stack(np.where(segm_image == mask_id), axis=0).mean(1)
            poses.append(pos)
        if flag_continue:
            continue
        if last_poses is None:
            last_poses = poses
            continue
        flag_unchanged = True
        for j in range(len(poses)):
            if np.linalg.norm(poses[j] - last_poses[j]) > 0.02:
                flag_unchanged = False
        if flag_unchanged:
            continue
        last_poses = poses
        indices.append(i)
    if len(indices) == 0:
        import ipdb;ipdb.set_trace()
    return indices


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
    cameras = ['corner3']

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    env_name = args.env_name
    print(env_name)
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
            all_segm_indices = []
            while cnt2 < 500 and not done:
                action = policy.get_action(obs)
                next_obs, _, _, info = env.step(action)
                obs = next_obs
                if int(info['success']) == 1:
                    done = True
                data = env.render(camera_name=camera, depth=True, body_invisible=True, segmentation=True, resolution=resolution)               
                ## TODO: instance paining
                # '''
                seg = data[:, :, 1]
                seg_ids = np.unique(seg)
                seg_img = np.zeros((seg.shape[0], seg.shape[1], 3))
                for seg_id in seg_ids:
                    if seg_id == -1:
                        continue
                    seg_img[seg == seg_id] = np.array([1, 0, 0]) * seg_id
                seg_img = seg_img.astype(np.uint8)
                cv2.imwrite('debug.png', seg_img)
                import ipdb;ipdb.set_trace()
                # '''
                ## TODO: end
                img, depth = env.render(camera_name=camera, depth=True, body_invisible=True, resolution=resolution)
                img = np.stack(img, axis=0)
                seg_img = np.zeros((img.shape[0], img.shape[1]))
                data[:, :, -1]
                seg_img[data[:, :, -1] == 51] = 1
                seg_img[data[:, :, -1] == 53] = 2
                segms.append(seg_img)
                # colors = np.random.randint(low=0, high=255, size=(100, 3))
                # seg_img = colors[data[:, :, -1]]
                images.append(img)
                depths.append(depth)
                all_segm_indices.append(data[:, :, -1])
                
            if len(images) <= 2000:
                imageio.mimsave('debug.mp4', images)
                print("success")
                SAVE_PATH = '/data/wltang/robotic-llm/AVDC/datasets/metaworld/metaworld_dataset_drawer_key'
                # SAVE_PATH = './tmp'
                save_path = SAVE_PATH
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, env_name.split("-v2")[0])
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, camera)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "{:03}".format(cnt))
                os.makedirs(save_path, exist_ok=True)
                indices = get_key_indices(all_segm_indices, env_name)
                for i in range(len(images)):
                    image, depth, segm = images[i], depths[i], segms[i]
                    depth = np.expand_dims(depth, axis=-1)
                    segm = np.expand_dims(segm, axis=-1)
                    image_depth_segm = np.concatenate([image, depth, segm], axis=-1)
                    np.save(os.path.join(save_path, "{:03}.npy".format(i)), image_depth_segm)
                import pickle
                with open(os.path.join(save_path, 'key_indices.pkl'), 'wb') as f:
                    pickle.dump(indices, f)
                
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
        
