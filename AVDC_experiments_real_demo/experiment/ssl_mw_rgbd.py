from goal_diffusion import GoalGaussianDiffusion, Trainer, GoalGaussianDiffusionFast
from mypolicy import MyPolicy_CL_rgbd
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames, collect_video_rgbd
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
from flowdiffusion.inference_utils import get_video_model_rgbd, pred_video
import random
import torch
from argparse import ArgumentParser
from datasets import SequentialDatasetv2_rgbd_otf
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)

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

success_rates = []
MAX_DATA_LEN = 30

def run(args):
    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    resolution = (320, 240)
    # cameras = ['corner', 'corner2', 'corner3']
    cameras = ['corner']
    max_replans = -1
    data_idx = 0

    video_model = get_video_model_rgbd(ckpts_dir=args.ckpt_dir, milestone=args.milestone)
    flow_model = get_flow_model()

    # add_lora(video_model.model.cpu())
    # video_model.model = video_model.model.cuda()

    tokenizer = video_model.tokenizer
    text_encoder = video_model.text_encoder

    try:
        with open(f"{result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}
    otf_samples = []

    for exp_cnt in tqdm(range(n_exps)):
        # env_name = np.random.choice(["door-open-v2-goal-observable", "door-close-v2-goal-observable", "basketball-v2-goal-observable", "shelf-place-v2-goal-observable", "button-press-v2-goal-observable", "button-press-topdown-v2-goal-observable", "faucet-close-v2-goal-observable", "faucet-open-v2-goal-observable", "handle-press-v2-goal-observable", "hammer-v2-goal-observable", "assembly-v2-goal-observable"])
        env_name = 'handle-press-v2-goal-observable'
        print(env_name)
        seg_ids = name2maskid[env_name]
        benchmark_env = env_dict[env_name]

        succes_rates = []
        reward_means = []
        reward_stds = []
        replans_counters = []

        for _ in cameras:
            camera = np.random.choice(cameras)
            rewards = []
            seed = np.random.randint(0, 10000000)
            # seed = 2
            retry = 0
            ## TODO: online 
            while True:
                env = benchmark_env(seed=seed)
                
                obs = env.reset()
                policy = MyPolicy_CL_rgbd(env, env_name, camera, video_model, flow_model, max_replans=max_replans, resolution=resolution)
                images, depths, segms, episode_return = collect_video_rgbd(obs, env, policy, camera_name=camera, resolution=resolution, body_invisible=True, show_traj=False)
                images = np.stack(images, axis=0)
                # images2 = [cv2.resize(x, (80, 60)) for x in images[::3, ...]]
                rewards.append(episode_return / len(images))
                # imageio.mimsave('debug.mp4', images)
                # import ipdb;ipdb.set_trace()
                ### save sample video
                # import ipdb;ipdb.set_trace()

                image_depth_segms = []
                # if len(images) <= 300:
                if len(images) <= 80:
                    success = 1
                else:
                    success = 0
                succes_rates = succes_rates[:100]
                succes_rates = [success] + succes_rates
                print("succes_rates:", sum(succes_rates) / len(succes_rates))
                if success == 0:
                    retry += 1
                    imageio.mimsave('debug.mp4', images)
                    print("fail, retry{}".format(retry))
                    if retry < 10:
                        continue
                    else:
                        break
                print('success: {}'.format(success))
                imageio.mimsave('debug.mp4', images)
                
                # image_depth_segms = policy.pred_image_depth_segm
                # import ipdb;ipdb.set_trace()
                for i in range(len(images)):
                    image = images[i]
                    depth = depths[i]
                    segm = segms[i]
                    image_depth_segm = np.concatenate([image, np.expand_dims(depth, axis=-1), np.expand_dims(segm, axis=-1)], axis=-1)
                    env_name = env_name.split('-v2-')[0]
                    # image_depth_segm = image_depth_segms[i]
                    # os.makedirs(f'otf_datasets/metaworlds/{env_name}', exist_ok=True)
                    # os.makedirs(f'otf_datasets/metaworlds/{env_name}/{camera}', exist_ok=True)
                    # os.makedirs(f'otf_datasets/metaworlds/{env_name}/{camera}/{data_idx}', exist_ok=True)
                    # os.makedirs(f'otf_datasets/metaworlds/{env_name}/{camera}/{data_idx}/{success}', exist_ok=True)
                    # np.save(os.path.join("otf_datasets/metaworlds/{}/{}/{}/{}/{:03}.npy".format(env_name, camera, data_idx, success, i)), image_depth_segm)
                print('save seed {} complete, data_idx {}'.format(seed, data_idx))
                # print("test eplen: ", len(images))

                ## TODO: train otf data
                sample_per_seq = 8
                target_size = (128, 128)

                train_set = SequentialDatasetv2_rgbd_otf(
                    sample_per_seq=sample_per_seq, 
                    path="./otf_datasets/metaworlds/", 
                    target_size=target_size,
                    randomcrop=True,
                    # start_idx=0
                )
                valid_set = train_set
                results_folder = './otf_ft_results'
                trainer = Trainer(
                    diffusion_model=video_model.model,
                    tokenizer=tokenizer, 
                    text_encoder=text_encoder,
                    train_set=train_set,
                    valid_set=valid_set,
                    train_lr=1e-5,
                    train_num_steps = min(40, len(train_set) * 3),
                    save_and_sample_every = min(40, len(train_set) * 3),
                    ema_update_every = min(10, len(train_set) * 3),
                    ema_decay = 0.999,
                    train_batch_size =1,
                    valid_batch_size =1,
                    gradient_accumulate_every = 1,
                    num_samples=1, 
                    results_folder =results_folder,
                    fp16 =True,
                    amp=True,
                )

                trainer.model.load_state_dict(video_model.model.state_dict())
                # if os.path.exists(os.path.join(results_folder, 'model-otf.pt')):
                #     trainer.load2('otf')
                #     print('loading otf ckpt')
                # else:
                #     print("no otf ckpt yet")
                trainer.train_rgbd(save_name='otf')
                video_model.model = trainer.model
                del trainer, train_set
                torch.cuda.empty_cache()

                if success == 1:
                    break
            data_idx += 1
            data_idx %= MAX_DATA_LEN
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()

   
    run(args)
        
