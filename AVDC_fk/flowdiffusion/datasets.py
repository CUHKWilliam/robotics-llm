from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
# from vidaug import augmentors as va
from lisa.model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)

random.seed(0)

### Sequential Datasets: given first frame, predict all the future frames

class SequentialDatasetNp(Dataset):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(os.path.join(path, "**/out.npy"), recursive=True)
        if debug:
            sequence_dirs = sequence_dirs[:10]
        self.sequences = []
        self.tasks = []
    
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)

        self.sequences = obss
        self.tasks = tasks
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("training_samples: ", len(self.sequences))
        print("Done")

    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        task = seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
        return outputs, [task] * len(outputs)

    def get_samples(self, seq):
        N = len(seq)
        ### uniformly sample {self.sample_per_seq} frames, including the first and last frame
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        images = [self.transform(Image.fromarray(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task
        
class SequentialDataset(SequentialDatasetNp):
    def __init__(self, path="../datasets/frederik/berkeley", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = get_paths(path)
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(seq_dir))
            if len(seq) > 1:
                self.sequences.append(seq)
            task = seq_dir.split('/')[-6].replace('_', ' ')
            self.tasks.append(task)
        self.sample_per_seq = sample_per_seq
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        images = [self.transform(Image.open(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task

class SequentialDatasetVal(SequentialDataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = sorted([d for d in os.listdir(path) if "json" not in d], key=lambda x: int(x))
        self.sample_per_seq = sample_per_seq
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(os.path.join(path, seq_dir)))
            if len(seq) > 1:
                self.sequences.append(seq)
            
        with open(os.path.join(path, "valid_tasks.json"), "r") as f:
            self.tasks = json.load(f)
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Markovian datasets: given current frame, predict the next frame
class MarkovianDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = torch.FloatTensor(samples[start_ind].transpose(2, 0, 1) / 255.0)
        x = torch.FloatTensor(samples[start_ind+1].transpose(2, 0, 1) / 255.0)
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(samples[0].transpose(2, 0, 1) / 255.0)
    
class MarkovianDatasetVal(SequentialDatasetVal):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = self.transform(Image.open(samples[start_ind]))
        x = self.transform(Image.open(samples[start_ind+1]))
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(Image.open(samples[0]))
        
class AutoregDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        pred_idx = np.random.randint(1, len(samples))
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.cat(images[:-1], dim=0)
        x_cond[:, 3*pred_idx:] = 0.0
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
        
class AutoregDatasetNpL(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        N = len(samples)
        h, w, c = samples[0].shape
        pred_idx = np.random.randint(1, N)
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.zeros((N-1)*c, h, w)
        x_cond[(N-pred_idx-1)*3:] = torch.cat(images[:pred_idx])
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
    
# SSR datasets
class SSRDatasetNp(SequentialDatasetNp):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128), in_size=(48, 64), cond_noise=0.2):
        super().__init__(path, sample_per_seq, debug, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.fromarray(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.fromarray(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class SSRDatasetVal(SequentialDatasetVal):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), in_size=(48, 64)):
        print("Preparing dataset...")
        super().__init__(path, sample_per_seq, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.open(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.open(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class MySeqDatasetMW(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0513", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/metaworld_dataset_faucet/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("-", " "))
        
        
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Randomly sample, from any intermediate to the last frame
# included_tasks = ["door-open", "door-close", "basketball", "shelf-place", "button-press", "button-press-topdown", "faucet-close", "faucet-open", "handle-press", "hammer", "assembly"]
# included_idx = [i for i in range(5)]
class SequentialDatasetv2(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset_2/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
    
        if randomcrop:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((160, 160)),
                video_transforms.RandomCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        else:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        print("Done")
    

    def get_samples(self, idx):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:
            start_idx = random.randint(0, len(seq)-1)
            seq = seq[start_idx:]
            N = len(seq)
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(N-1)
        else:
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            samples = self.get_samples(idx)
            images = self.transform([Image.open(s) for s in samples]) # [c f h w]
            x_cond = images[:, 0] # first frame
            x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
            task = self.tasks[idx]
            return x, x_cond, task
        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1 % self.__len__()) 

import pickle
class SequentialDatasetv2_rgbd_tfds2(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip
        
        self.file_paths = []
        for root, dirs, files in os.walk(path):
            if 'processed' in root:
                for file in files:
                    if file.endswith('pkl'):
                        self.file_paths.append(os.path.join(root, file))

        if randomcrop:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((160, 160)),
                video_transforms.RandomCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        else:
            self.transform = video_transforms.Compose([
                video_transforms.CenterCrop((128, 128)),
                video_transforms.Resize(target_size),
                volume_transforms.ClipToTensor()
            ])
        print("Done")
    

    def get_samples(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        instruction = str(data['steps'][0]['language_instruction'].numpy(), encoding='utf-8')
        keys = list(data['steps'][0]['observation'].keys())
        image_keys = []
        image_indices = []
        for key in keys:
            if 'image' in key:
                image_keys.append(key)
                image_indices.append(int(key.split("-")[1]))
        choice_idx = np.random.randint(0, len(image_indices) - 1)
        image_key = image_keys[choice_idx]
        image_idx = image_indices[choice_idx]
        rgbs = []
        depths = []
        for i in range(len(data['steps'])):
            rgbs.append(data['steps'][i]['observation'][image_key])
            depths.append(data['steps'][i]['observation'][image_key])
        rgbds = np.concatenate([rgbd, depths[:, None]], axis=-1)
        seq = rgbds
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:
            start_idx = random.randint(0,max(len(seq)-10), 0)
            seq = seq[start_idx:]
            N = len(seq)
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(N-1)
        else:
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples], instruction
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        samples, task = self.get_samples(idx)
        images = self.transform([Image.open(s) for s in samples]) # [c f h w]
        x_cond = images[:, 0] # first frame
        x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[idx]
        return x, x_cond, task
        

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
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    depths = np.expand_dims(depth, axis=-1)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

class SequentialDatasetv2_rgbd(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        # sequence_dirs = glob(f"{path}/**/metaworld_dataset_all_key/*/*/*/", recursive=True)
        sequence_dirs = glob(f"{path}/**/franka-kitchen_dataset/*/*/*/", recursive=True)
        
        self.tasks = []
        self.full_task_names = []
        self.sequences = []
        # self.key_indices = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.npy"), key=lambda x: int(x.split("/")[-1].rstrip(".npy")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
            self.full_task_names.append(seq_dir.split('/')[-4])
            # import pickle
            # with open(os.path.join(seq_dir, 'key_indices.pkl'), 'rb') as f:
                # key_indice = pickle.load(f)
            # self.key_indices.append(key_indice)

        self.transform = video_transforms.Compose([
            video_transforms.CenterCrop((128, 128)),
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])
        self.transform_depth = video_transforms.Compose([
            video_transforms.CenterCrop((128, 128)),
            video_transforms.Resize(target_size),
        ])
        print("Done")
        import json
        with open('name2maskid.json', 'r') as f:
            self.name2maskid = json.load(f)

    ## TODO: next
    def get_samples(self, idx, task_name):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        # if self.frame_skip is None:
        if True:
            start_idx = random.randint(0, max(0, len(seq) - 2))
            seq = seq[start_idx:]
            N = len(seq)
            samples = [0]
            for i in range(self.sample_per_seq-2):
                samples.append(1 + int(i*(N-2)/(self.sample_per_seq-2)))
            samples.append(N-1)
        else:
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx, self.full_task_names[idx])

        images = []
        depths = []
        segms = []
        for i in range(len(samples)):
            sample = samples[i]
            data = np.load(sample)
            image = data[:, :, :3]
            depth = data[:, :, 3:4]
            # segm = data[:, :, 4:5]
            segm = np.zeros((data.shape[0], data.shape[1], 2))
            images.append(image)
            depths.append(depth)
            segms.append(segm)
        images = self.transform(images) # [c f h w]
        depths = self.transform_depth(depths)
        depths = torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(0)
        segms = self.transform_depth(segms)
        segms = torch.from_numpy(np.stack(segms, axis=0)).unsqueeze(0)
        segms1 = (segms == 1).type(depths.dtype)
        segms2 = (segms == 2).type(depths.dtype)
        ## TODO:
        low, high = -8., -1.5
        depths[depths < low] = low
        depths[depths > high] = high
        depths -= low
        depths /= (high - low)

        images_depth = torch.cat([images, depths, segms[0].permute(3, 0, 1, 2)], dim=0)
        x_cond = images_depth[:, 0] # first frame
        x = rearrange(images_depth[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[idx]
        return x.float(), x_cond.float(), task



class SequentialDatasetv2_rgbd_with_segm(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        # sequence_dirs = glob(f"{path}/**/metaworld_dataset_all_key/*/*/*/", recursive=True)
        sequence_dirs = glob(f"{path}/**/franka-kitchen_dataset-micro-open/*/*/*/", recursive=True)
        
        self.tasks = []
        self.full_task_names = []
        self.sequences = []
        # self.key_indices = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.npy"), key=lambda x: int(x.split("/")[-1].rstrip(".npy")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
            self.full_task_names.append(seq_dir.split('/')[-4])
            # import pickle
            # with open(os.path.join(seq_dir, 'key_indices.pkl'), 'rb') as f:
                # key_indice = pickle.load(f)
            # self.key_indices.append(key_indice)

        self.transform = video_transforms.Compose([
            # video_transforms.CenterCrop((120, 120)),
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])
        self.transform_depth = video_transforms.Compose([
            # video_transforms.CenterCrop((120, 120)),
            video_transforms.Resize(target_size),
        ])
        print("Done")
        import json
        with open('name2maskid.json', 'r') as f:
            self.name2maskid = json.load(f)

    ## TODO: next
    def get_samples(self, idx, task_name):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        # if self.frame_skip is None:
        if True:
            start_idx = random.randint(0, max(0, len(seq) - 2))
            seq = seq[start_idx:]
            N = len(seq)
            samples = [0]
            for i in range(self.sample_per_seq-2):
                samples.append(1 + int(i*(N-2)/(self.sample_per_seq-2)))
            samples.append(N-1)
        else:
            start_idx = random.randint(0, len(seq)-1)
            samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [seq[i] for i in samples]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx, self.full_task_names[idx])

        images = []
        depths = []
        segms = []
        full_task_name = self.full_task_names[idx]
        segm_idx = self.name2maskid[full_task_name][0] + 20
        for i in range(len(samples)):
            sample = samples[i]
            data = np.load(sample)
            image = data[:, :, :3]
            depth = data[:, :, 3:4]
            segm = np.zeros((data.shape[0], data.shape[1], 2))
            segm[:, :, :1] = (data[:, :, 4:5] == segm_idx).astype(np.float32)
            images.append(image)
            depths.append(depth)
            segms.append(segm)
        images = self.transform(images) # [c f h w]
        depths = self.transform_depth(depths)
        depths = torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(0)
        segms = self.transform_depth(segms)
        segms = torch.from_numpy(np.stack(segms, axis=0)).unsqueeze(0)
        # import ipdb;ipdb.set_trace()
        # segms = torch.from_numpy(np.stack(segms, axis=0)).unsqueeze(0)
        # import cv2
        # print(full_task_name)
        # cv2.imwrite('debug.png', segms[0, :, :, :, 0].permute(1, 2, 0).detach().cpu().numpy()[..., 0] * 255)
        # cv2.imwrite("debug2.png", images[:, 0, :, :].permute(1, 2, 0).detach().cpu().numpy() * 255)
        ## TODO:
        low, high = -8., -1.5
        depths[depths < low] = low
        depths[depths > high] = high
        depths -= low
        depths /= (high - low)

        images_depth = torch.cat([images, depths, segms[0].permute(3, 0, 1, 2)], dim=0)
        x_cond = images_depth[:, 0] # first frame
        x = rearrange(images_depth[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[idx]
        return x.float(), x_cond.float(), task



class SequentialFlowDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset_2/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.flows = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            flows = sorted(glob(f"{seq_dir}flow/*.npy"))
            self.sequences.append(seq)
            self.flows.append(np.array([np.load(flow) for flow in flows]))
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        return seq[0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # try:
            s = self.get_samples(idx)
            x_cond = self.transform(Image.open(s)) # [c f h w]
            x = rearrange(torch.from_numpy(self.flows[idx]), "f w h c -> (f c) w h") / 128
            task = self.tasks[idx]
            return x, x_cond, task
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(idx + 1 % self.__len__()) 

class SequentialNavDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/thor_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-3]
            seq = sorted(glob(f"{seq_dir}frames/*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(task)

        self.transform = video_transforms.Compose([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])

        num_seqs = len(self.sequences)
        num_frames = sum([len(seq) for seq in self.sequences])
        self.num_frames = num_frames
        self.frameid2seqid = [i for i, seq in enumerate(self.sequences) for _ in range(len(seq))]
        self.frameid2seq_subid = [f - self.frameid2seqid.index(self.frameid2seqid[f]) for f in range(num_frames)]

        print(f"Found {num_seqs} seqs, {num_frames} frames in total")
        print("Done")

    def get_samples(self, idx):
        seqid = self.frameid2seqid[idx]
        seq = self.sequences[seqid]
        start_idx = self.frameid2seq_subid[idx]
        
        samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.sample_per_seq)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        images = self.transform([Image.open(s) for s in samples]) # [c f h w]
        x_cond = images[:, 0] # first frame
        x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[self.frameid2seqid[idx]]
        return x, x_cond, task

class MySeqDatasetReal(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0606/processed_data", sample_per_seq=7, target_size=(48, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/*/*/", recursive=True)
        print(f"found {len(sequence_dirs)} sequences")
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*.png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("_", " "))
        
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")


if __name__ == "__main__":
    dataset = SequentialNavDataset("../datasets/thor")
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

import sys
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from lisa.model.segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from lisa.model.llava import conversation as conversation_lib

class SequentialDatasetv2_contact_detection(Dataset):
    def __init__(self, 
                 path="../datasets/valid", 
                 sample_per_seq=7, 
                 target_size=128, 
                 frameskip=None, 
                 randomcrop=False,
                 precision='fp32',
                 vision_tower=None,
                 inference=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        # sequence_dirs = glob(f"{path}/**/metaworld_dataset_all_key/*/*/*/", recursive=True)
        sequence_dirs = glob(f"{path}/**/franka-kitchen_dataset-micro-open/*/*/*/", recursive=True)
        
        self.tasks = []
        self.full_task_names = []
        self.sequences = []
        # self.key_indices = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.npy"), key=lambda x: int(x.split("/")[-1].rstrip(".npy")))
            self.sequences.append(seq)
            if "v2" in seq_dir:
                self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))
            elif "v3" in seq_dir:
                self.tasks.append(
                    seq_dir.split("/")[-4].split('-v3')[0].replace("_", " ").replace("sdoor", "sliding door").replace("ldoor", 'opening door').replace('micro', 'microwave')
                )
            self.full_task_names.append(seq_dir.split('/')[-4])
            # import pickle
            # with open(os.path.join(seq_dir, 'key_indices.pkl'), 'rb') as f:
                # key_indice = pickle.load(f)
            # self.key_indices.append(key_indice)
        import json
        with open('name2maskid.json', 'r') as f:
            self.name2maskid = json.load(f)
        with open('task2description.json', 'r') as f:
            self.task2description = json.load(f)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.transform = ResizeLongestSide(target_size)
        self.precision = precision
        self.inference = inference

    ## TODO: next
    def get_samples(self, idx, task_name):
        seq = self.sequences[idx]
        return seq

    def __len__(self):
        return len(self.sequences)
    
    def preprocess(
        self,
        x,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
    ) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess_mask(
        self,
        x,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
    ) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def preprocess_mask(
        self,
        x,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
    ) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx, self.full_task_names[idx])
        images = []
        segms = []
        task = self.tasks[idx]
        full_task_name = self.full_task_names[idx]
        image_clips = []
        image_paths = []
        for i in range(len(samples)):
            sample = samples[i]
            image_paths.append(sample)
            data = np.load(sample)
            image = data[:, :, :3].astype(np.uint8)
            image_clip = (self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
            if self.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif self.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()
            image = self.transform.apply_image(image)
            resize = image.shape[:2]
            image = (self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
            if self.precision == "bf16":
                image = image.bfloat16()
            elif self.precision == "fp16":
                image = image.half()
            else:
                image = image.float()
            segm = data[:, :, 4:5]
            mask_id = self.name2maskid[full_task_name]
            mask_id = [a_mask_id + 20 for a_mask_id in mask_id]
            segm_mask = np.zeros_like(segm).astype(np.bool_)
            for a_mask_id in mask_id:
                segm_mask = np.logical_or(segm_mask, (segm == a_mask_id).astype(np.bool_)).astype(np.float32)
            segm_mask = self.transform.apply_image(segm_mask)
            images.append(image)
            image_clips.append(image_clip)
            segms.append(segm_mask)
        idx = np.random.randint(low=0, high=len(images) - 1, size=(1,))[0]
        image, segm, image_clip = images[idx], segms[idx], image_clips[idx]

        ## TODO: for vis
        # tmp = image[0].permute(1, 2, 0)
        # tmp -= tmp.min()
        # tmp /= tmp.max()
        # import cv2
        # cv2.imwrite('debug.png', (tmp.float()  * 255).detach().cpu().numpy().astype(np.uint8))
        # cv2.imwrite('debug2.png', (segm_mask[0] * 255).detach().cpu().numpy().astype(np.uint8))
        # import ipdb;ipdb.set_trace()

        image_path = image_paths[idx]

        masks = torch.from_numpy(np.expand_dims(segm, axis=0)).float()
        label = segm

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        description = self.task2description[full_task_name]
        text = "Where should I grasp if I need to {} ? Please output segmentation mask.".format(description)
        questions = [DEFAULT_IMAGE_TOKEN + "\n" + text]
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], "It is [SEG].")
        conversations.append(conv.get_prompt())
        sampled_classes = "grasp area"
        return (
            image_path,
            image[0],
            image_clip[0],
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            self.inference
        )
    
