from goal_diffusion import GoalGaussianDiffusion, Trainer
from datasets import SequentialDatasetv2_rgbd
from torch.utils.data import Subset
import argparse
import sys
import torch
sys.path.append("/media/msc-auto/HDD/wltang/robotics-llm/AVDC_experiments/experiment/CamLiFlow")
from factory import model_factory

scene_flow_model_config = {'name': 'camliraft', 'batch_size': 8, 'freeze_bn': False, 'backbone': {'depth': 50, 'pretrained': 'pretrain/resnet50-11ad3fa6.pth'}, 'n_iters_train': 10, 'n_iters_eval': 20, 'fuse_fnet': True, 'fuse_cnet': True, 'fuse_corr': True, 'fuse_motion': True, 'fuse_hidden': False, 'loss2d': {'gamma': 0.8, 'order': 'l2-norm'}, 'loss3d': {'gamma': 0.8, 'order': 'l2-norm'}}
scene_flow_model_config = argparse.Namespace(**scene_flow_model_config)
scene_flow_model_config.backbone = argparse.Namespace(**{"depth": 50, "pretrained": './CamLiFlow/pretrain/resnet50-11ad3fa6.pth'})
scene_flow_model_config.loss2d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})
scene_flow_model_config.loss3d = argparse.Namespace(**{"gamma": 0.8, "order": 'l2-norm'})


scene_flow_model = model_factory(
    scene_flow_model_config,
)

scene_flow_model_ckpt = torch.load("./CamLiFlow/camliraft_things150e.pt")
self.scene_flow_model.load_state_dict(scene_flow_model_ckpt['state_dict'], strict=True)
self.scene_flow_model.to("cuda:0")
self.scene_flow_model.eval()

def main(args):
    valid_n = 1
    ## TODO: next
    sample_per_seq = 8
    target_size = (128, 128)

    train_set = SequentialDatasetv2_rgbd(
        sample_per_seq=sample_per_seq, 
        path="../datasets/metaworld", 
        target_size=target_size,
        randomcrop=True
    )
    dl = DataLoader(train_set, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)

    for (x, x_cond, goal) in tqdm(dl):
        import ipdb;ipd.set_trace()
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

main()
