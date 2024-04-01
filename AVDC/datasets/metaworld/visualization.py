import os
import numpy as np
import cv2
import imageio
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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



src = 'metaworld_dataset_2_test'
target = 'metaworld_dataset_2_test_vis'
cmat = np.array([[ 9.63268099e+01, -3.13378818e+02,  4.34104016e+01,
        -4.54382771e+01],
    [-2.55555772e+01, -2.55555772e+01,  3.11293150e+02,
        -2.25109256e+02],
    [-6.80413817e-01, -6.80413817e-01,  2.72165527e-01,
        -1.18392004e+00]])
        
tasks = os.listdir(src)
for task in tasks:
    src_path = os.path.join(src, task)
    tgt_path = os.path.join(target, task)
    os.makedirs(tgt_path, exist_ok=True)
    cams = os.listdir(src_path)
    for cam in cams:
        src_path2 = os.path.join(src_path, cam)
        tgt_path2 = os.path.join(tgt_path, cam)
        os.makedirs(tgt_path2, exist_ok=True)
        indices = os.listdir(src_path2)
        for idx in indices:
            src_path3 = os.path.join(src_path2, idx)
            tgt_path3 = os.path.join(tgt_path2, idx)
            os.makedirs(tgt_path3, exist_ok=True)
            series = os.listdir(src_path3)
            tgt_path4 = os.path.join(tgt_path3, "video.mp4")
            rgbs = []
            rgbs2 = []
            for i in range(len(series)):
                src_path4 = os.path.join(src_path3, "{:03}.npy".format(i))
                data = np.load(src_path4)
                rgb = data[:, :, :3]
                rgbs.append(rgb.astype(np.uint8))
                # depth = data[:, :, 3]
                # pts_2d = np.stack(np.meshgrid(np.arange(rgb.shape[1]-1), np.arange(rgb.shape[0]-1)), axis=-1)
                # pts_2d = pts_2d.reshape(-1, 2)
                # pts = to_3d(pts_2d, depth, cmat)
                # dist = np.linalg.norm(pts, axis=-1)
                # selected = dist < 30
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.set_aspect('equal', adjustable='box')
                # ax.axes.set_xlim3d(-2, 4)
                # ax.axes.set_xlim3d(0, 4)
                # ax.axes.set_xlim3d(-4, -2)
                # colors = rgb[:-1, :-1, :].reshape(-1, 3) / 256.
                # pts = pts[selected]
                # colors = colors[selected]
                # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1)
                # ax.view_init(elev=10, azim=-10., )
                # plt.show()

            imageio.mimsave(tgt_path4, rgbs)
