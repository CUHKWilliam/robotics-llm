from mujoco_py.generated import const
import numpy as np
import cv2
from .inverse_kinematics import calculate_ik


def get_robot_seg(env):
    seg = env.render(segmentation=True)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])

    for i in geoms_ids:
        if 2 <= i <= 33:
            img[ids == i] = True
    return img

def get_seg(env, camera, resolution, seg_ids):
    seg = env.render(segmentation=True, resolution=resolution, camera_name=camera)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])
    # print(geoms_ids)

    for i in geoms_ids:
        if i in seg_ids:
            img[ids == i] = True
    img = img.astype('uint8') * 255
    return cv2.medianBlur(img, 3)

def get_cmat(env, cam_name, resolution):
    id = env.sim.model.camera_name2id(cam_name)
    fov = env.sim.model.cam_fovy[id]
    pos = env.sim.data.cam_xpos[id]
    rot = env.sim.data.cam_xmat[id].reshape(3, 3).T
    width, height = resolution
    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos
    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0 # focal length
    focal = np.diag([-focal_scaling, -focal_scaling, 1.0, 0])[0:3, :]
    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (width - 1) / 2.0
    image[1, 2] = (height - 1) / 2.0

    return image @ focal @ rotation @ translation



def collect_video(init_obs, env, policy, camera_name='corner3', resolution=(640, 480)):
    images = []
    depths = []
    episode_return = 0
    done = False
    obs = init_obs
    if camera_name is None:
        cameras = ["corner3", "corner", "corner2"]
        camera_name = np.random.choice(cameras)

    image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution, body_invisible=True)
    images += [image]
    depths += [depth]
    
    dd = 10 ### collect a few more steps after done
    while dd:
        action = policy.get_action(obs)
        try:
            obs, reward, done, info = env.step(action)
            done = info['success']
            dd -= done
            episode_return += reward
        except Exception as e:
            print(e)
            break
        if dd != 10 and not done:
            break
        image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution, body_invisible=True)
        images += [image]
        depths += [depth]
                
    return images, depths, episode_return

def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])

import mujoco_py
from scipy.spatial.transform import Rotation as R
cnt = 0
def collect_video_rgbd(init_obs, env, policy, camera_name='corner3', resolution=(640, 480), body_invisible=False, show_traj=True):
    show_traj = False
    images = []
    depths = []
    segms = []
    episode_return = 0
    done = False
    obs = init_obs
    if camera_name is None:
        cameras = ["corner3", "corner", "corner2"]
        # cameras = ['corner']
        camera_name = np.random.choice(cameras)

    if env.__class__.__name__ == "MuJoCoPixelObs":
        image, depth = env.get_image_depth(body_invisible=body_invisible)
    else:
        image, depth = env.render(depth=True, camera_name=camera_name, body_invisible=body_invisible, resolution=resolution)
    
    images += [image]
    depths += [depth]
    seg_img = np.zeros((image.shape[0], image.shape[1]))
    segms += [seg_img]
    dd = 10 ### collect a few more steps after done
    cnt = 0
    while dd:
        # try:
        if True:
            action, previous_pos = policy.get_action(obs, return_hand=True, qpos_control=True)
            if env.__class__.__name__ == "MuJoCoPixelObs":
                reward = 0
                print("mocap_pos:", env.sim.data.mocap_pos  )
                action_pos = action['delta_pos']
                env.sim.data.set_mocap_pos('ee_mocap', env.sim.data.mocap_pos * 1 +  action_pos * 0.1)
                # env.sim.data.set_mocap_pos('ee_mocap', np.array([0., 0., 2.7]))
                rot_quat = action['rot'] 
                ctrl = env.sim.data.ctrl.copy()
                if rot_quat is not None:
                    print("rot_quat", rot_quat)
                    print("delta_rot_value", action['delta_rot_value'])
                    # rotvec = R.from_quat(rot_quat).as_rotvec()
                    # rotvec = np.array([0., -np.pi / 2., 0.])
                    # rot_quat = R.from_rotvec(rotvec).as_quat()

                    # env.sim.data.set_mocap_quat('ee_mocap', np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]))
                    # debug_quat = R.from_rotvec(R.from_quat(env.sim.data.get_mocap_quat('ee_mocap')).as_rotvec() + np.array([0., 0.0002 * cnt, 0])).as_quat()
                    # cnt += 1
                    # env.sim.data.set_mocap_quat('ee_mocap', debug_quat)

                    # if delta_rot_value > 0:
                    #     env.sim.data.qpos[:][6] = 1
                    # elif delta_rot_value < 0:
                    #     env.sim.data.qpos[:][6] = 1
                if action['grab_effort'] > 0:
                    ctrl[7] = -3
                    ctrl[8] = -3
                else:
                    ctrl[7] = 3
                    ctrl[8] = 3
                env.sim.data.ctrl[:] = ctrl.copy()
                for _ in range(10):
                    env.sim.step()
                # mujoco_py.functions.mj_step(env.sim.model, env.sim.data)
                obs = env.sim.data.qpos
                rwd_dict = env.get_reward_dict(env.get_obs_dict(env.sim))
                done = rwd_dict['solved']
            else:
                obs, reward, done, info = env.step(action)
                done = info['success']
            dd -= done
            episode_return += reward
        # except Exception as e:
        #     print(e)
        #     break

        if dd != 10 and not done:
            break

        if env.__class__.__name__ == "MuJoCoPixelObs":
            image, depth = env.get_image_depth(body_invisible=body_invisible)
        else:
            image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution, body_invisible=body_invisible)
        seg_img = np.zeros((image.shape[0], image.shape[1]))
        cv2.imwrite('debug.png', image)
        images += [image]
        depths += [depth]
        segms += [seg_img]
        cnt += 1
        if cnt > 500:
            break
    return images, depths, segms, episode_return


    
    
