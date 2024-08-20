import numpy as np
from read_scene_pcs import RealSense
import os
import cv2

save_root = "../../AVDC/datasets/realworld_data"
os.makedirs(save_root, exist_ok=True)

task = "put_black_stick_to_while_modal"
save_dir = os.path.join(save_root, task)
os.makedirs(save_dir, exist_ok=True)

camera = "camera1"
save_dir = os.path.join(save_dir, camera)
os.makedirs(save_dir, exist_ok=True)

example_inds = 10
save_dir = os.path.join(save_dir, "{}".format(example_inds))
os.makedirs(save_dir, exist_ok=True)

realsense = RealSense()
seq_idx = 0
while True:
    image, depth, flag_stop = realsense.get_data()
    if flag_stop:
        break
    image_depth = np.concatenate([image, depth[..., None]], axis=-1)
    save_path = os.path.join(save_dir, "{}.npy".format(seq_idx))
    print("save to {}".format(save_path))
    np.save(save_path, image_depth)
    seq_idx += 1


## upload to server
import paramiko
from scp import SCPClient

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

# server = "128.32.164.89"
# port = "22"
# user = "msc-auto"
# password = "MSCAuto-2021!"
# remote_data_root = "/media/msc-auto/HDD/wltang/robotics-llm/AVDC/datasets/"
# ssh = createSSHClient(server, port, user, password)
# scp = SCPClient(ssh.get_transport())
# scp.put("{}/*".format(save_root), remote_path=remote_data_root)

## TODO: push to server
# scp -r -P 22 ../../AVDC/datasets/realworld_data/ msc-auto@128.32.164.89:/media/msc-auto/HDD/wltang/robotics-llm/AVDC/datasets/