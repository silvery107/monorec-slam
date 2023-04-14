import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_loader.tum_rgbd_dataset import TUMRGBDDataset
from data_loader.kitti_odometry_dataset import KittiOdometryDataset
from model.monorec.monorec_model import MonoRecModel
from utils import unsqueezer, map_fn, to
from data_loader.data_loaders import BaseDataLoader
import shutil
import cv2
import os
import numpy as np
import argparse

target_image_size = (256, 512)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kitti", required=True, choices=["kitti", "tum"])
    parser.add_argument("--seq", type=int, default=7, required=False)
    parser.add_argument("--seq_name", type=str, default="rgbd_dataset_freiburg3_walking_halfsphere", required=False)
    args = parser.parse_args()

    if args.dataset == "kitti":
        dataset = KittiOdometryDataset("data/kitti", sequences=[f"{args.seq:02d}"], target_image_size=target_image_size, frame_count=2,
                                    depth_folder="image_depth_annotated", lidar_depth=False, use_dso_poses=True,
                                    use_index_mask=None)
    elif args.dataset == "tum":
        dataset = TUMRGBDDataset(dataset_dir=f"data/tum/{args.seq_name}", frame_count=2, target_image_size=target_image_size)
    else:
        raise NotImplementedError(f"Dataset type {args.dataset} is not supported for now")

    data_loader = BaseDataLoader(dataset, batch_size=1, shuffle=False, validation_split=0, num_workers=8)

    checkpoint_location = Path("modules/MonoRec/saved/checkpoints/monorec_depth_ref.pth")
    inv_depth_min_max = [0.33, 0.0025]

    print("Initializing model...")
    monorec_model = MonoRecModel(checkpoint_location=checkpoint_location, 
                                inv_depth_min_max=inv_depth_min_max,
                                pretrain_mode=2)

    monorec_model.to(device)
    monorec_model.eval()

    if args.dataset == "kitti":
        output_path = f"data/{args.dataset}/sequences/{args.seq:02d}_mask"
    elif args.dataset == "tum":
        output_path = f"data/{args.dataset}/{args.seq_name}/mask"

    print(f"Resulting masks will be saved to {output_path}")
    confirm = input("Are you sure? (y/n)")
    if (confirm.lower().find("y") != -1 or confirm.lower().find("yes") != -1):
        print("Confirmed")
    else:
        sys.exit("User rejected")
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    dilatation_size = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilatation_size+1, 2*dilatation_size+1), (dilatation_size, dilatation_size))

    inference_times = []
    counter = 0
    print(f"Start processing data...")
    for batch_idx, (data, target) in enumerate(data_loader):
        counter += 1
        data = to(data, device)
        s = time.time()
        with torch.no_grad():
            output_dict = monorec_model(data)

        mask = output_dict["cv_mask"][0, 0].detach().cpu().numpy() # (H, W)
        mask_to_save = mask
        # mask_to_save = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=6)
        mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=6)
        mask_to_save = cv2.erode(mask_to_save, kernel, iterations=4)
        mask_to_save = (255*(mask_to_save - np.min(mask_to_save))/np.ptp(mask_to_save)).astype(np.uint8)
        _, mask_to_save = cv2.threshold(mask_to_save, 50, 255, cv2.THRESH_BINARY)

        plt.imsave(f"{output_path}/{counter:06d}.png", mask_to_save, cmap='gray')
        # print(f"mask: {mask.shape}, {mask.dtype}")
        e = time.time()
        # print(f"Inference took {e - s}s")
        inference_times.append(e-s)

    inference_times = np.asarray(inference_times)
    print(f"Processed {counter} images")
    print(f"Average inference time: {inference_times.mean():.4f} s")