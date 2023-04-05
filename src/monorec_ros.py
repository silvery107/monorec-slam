import argparse
import cv2
import numpy as np
from pathlib import Path

import rospy
from sensor_msgs.msg import Image

import torch
from data_loader.kitti_odometry_dataset import KittiOdometryDataset
from model.monorec.monorec_model import MonoRecModel
from data_loader.data_loaders import BaseDataLoader
from utils import to


class MonoRecNode:
    def __init__(self, dataset_type, seq, pub_rate=10) -> None:
        self.img_pub = rospy.Publisher("camera/image_raw", Image, queue_size=1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.target_image_size = (256, 512)
        self.rate = rospy.Rate(pub_rate)
        
        dataset = None
        if dataset_type == "kitti":
            dataset = KittiOdometryDataset("data/kitti", sequences=[f"{seq:02d}"], target_image_size=self.target_image_size, frame_count=2,
                                        depth_folder="image_depth_annotated", lidar_depth=False, use_dso_poses=True,
                                        use_index_mask=None)
        else:
            raise NotImplementedError(f"Dataset type {dataset_type} is not supported for now")

        self.data_loader = BaseDataLoader(dataset, batch_size=1, shuffle=False, validation_split=0, num_workers=8)

        checkpoint_location = Path("modules/MonoRec/saved/checkpoints/monorec_depth_ref.pth")
        inv_depth_min_max = [0.33, 0.0025]

        rospy.loginfo("Initializing model...")
        self.monorec_model = MonoRecModel(checkpoint_location=checkpoint_location, 
                                        inv_depth_min_max=inv_depth_min_max,
                                        pretrain_mode=2)
        self.monorec_model.to(self.device).eval()

        dilatation_size = 8
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilatation_size+1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        
        self.input_size = (1241, 376) if seq in [0, 20] else (1226, 370)
        self.interm_size = (753, 376) if seq in [0, 20] else (740, 370)
        self.pad_length = (self.input_size[0]-self.interm_size[0]) // 2

    def run(self):
        rospy.loginfo("Start pumping images!")
        for batch_idx, (data, _) in enumerate(self.data_loader):
            if rospy.is_shutdown():
                break
            # img = data["keyframe"][0].permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
            img = data["keyframe_orig"][0].detach().numpy()
            # print(img.shape, img.dtype)
            timestamp = data["timestamp"]
            # timestamp = 0.

            data = to(data, self.device)
            with torch.no_grad():
                output_dict = self.monorec_model(data)

            mask_orig = output_dict["cv_mask"][0, 0].detach().cpu().numpy() # (H, W)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=6)
            mask_orig = cv2.dilate(mask_orig, self.kernel, iterations=6)
            mask_orig = cv2.erode(mask_orig, self.kernel, iterations=4)
            mask_orig = (255*(mask_orig - np.min(mask_orig))/np.ptp(mask_orig)).astype(np.uint8)
            _, mask_orig = cv2.threshold(mask_orig, 50, 255, cv2.THRESH_BINARY)


            
            # Align output mask to input image size
            mask_align = mask_orig
            mask_align = cv2.resize(mask_align, self.interm_size)
            mask_align = cv2.copyMakeBorder(mask_align, 
                                            0, 0, self.pad_length, self.pad_length, 
                                            borderType=cv2.BORDER_CONSTANT, value=0)

            # Apply mask to input image
            img_to_pub = img
            if mask_align is not None:
                mask_inv = cv2.bitwise_not(mask_align)
                # print(mask_inv.shape, img_to_pub.shape)
                # print(img_to_pub.dtype, mask_inv.dtype)
                assert mask_inv.shape[0] == img_to_pub.shape[0]
                img_to_pub = cv2.bitwise_and(img_to_pub, img_to_pub, mask=mask_inv)
            else:
                rospy.logwarn(f"Skipped mask for image {batch_idx:06d}")

            self.publish_image(img_to_pub, timestamp)
            
            self.rate.sleep()
        
        rospy.loginfo("Sequence finished!")

    def publish_image(self, img, timestamp):
        img_msg = Image()
        img_msg.header.stamp = rospy.Time(timestamp)
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = "rgb8" # RGB from PIL Image format
        # img_msg.encoding = "bgr8" # BGR from cv2 format
        if img.dtype.byteorder == '>':
            img_msg.is_bigendian = 1
        img_msg.data = img.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height
        # img_msg.step = img.strides[0]

        self.img_pub.publish(img_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kitti", required=True, choices=["kitti", "tum"])
    parser.add_argument("--seq", type=int, default=7, required=True)
    args = parser.parse_args()

    rospy.init_node("MonoRecSLAM")
    
    rospy.loginfo("Initializing MonoRec SLAM node...")

    node = MonoRecNode(args.dataset, args.seq)

    try:
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
