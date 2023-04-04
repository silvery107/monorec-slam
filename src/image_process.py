import cv2
import os
import numpy as np
import shutil
import argparse

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))

def align_img_with_mask(data_path, output_path, sequence_id, n_skip=0):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=False)
    dsize_1 = (753, 376) if sequence_id==20 else (740, 370)
    dir_list = os.listdir(data_path)
    dir_list.sort()
    counter = 0
    for filename in dir_list:
        counter += 1
        if counter <= n_skip:
            print(counter, counter<=n_skip, filename)
            continue
        img_id = int(filename[:-4].replace("mask_", ""))
        assert img_id-n_skip>=0
        img = cv2.imread(data_path+filename)
        # print(img.shape)
        img = center_crop(img, dsize_1)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(output_path+f"{img_id-n_skip:06d}.png", img)

    print(f"Processed {counter} images and skipped {n_skip} images")

def align_mask_with_img(data_path, mask_path, output_path, sequence_id):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=False)
    dir_list = os.listdir(mask_path)
    dir_list.sort()
    counter = 0
    org_size = (1241, 376) if sequence_id==20 else (1226, 370)
    dsize = (753, 376) if sequence_id==20 else (740, 370)
    pad_length = (org_size[0] - dsize[0]) // 2
    print(f"Aligning masks from {mask_path}, output to {output_path}")
    for filename in dir_list:
        counter += 1
        img = cv2.imread(mask_path+filename, cv2.IMREAD_GRAYSCALE)
        # print(img.shape, img.dtype)
        h, w = img.shape[:2]
        mask_to_save = img
        mask_to_save = cv2.resize(img, dsize)
        mask_to_save = cv2.copyMakeBorder(mask_to_save, 0, 0, pad_length, pad_length, borderType=cv2.BORDER_CONSTANT, value=0)

        # assert mask_to_save.shape[0] == 376 and mask_to_save.shape[1] == 1241
        cv2.imwrite(output_path+filename, mask_to_save)
    
    print(f"Processed {counter} images")

def mask_out_img(data_path, mask_path, output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=False)
    dir_list = os.listdir(data_path)
    dir_list.sort()
    counter = 0
    print(f"Masking images from {data_path} with masks from {mask_path}, output to {output_path}")

    for filename in dir_list:
        counter += 1
        img = cv2.imread(data_path+filename)
        mask = cv2.imread(mask_path+filename, cv2.IMREAD_GRAYSCALE)
        # print(img.shape, img.dtype)
        # print(img.shape, mask.shape)
        img_to_save = img.copy()
        if mask is not None:
            mask_inv = cv2.bitwise_not(mask)
            assert mask_inv.shape[0] == img_to_save.shape[0]
            img_to_save = cv2.bitwise_and(img_to_save, img_to_save, mask=mask_inv)
        else:
            print(f"Skipped mask for image {data_path+filename}")
        # assert mask_to_save.shape[0] == 376 and mask_to_save.shape[1] == 1241
        cv2.imwrite(output_path+filename, img_to_save)
    
    print(f"Processed {counter} images")

def recover_inpaint_img(data_path, inpaint_path, output_path, sequence_id):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=False)
    dir_list = os.listdir(data_path)
    dir_list.sort()
    counter = 0
    dsize = (753, 376) if sequence_id==20 else (740, 370)
    for filename in dir_list:
        counter += 1
        img = cv2.imread(data_path+filename)
        img_to_save = img
        h, w = img.shape[:2]
        # print(f"h: {h}, w: {w}")

        inpaint = cv2.imread(inpaint_path+filename[1:])
        # print(inpaint.shape, inpaint.dtype)
        inpaint_recover = cv2.resize(inpaint, dsize)
        # print(inpaint_recover.shape, inpaint_recover.dtype)

        crop_width = dsize[0]
        crop_height = dsize[1]
        mid_x, mid_y = int(w/2), int(h/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        img[:, mid_x-cw2:mid_x+cw2+1] = inpaint_recover

        cv2.imwrite(output_path+filename, img_to_save)
    
    print(f"Processed {counter} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=7, required=True)
    args = parser.parse_args()


    sequence_id = args.seq

    # align_img_with_mask("data/kitti/20/", "data/kitti/20_resize/")
    align_mask_with_img(f"data/kitti/{sequence_id:02d}/", 
                        f"data/kitti/{sequence_id:02d}_mask/", 
                        f"data/kitti/{sequence_id:02d}_mask_align/",
                        sequence_id)
    mask_out_img(f"data/kitti/{sequence_id:02d}/", 
                 f"data/kitti/{sequence_id:02d}_mask_align/", 
                 f"data/kitti/{sequence_id:02d}_mask_out/")
