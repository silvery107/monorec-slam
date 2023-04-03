# MonoRec SLAM



## TODOs
- [x] Run MonoRec SLAM on KITTI sequence 00, 04, 07 and 08
- [x] Organizing monorec and orb-slam in `modules` folder as git submodules
- [ ] Test all scripts can be run in a correct workpath
- [ ] Implment a monorec ROS node



## Installation

1. Create a conda environment
    `conda env create -f environment.yml`

2. Download pretrained MonoRec model
    ```cd modules/MonoRec && ./download_models.sh```

3. Download the KITTI odometry dataset

    To setup KITTI Odometry, download the color images and calibration files from the 
    [official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (around 65 GB). Then unzip the color images and calibration files into `data` folder. 

    For evaluation, MonoRec uses the poses estimated by [Deep Virtual Stereo Odometry (DVSO)](https://vision.in.tum.de/research/vslam/dvso). They can be downloaded from [here](https://vision.in.tum.de/_media/research/monorec/poses_dvso.zip) and should be placed under ``data/{kitti_path}/poses_dso``. This folder structure is ensured when unpacking the zip file in the ``{kitti_path}`` directory.

4. Install `evo` for SLAM trajectory evaluation
   `pip install evo --upgrade --no-binary evo`

5. Install ORB-SLAM3 according to its [instructions](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/c++14_comp) and remember to install it on the `c++14_comp` branch

## Quick Start



### Dataset Preprocessing
1. Run MonoRec model to get binary masks of moving objects
   `python monorec_mask.py`

2. Prepare masked images from masks
   `python src/image_preprocess.py`



### SLAM with ORB-SLAM3
```
cd modules/ORB_SLAM3
./Examples/Monocular/mono_kitti ./Vocabulary/ORBvoc.txt ./Examples/Monocular/{config}.yaml ../../data/{kitti_path}/{seq_id}
```


### Trajectory Evaluation
1. Convert KITTI ground truth poses into TUM format for monocular KITTI evaluation purpose.
   ```
   python src/kitti_poses_and_timestamps_to_trajectory.py \
   ../data//{seq_id}/{seq_id}.txt \
   ../data/{seq_id}/times.txt \
   ../data/{seq_id}/{seq_id}_gt.txt
   ```
2. Plot multiple trajectories with ground truth
    ```
    evo_traj tum traj1.txt traj2.txt --ref traj_gt.txt -p --plot_mode=xz
    ```
    Note that we use `tum` here since trajectories from mono SLAM on KITTI can only be saved in TUM format.
3. Compute absolute pose error on trajectories
   ```
   evo_ape tum traj_gt.txt traj.txt --align --correct_scale
   ```


## ROS Support
Coming soon...

The combined implementation of MonoRec SLAM is under implementing by piping masked image onto the `/camera/image_masked` ROS topic and utilizing the ROS interface from ORB-SLAM3 to perform SLAM.

## Benchmarks

Compare SLAM result on KITTI dataset using ORB-SLAM3, DynaSMAL and MonoRecSLAM

| Sequence | ORB-SLAM | DynaSLAM | MonoRecSLAM |
|:--------:|:--------:|:--------:|:-----------:|
|    00    |   5.33   |   7.55   |             |
|    04    |   1.62   |   0.97   |             |
|    07    |   2.26   |   2.36   |             |
|    08    |   46.68  |   40.28  |             |

## Dependencies
- Ubuntu 20.04 with ROS Noetic
- Python >= 3.7
- OpenCV >= 4.5
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [MonoRec](https://github.com/Brummi/MonoRec)
- [evo](https://github.com/MichaelGrupp/evo)