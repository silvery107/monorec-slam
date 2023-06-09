# MonoRec SLAM
In this project, our method:

- Alleviate the tracking algorithm from using matches that belong to dynamic objects, in most cases achieving a higher accuracy in evaluation.

- Alleviate the mapping algorithm from including moving objects as part of the 3D map, obtaining a more static map of scenes.

- Achieve a good balance between real-time capability and dynamic object masking.


<img src="figures/monorec_slam.png" width="800">

<img src="figures/teaser.png" width="800">

<img src="figures/demo_keyframe.gif" width="800">

## Benchmarks

Absolute trajectory RMSE (m) for ORB-SLAM3, DynaSLAM and MonoRecSLAM on the KITTI dataset

| Sequence | ORB-SLAM | DynaSLAM | MonoRecSLAM |
|:--------:|:--------:|:--------:|:-----------:|
|    00    | **5.33** |   7.55   |    6.71     |
|    04    |   1.62   | **0.97** |    1.39     |
|    05    |   4.85   | **4.60** |  **4.60**   |
|    07    |   2.26   |   2.36   |  **2.09**   |

## Installation

<details>
<summary>Click to view details</summary>
1. Download this repo and initialize all submodules

    ```bash
    git clone git@github.com:silvery107/monorec-slam-ros.git
    git submodule update --init
    ```
    Or use `--recurse` option to clone submodules at the same time.

2. Create a conda environment
   
    `conda env create -f environment.yml`

3. Download pretrained MonoRec model and install MonoRec as a submodule
   
    ```
    cd modules/MonoRec && ./download_models.sh
    pip install -e .
    ```

4. Download the KITTI odometry dataset

    To setup KITTI Odometry, download the color images and calibration files from the 
    [official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (around 65 GB). Then unzip the color images and calibration files into `data` folder. 

    For evaluation, MonoRec uses the poses estimated by [Deep Virtual Stereo Odometry (DVSO)](https://vision.in.tum.de/research/vslam/dvso). They can be downloaded from [here](https://vision.in.tum.de/_media/research/monorec/poses_dvso.zip) and should be placed under ``data/{kitti_path}/poses_dso``. This folder structure is ensured when unpacking the zip file in the ``{kitti_path}`` directory.

5. Install `evo` for SLAM trajectory evaluation
   
   `pip install evo --upgrade --no-binary evo`

6. Install ORB-SLAM3 according to its [instructions](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/c++14_comp) and remember to install it on the `c++14_comp` branch. Install its ROS interface as well.

7. The dataset structure should be like this

    ```
    data
    ├── kitti
    │   ├── poses
    │   ├── poses_dvso
    │   └── sequences
    |       └── ...
    └── tum
        ├── rgbd_dataset_freiburg3_walking_xyz
        └── ...
    ```
</details>

## Quick Start

Here we take the sequence 07 of KITTI dataset as an example.

#### Run with ROS
The combined implementation of MonoRec SLAM is done by pumping masked image onto the `/camera/image_raw` ROS topic and utilizing the ROS interface from ORB-SLAM3 to perform SLAM. 

```bash
# terminal 1
roscore
# terminal 2
rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples/Monocular/KITTI04-12.yaml 
# terminal 3
python src/monorec_ros.py --dataset kitti --seq 7
```


or ...

#### Dataset Preprocessing
1. Run MonoRec model to get binary masks of moving objects
   
   `python src/generate_mask.py --dataset kitti --seq 7`

2. Prepare masked images from masks
   
   `python src/process_image.py --dataset kitti --seq 7`



#### SLAM with ORB-SLAM3
1. Run SLAM on precessed dataset, e.g.
    ```bash
    cd modules/ORB_SLAM3
    ./Examples/Monocular/mono_kitti ./Vocabulary/ORBvoc.txt ./Examples/Monocular/{config}.yaml ../../data/kitti/squences/07
    ```
2. Copy the resulted `KeyFrameTrajectory.txt` onto e.g. `results/kitti/07/`


## Trajectory Evaluation
1. Convert KITTI ground truth poses into TUM format for monocular KITTI evaluation purpose.
    ```bash
    python src/kitti_poses_and_timestamps_to_trajectory.py \
        data/kitti/poses/07.txt \
        data/kitti/sequences/07/times.txt \
        results/kitti/07/pose.txt
    ```
    Note that we need TUM format here since trajectories from monocular SLAM on KITTI can only be saved in TUM format.

2. Plot multiple trajectories with ground truth
    ```bash
    evo_traj tum {traj1}.txt {traj2}.txt --ref pose.txt -s --align_origin -p --plot_mode xz --save_plot plot_name
    ```
    Note that `-as` stands for `--align --correct_scale`

3. Compute absolute pose error on trajectories
    ```bash
    evo_ape tum pose.txt {traj}.txt -as -p --plot_mode xz --save_plot plot_name
    ```

4. Save plots
    ```bash
    evo_res results/*.zip -p --save_table results/table.csv
    ```


## Dependencies
- Ubuntu 20.04 with ROS Noetic
- Python >= 3.7
- OpenCV >= 4.2
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [MonoRec](https://github.com/silvery107/MonoRec)
- [evo](https://github.com/MichaelGrupp/evo)

## Acknowledgement

Team 21 of ROB 530 Mobile Robotics W23: [Yulun Zhuang](https://github.com/silvery107), [Hao Liu](https://github.com/wdliu356), [Xingqiao Zhu](https://github.com/xqiao12), [Vaishnavi Harikumar](https://github.com/ivaishi) and [Janani Peri](https://github.com/perij-codes).
