# Installation and usage of yolact_ros and yolact

- Yolact means "**Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s". It is a simple, fully convolutional model for real-time instance segmentation. The paper of yolact is [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689).   
- Yolact_ros is a ROS wrapper for Yolact. Yolact uses Python 3. If you use a ROS version built with Python 2, additional steps are necessary to run the node.

## Installation of yolact
- Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
- Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:       
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
- Evaluation      
  Here are some YOLACT models (released on April 5th, 2019):

   | Image Size | Backbone      | FPS  | mAP  | Weights                                                                                                              |  |
   |:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
   | 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
   | 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw)
   | 550        | Resnet101-FPN | 33.5 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)
   | 700        | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)     | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/Eagg5RSc5hFEhp7sPtvLNyoBjhlf2feog7t8OQzHKKphjw)
   
   To process an image or a vedio, put the corresponding weights file in the `./weights` directory and run one of the following commands.

## Installation of yolact_ros 
   
### Build cv_bridge
- Install the packages rospkg and empy in the virtual environment.
  ```Shell
  sudo apt-get install python-rospkg
  pip install empy
  ```
- You need to build the cv_bridge module of ROS with Python 3. I recommend using a workspace separate from other ROS packages. Clone the package to the workspace. You might need to adjust some of the following instructions depending on your Python installation.
   - Create a folder called `cv_bridge_folder`.
   - In `cv_bridge_folder`, create a folder `catkin_ws`.
   - In `catkin_ws`, create a folder `src`.
   - Clone the package in `src`.     
     ```Shell
     git clone -b noetic https://github.com/ros-perception/vision_opencv.git
     ```
- In `catkin_ws`, use catkin_make, compile with
  ```Shell
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  ```
   - If you meet the error: 
     ```Shell
     Found package configuration file:

     /lib/x86_64-linux-gnu/cmake/boost_python-1.71.0/boost_python-config.cmake

     but it set boost_python_FOUND to FALSE so package "boost_python" is
     considered to be NOT FOUND.  Reason given by package:

     No suitable build variant has been found.

     The following variants have been tried and rejected:

     * libboost_python38.so.1.71.0 (3.8, Boost_PYTHON_VERSION=3.7)

     * libboost_python38.a (3.8, Boost_PYTHON_VERSION=3.7)
     ```   
     You can edit `vision_opencv/cv_bridge/CMakeLists.txt`, Change line No. 11 from `find_package(Boost REQUIRED python36)` to `find_package(Boost REQUIRED python)`.

   - If you meet the error: 
     ```Shell
     error: return-statement with no value, in function returning ‘void*’ [-fpermissive]
     ```
     - You can first run these codes to check the position of the folder `python3`.            
       ```Shell
       conda install setuptools
       pip install -U rosdep rosinstall_generator wstool rosinstall six vcstools
       whereis python3
       ```
     - And then edit the catkin_make command by: `catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/home/****/anaconda3/envs/****/bin/python3`   
       
- Add the following lines to the postactivate script of your virtual environment (Change the paths according to your workspace path, virtual environment and Python installation):
  ```Shell
  source $HOME/ros_python3/devel/setup.bash
  export OLD_PYTHONPATH="$PYTHONPATH"
  export PYTHONPATH="$HOME/.virtualenvs/yolact/lib/python3.6/site-packages:$PYTHONPATH"
  ```
- Add the following lines to the postdeactivate script of your virtual environment:
  ```Shell
  export PYTHONPATH="$OLD_PYTHONPATH"
  ```

### Build yolact_ros
- Create a workspace
   - Create a folder `yolact_ros_folder` 
   - In `yolact_ros_folder`, create a folder `src`
   - In `src`, download [yolact_ros](https://github.com/Eruvae/yolact_ros.git) and the related packages [yolact_ros_msgs](https://github.com/Eruvae/yolact_ros_msgs)
- In `yolact_ros_folder`, compile with the command:       
  ```Shell
  ctakin_make
  ```
- Copy the folder `yolact` to the folder `****/yolact_ros_folder/yolact_ros/scripts/yolact`.

## Usage

### Wrap yolact to a ros node
- Run `roscore`
- Wrap yolact to a ros node        
   The default model is [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing).   
   In `yolact_ros_folder`, you can run yolact using rosrun:
   ```Shell
   rosrun yolact_ros yolact_ros
   ```

   If you want to change the default parameters, e.g. the model or image topic, you can specify them:
   ```Shell
   rosrun yolact_ros yolact_ros _model_path:="$(rospack find yolact_ros)/scripts/yolact/weights/yolact_base_54_800000.pth" _image_topic:="/camera/color/image_raw"
   ```

   Alternatively, you can add the node to a launch file. An example can be found in the launch folder. You can run that launch file using:
   ```Shell
   roslaunch yolact_ros yolact_ros.launch
   ```

- You can check ros topics and ros nodes.
   ```Shell
   rostopic list
   rosnode list
   ```

### Evaluate images and videos      
You can run these commands in `****/yolact_ros_folder/yolact_ros/scripts/yolact`.        
- Images    
   ```Shell
   # Display qualitative results on the specified image.
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

   # Process an image and save it to another file.
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

   # Process a whole folder of images.
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
   ```
     
- Video    
   ```Shell
   # Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
   # If you want, use "--display_fps" to draw the FPS directly on the frame.
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

   # Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

   # Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
   python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
   ```
    
   As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.   
   ```Shell
   python eval.py --help
   ```



