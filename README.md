# Efficient Multi-Object Tracking for Edge devices

Object tracking is a representative computer vision task to
detect and track objects throughout the video. Object 
tracking systems are widely used in important applications like autonomous driving, surveillance from 
security cameras and activity recognition. Multi-object tracking is a subgroup of object tracking
that tracks multiple objects belonging to one or more categories by identifying the trajectories as the objects move
through consecutive video frames. Multi-object tracking has
been widely applied to autonomous driving, surveillance with
security cameras, and activity recognition. A popular paradigm
for MOT is tracking-by-detection, which first detects
objects by marking them with object class labels and bounding
boxes, computes the similarity between object detections, and
associate tracklets by assigning IDs to detections and tracklets
belonging to the same object. Online object tracking aims
to process incoming video frames in real time as they are
captured. However, deep neural networks (DNNs) powered
multi-object trackers are compute-intensive, e.g., using convolutional neural networks (CNNs), such as YOLOv3
and Faster RCNN, for detecting objects. When deployed
on edge devices with resource constraints, the video frame
processing rate on the edge device may not keep pace with
the incoming video frame rate. This mismatch can result in lag
or dropped frames, ultimately diminishing online object
tracking quality. 

In this project, we focus on reducing the computational cost
of multi-object tracking by selectively skipping detections
while still delivering comparable object tracking quality. First,
we analyze the performance impacts of periodically skipping
detections on frames at different rates on different types of
videos in terms of accuracy of detection, localization, and
association. Second, we introduce a context-aware skipping
approach that can dynamically decide where to skip the detections and accurately predict the next locations of tracked objects. 
Third, we conduct a systematic experimental evaluation
on the MOTChallenge datasets, which demonstrates
that the proposed approach can effectively reduce computation
costs of multi-object tracking by skipping detections and
maintain comparable multi-object tracking quality to the no
skipping baseline.

## Sample Results
When the incoming frame rate is greater than processing rate of the tracker, if the model skips performing detection on frames periodically and uses the detections from previous frames, the result of the tracking using FairMOT on a few videos from MOT-15 dataset can be seen below. It can be observed that the negative effects of skipping frames are more noticeable when the people in the scene as moving faster in between successive frames.

|Video|No frames skipped | Every alternate frame (1/2) skipped | 2 of every 3 frames skipped|
|-----|------------------|-------------------------------------|----------------------------|
|TUD-Stadtmitte|  <video autoplay src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4"> | <video  autoplay  src="https://user-images.githubusercontent.com/82513364/195884805-c675497c-be97-4d44-9b4b-1d0e4d1e087c.mp4">|  <video autoplay src="https://user-images.githubusercontent.com/82513364/195884827-a5f55a6a-807c-4e22-871c-37c9fbebf745.mp4">|
|KITTI-17|  <video src="https://user-images.githubusercontent.com/82513364/195884865-72b13f60-79f4-4c35-b8ea-62020e06558c.mp4"> |  <video src="https://user-images.githubusercontent.com/82513364/195884883-71402b2a-b5ce-43fd-8ce3-b97a70d58f75.mp4"> |  <video src="https://user-images.githubusercontent.com/82513364/195885097-33ea48fe-915f-418b-930a-b9da1e8a5f3d.mp4">|

Rather than dropping frames periodically, eigenvalue based similarity between successive frames can be computed and used to determine a particular frame can be dropped without much impact or if it should not be skipped. Below are the results when skipping detection on frames that are very similar to the previously detected frames.

|Video|No frames skipped | Skipping detection using eigenvalue based similarity |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|   <video src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4">  | <video src="https://user-images.githubusercontent.com/82513364/195933899-2134e4df-8fdf-48e1-9416-e90444efc5d9.mp4">|
|PETS09-S2L1| <video src="https://user-images.githubusercontent.com/82513364/195932559-951a08ee-d1b8-415f-a6c0-6192da3e4228.mp4"> | <video src="https://user-images.githubusercontent.com/82513364/195932831-185bb5e1-80ae-4d69-b788-56417b1ffcfd.mp4">  |

Combining both the previous approaches to avoid accumulation of errors by skipping multiple consecutive frames by ensuring that detection is performed atleast on one in every given number of frames, the below results are observed

|Video|No frames skipped | Skipping detection using eigenvalue based frame similarity & enforcing detection on atleast one in every 4 frames |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|   <video src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4">  | <video src="https://user-images.githubusercontent.com/82513364/195933869-d82fa65e-f896-47de-88c0-da04447eb341.mp4">  |
|PETS09-S2L1| <video src="https://user-images.githubusercontent.com/82513364/195932559-951a08ee-d1b8-415f-a6c0-6192da3e4228.mp4">   | <video src="https://user-images.githubusercontent.com/82513364/195932761-88a3993a-a572-4aa1-82fc-b45a09db98ae.mp4">  |


Qualitative results for Context-aware skipping (using Kalman filter based estimation, Normalized cross correlation and Histogram of Oriented Gradients features based similarity) to be updated soon.

## Usage Instructions

### Environment setup:

```
conda create --name <env_name> python=3.7
conda activate <env_name>

git clone https://github.com/git-disl/EMO.git
pip install -r requirements.txt

git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd ../DCNv2
sh make.sh
```

A trained model from https://github.com/ifzhang/FairMOT#pretrained-models-and-baseline-model can be used for the inference below

### Script to run evaluation

#### Periodic skipping + estimation
```
python src/track.py mot --load_model <path_to_FairMOT_trained_model> --conf_thres 0.6 <--val_mot15 VAL_MOT15 / --val_mot17 VAL_MOT17> --data_path <path_to_MOT15/MOT17_dataset> --detect_frame_interval 4 --similarity_computation 'no' --adaptive_freq_forced_detection 'False'
```
#### Context-aware skipping (NCC/HOG + estimation)
```
python src/track.py mot --load_model <path_to_FairMOT_trained_model> --conf_thres 0.6 <--val_mot15 VAL_MOT15 / --val_mot17 VAL_MOT17> --data_path <path_to_MOT15/MOT17_dataset> --similarity_threshold <0-1, recommended range 0.7-0.9> --detect_frame_interval 4 --similarity_computation <'hog'/'ncc'> --adaptive_freq_forced_detection 'False'
```

## Citation
The arxiv version of the paper can be found [here](https://arxiv.org/abs/2309.02666)
```
@article{ganesh2023fast,
  title={Fast and Resource-Efficient Object Tracking on Edge Devices: A Measurement Study},
  author={Ganesh, Sanjana Vijay and Wu, Yanzhao and Liu, Gaowen and Kompella, Ramana and Liu, Ling},
  journal={arXiv preprint arXiv:2309.02666},
  year={2023}
}
```

## Acknowledgement
This code in this repository is developed on using the following repositories

[FairMOT](https://github.com/ifzhang/FairMOT) - [1] Zhang, Y., Wang, C., Wang, X., Zeng, W., & Liu, W. (2021). Fairmot: On the fairness of detection and re-identification in multiple object tracking. International Journal of Computer Vision, 129(11), 3069-3087`

[HOTA metrics](https://github.com/nekorobov/HOTA-metrics) - [2] J. Luiten, A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixe, and B. Leibe. “Hota: A higher order metric for evaluating multi-object tracking,” in IJCV, 2020.


The videos used for the analysis are from 

https://motchallenge.net/data/MOT15/
[3] Leal-Taixé, L., Milan, A., Reid, I., Roth, S. & Schindler, K. MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking. arXiv:1504.01942 [cs], 2015., (arXiv: 1504.01942)
