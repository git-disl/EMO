# Efficient Multi-Object Tracking for Edge devices

Object Tracking is a computer vision task whose goal is to detect objects and to track the location of the 
objects throughout the video. Multi-object tracking is a sub-group of object tracking that involves tracking 
multiple instances of objects belonging to one or more categories to identify the trajectories followed by 
each of the instances, as the objects continue to move through consecutive frames of the video. Object 
tracking systems are widely used in important applications like autonomous driving, surveillance from 
security cameras and activity recognition.

The main challenges in multi object tracking are dealing with occlusions, interactions between object
instances that have similar appearance. Many of the MOT approaches perform tracking by first detecting
the object ( identify the objects belonging to a target class and denote it with bounding boxes), computing 
the similarity between detections and then associating tracklets and assigning the IDs to detections /
tracklets belonging to the same target instance. Many of the multi-object tracking systems make use of 
deep neural networks especially Convolutional Neural Networks to perform object detection on image 
frames.

For critical services that run on edge devices, latency is an important performance measure, and the 
systems are expected to achieve near real-time performance in online object tracking. Zhang et al.[1] report that FairMOT performs tracking at the rate of 25.9 FPS (Frames per second). However, as deep 
neural networks are computationally intensive, the accuracy and speed of online detection and tracking 
may degrade on edge devices which could have lesser computational resources. In cases where the 
number of frames that could be processed on the edge device is less than the number of frames in the 
incoming video, frames could be dropped at random leading to lower quality of detection and tracking.
This project aims to address these problems using optimizations that will enable multi-object tracking on 
a video to be performed faster and at the same time with minimal loss in performance metrics like MOTA
and IDF1.

When the incoming frame rate is greater than processing rate of the tracker, if the model skips performing detection on frames periodically and uses the detections from previous frames, the result of the tracking using FairMOT on a few videos from MOT-15[3] dataset can be seen below. It can be observed that the negative effects of skipping frames are more noticeable when the people in the scene as moving faster in between successive frames.

|Video|No frames skipped | Every alternate frame (1/2) skipped | 2 of every 3 frames skipped|
|-----|------------------|-------------------------------------|----------------------------|
|TUD-Stadtmitte|  <video autoplay src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4"> | <video  autoplay  src="https://user-images.githubusercontent.com/82513364/195884805-c675497c-be97-4d44-9b4b-1d0e4d1e087c.mp4">|  <video autoplay src="https://user-images.githubusercontent.com/82513364/195884827-a5f55a6a-807c-4e22-871c-37c9fbebf745.mp4">|
|KITTI-17|  <video src="https://user-images.githubusercontent.com/82513364/195884865-72b13f60-79f4-4c35-b8ea-62020e06558c.mp4"> |  <video src="https://user-images.githubusercontent.com/82513364/195884883-71402b2a-b5ce-43fd-8ce3-b97a70d58f75.mp4"> |  <video src="https://user-images.githubusercontent.com/82513364/195885097-33ea48fe-915f-418b-930a-b9da1e8a5f3d.mp4">|

Rather than dropping frames periodically, similarity between successive frames can be computed and used to determine a particular frame can be dropped without much impact or if it should not be skipped. Below are the results when skipping detection on frames that are very similar to the previously detected frames.

|Video|No frames skipped | Skipping detection based on frame similarity |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|   <video src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4">  | <video src="https://user-images.githubusercontent.com/82513364/195933899-2134e4df-8fdf-48e1-9416-e90444efc5d9.mp4">|
|PETS09-S2L1| <video src="https://user-images.githubusercontent.com/82513364/195932559-951a08ee-d1b8-415f-a6c0-6192da3e4228.mp4"> | <video src="https://user-images.githubusercontent.com/82513364/195932831-185bb5e1-80ae-4d69-b788-56417b1ffcfd.mp4">  |

Combining both the previous approaches to avoid accumulation of errors by skipping multiple consecutive frames by ensuring that detection is performed atleast on one in every given number of frames, the below results are observed

|Video|No frames skipped | Skipping detection based on frame similarity & enforcing detection on atleast one in every 4 frames |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|   <video src="https://user-images.githubusercontent.com/82513364/195884778-e4bcb9a9-628e-4faf-a411-e7ee73e28834.mp4">  | <video src="https://user-images.githubusercontent.com/82513364/195933869-d82fa65e-f896-47de-88c0-da04447eb341.mp4">  |
|PETS09-S2L1| <video src="https://user-images.githubusercontent.com/82513364/195932559-951a08ee-d1b8-415f-a6c0-6192da3e4228.mp4">   | <video src="https://user-images.githubusercontent.com/82513364/195932761-88a3993a-a572-4aa1-82fc-b45a09db98ae.mp4">  |

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
