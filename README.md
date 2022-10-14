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
systems are expected to achieve near real-time performance in online object tracking. Zhang et al.[6]
, report that FairMOT[1] performs tracking at the rate of 25.9 FPS (Frames per second). However, as deep 
neural networks are computationally intensive, the accuracy and speed of online detection and tracking 
may degrade on edge devices which could have lesser computational resources. In cases where the 
number of frames that could be processed on the edge device is less than the number of frames in the 
incoming video, frames could be dropped at random leading to lower quality of detection and tracking.
This project aims to address these problems using optimizations that will enable multi-object tracking on 
a video to be performed faster and at the same time with minimal loss in performance metrics like MOTA
and IDF1.

When the incoming frame rate is greater than processing rate of the tracker, if the model skips performing detection on frames periodically and uses the detections from previous frames, the result of the tracking using FairMOT on a few videos from MOT-15 dataset can be seen below. It can be observed that the negative effects of skipping frames are more noticeable when the people in the scene as moving faster in between successive frames.

|Video|No frames skipped | Every alternate frame (1/2) skipped | 2 of every 3 frames skipped|
|-----|------------------|-------------------------------------|----------------------------|
|TUD-Stadtmitte|    |   |   |
|KITTI-17|    |   |   |
|PETS09-S2L1|    |   |   |

Rather than dropping frames periodically, similarity between successive frames can be computed and used to determine a particular frame can be dropped without much impact or if it should not be skipped. Below are the results when skipping detection on frames that are very similar to the previously detected frames.

|Video|No frames skipped | Skipping detection based on frame similarity |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|    |   |
|KITTI-17|    |   |
|PETS09-S2L1|    |   |

Combining both the previous approaches to avoid accumulation of errors by skipping multiple consecutive frames by ensuring that detection is performed atleast on one in every given number of frames, the below results are observed

|Video|No frames skipped | Skipping detection based on frame similarity & enforcing detection on atlest one in every 4 frames |
|-----|------------------|-------------------------------------|
|TUD-Stadtmitte|    |   |
|KITTI-17|    |   |
|PETS09-S2L1|    |   |
