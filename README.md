# **Systems Improvements for Depth Estimation**

Course project for CS 8803 - Data Analysis with Deep Learning at Georgia Institute of Technology.

**Contributors**:  
  jShay - Jhanavi Sanjay Sheth  
  vaishnavik22 - Vaishnavi Kannan  
  imvinod - Vinod Kumar  
  pratrajv - Prateek Rajvanshi

**Problem Statement**
1. Deep learning model involves mathematical operations on high dimension image frame data at several layers.
2. Current hardwares are not designed to perform such processor intensive operations in real time.
3. Reducing the dimensionality of the input would have a direct measurable positive impact in terms of execution time.


**Why is this important?**

Autonomous vehicles, warehouse robots, industrial mobile robots are being utilized in a scale larger than ever. Perception plays a key role in enabling intelligent robots to sense the environment accurately. Any technique that reduces processing time and results in an increase of number of frames per second works in favour of better perception. Potential to directly impact how future robots perceive the world.

**Approach**

1. Input image from a monocular (single) camera. Size of the input image is : 375 X 1242 X 3
2. Dimensionality reduction using PCA to reduce the size of the input image passed to the Depth Estimator model.
3. Spatial Frame filtering - We create a mask of the pixels we want to remove from the current frame and reduce the input image size before applying object detection.
4. DL Model Architecture: Has an encoder - decoder layer.
          Encoder: Convolution
          Decoder: Upsampling
5. Skipping depth estimate for frames: if difference in depth map less than threshold (hyperparameter), skip depth calculation 5. Output is a depth estimate of the input image having size 375 X 1242 X 1

**Dataset**

Dataset used for this project is the Kitti dataset. Available for download at : http://www.cvlibs.net/datasets/kitti/

**Usage**
There are two approaches to execute the project.

Approach 1:
Main file to simulate DL model - process.ipynb
The python notebook simulates performance comparison of the DL model for all improvements done:
1. PCA
2. Frame skipping
3. Depth map projection

The notebook invokes functions implemented in the source python files.

Approach 2:
Main file to simulate DL model - process.py
The python notebook simulates performance comparison of the DL model for all improvements done:
1. PCA
2. Frame Filtering
3. Depth map projection

At present, some of the pre processing features are not integrated together in one module, and hence we have 2 approaches to run those features individually. In next phase we plan to integrate them together and keep one main file.

**TaskList**

Completed:
* [x] Research on techniques for depth estimation.
* [x] Research on system optimizations for depth estimation
* [x] Environment setup
* [x] Design architecture and implement depth estimator DL model
* [x] Trained DL model on kitti dataset
* [x] Design and implement frame skipping
* [x] Verified frame skipping logic on different test dataset
* [x] Comparison of with and without frame skipping based on time and rmse
* [x] Implement spatial frame filtering to reduce size of image passed to DL model

To do:
* [ ] Improvise PCA logic to eliminate inverse transformation

