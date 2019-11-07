# **Systems Improvements for Depth Estimation**

Course project for CS 8803 - Data Analysis with Deep Learning at Georgia Institute of Technology.

**Contributors**:  
  jShay - Jhanavi Sanjay Sheth  
  vaishnavik22 - Vaishnavi Kannan  
  imvinod - Vinod Kumar  
  Prateek  

**Problem Statement**
1. Deep learning model involves mathematical operations on high dimension image frame data at several layers.
2. Current hardwares are not designed to perform such processor intensive operations in real time.
3. Reducing the dimensionality of the input would have a direct measurable positive impact in terms of execution time.


**Why is this important?**
Autonomous vehicles, warehouse robots, industrial mobile robots are being utilized in a scale larger than ever. Perception plays a key role in enabling intelligent robots to sense the environment accurately. Any technique that reduces processing time and results in an increase of number of frames per second works in favour of better perception. Potential to directly impact how future robots perceive the world.

**Approach**
1. Input image from a monocular (single) camera. Size of the input image is : 224 X 224 X 3
2. Dimensionality reduction using PCA / Spatial Frame filtering to reduce the size of the input image passed to the Depth Estimator model.
3. DL Model Architecture: Has an encoder - decoder layer. 
          Encoder: Convolution
          Decoder: Upsampling
4. Skipping depth estimate for frames: if difference in depth map less than threshold (hyperparameter), skip depth calculation 5. Output is a depth estimate of the input image having size 224 X 224 X 1
