**Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

You can also view the video at run1.mp4

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers of 5x5 filter sizes and depths of 32 and 16 with a stride of 2, and one 5x5 layer with a stride of 1 and a depth of 8.  (model.py lines 100-109). I then have 2 layers of 2x2 filter sizes and depths of 4 and 2 with a stride of 1. (model.py 112-117). We then apply max pooling and an initial dropout layer before flattening the network into a fully connected neural network. Here we have 4 fully connected layers, with a dropout layer before a final output node. (model.py 126-141). Each activation is a RELU layer. The data has been normalized and cropped to reduce noise using 2 Keras Lambda layers. (model.py 90-93).

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, I have two dropout layers. One before flattening and one before outputting the final steering angle. (model.py 123 & 139).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 144).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. First, I drove straight around for two laps. Then, I drove as if I was constantly oversteering and then correcting my oversteer. I then supplemented this data with recovery from the dirt patch and recovery from oversteering before the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to at first recreate the NVIDIA self driving neural network(https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). So I googled "Nvidia architecture in keras" and found this code from the founder of vector ai - https://github.com/andrewraharjo/SDCND_Behavioral_Cloning/blob/master/model.py. This isn't exactly the NVIDIA architecture but it was influenced by it and it adds some interesting layers like dropout and max pooling. The downside is that it is very slow and outputs very many parameters in the convolution layers. It took about 45 minutes to train on my GPU-less laptop. 

I then decided to mimick the NVIDIA architecture. This cut down the training time to about 3 minutes, but the model was not ideal. I went back to collect some more training data. This was still not enough. 

I then used the left and right images, and flipped those images as well. I also manually appended my training data to include training data where I avoid the dirt patch, and where I correct myself from swerving off the road. For each row of data, I now had 6 data points, the original central image, the left image, the right image, and the flipped of each, which gave me about 160000 training samples. I was convinced that this was enough data. The model was still too simple, so I tweaked the depths a little, by reducing them.

At this point I had gotten the model to almost run the entire track, but it clipped a bank during one of the turns. I decided to sacrifice the speed of the training process, by changing the last 5x5 layer stride length from 2 to 1, and change the depths to a sequentially decreasing pattern like andrewraharjo's model. My convolutional depths were 32, 16, 8, and 4. I then had a MaxPooling layer and a 25% dropout layer in between the flattening and another 50% dropout layer at the end of the fully connected neural network. The dropuout and max pooling positions I got from andrewraharjo's code. I tried tweaking the 25% dropout layer to 50% to see what would happen but this only made the model worse. I guess any dropout layer that isn't the penultimate layer works best if it's not 50%. The Max Pooling layer is interesting because it deviates from LeNet's implementation where there was a max pooling layer in between each convolutional layer. 

The normalization code was taken straight from the lectures about Lambda layers. I really like the lambda layers because it abstracts away going through each image and manually messing with raw pixel data. 

I'm fairly proud of my final model. It took about 12 minutes to train on my laptop all while staying on the track. Although at one point in the video (1:06) it comes dangerously close to moving off the track. I attribute this to faulty training data. When I was recording the "recovery" turning, I sometimes would leave in the recording where I swerved off the road.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of:
* 2 5x5 convolutional layers with a stride of 2x2 
* 1 5x5 convolution with a stride of 1x  
* 1 2d max pooling layer with 2x2 pool size, 
* 1 25% dropout layer 
* 4 fully connected layers with a penultimate 50% droupout layer.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior I recorded two laps of myself driving through the center of the lane, followed by another lap where I drove very crookedly. What I mean is that I would oversteer and then correct myself. This lead to my car recovering pretty well, but since the act of oversteering was in the training data, it also lead to my car falling off the road in some specific areas of the track. 

[image1]: https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/ "Over Steering"

I then supplemented my training data with recovery turns, where I was on the edge of the road but then I go back to a central driving lane. This helped drastically but then my card would never turn away from the dirt road. So I trained it to steer away from the dirt road like this:

![dirt-road-sample](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/center_2017_07_07_15_03_20_716.jpg)

Not actual sample, but essentially what I did:

![dirty-road](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/recovery-from-dirt.png)

#### Other training samples:

Left camera recovery from oversteer

![left-camera-recovery-from-oversteer](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/left_2017_07_07_15_00_46_270.jpg)

Turning left

![turning-left](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/right_2017_07_07_15_02_17_523.jpg)


Right camera oversteer

![right-oversteer](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/right_2017_07_07_15_03_17_598.jpg)


Avoiding the road

![left-avoiding-dirt-road](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/left_2017_07_07_15_03_20_922.jpg)


Center Oversteer

![center-oversteer](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/center_2017_07_07_14_59_38_448.jpg)

Center

![center](https://github.com/krashidov/sdc-p3-Behavioral-Cloning/raw/master/center_2017_07_07_14_59_34_853.jpg)

This resulted in about 160000 image samples, which I was pretty happy about. Please note that data-2 is the good version of the data. The 'data' directory is my attempt to solve the problem with only driving in the center lane. I realized that the network needs to know how to recover otherwise it won't be very safe.

The validation set is 20% of the data set. I used the adam optimizer. 

