# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./images/nvidia-architecture.png "NVIDIA architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```   

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with
* filter varying between 3x3 and 5x5, with stride of 2 and 1 respectively
* depth varying between 24 and 64

The structure implemented (model.py lines 108-122) looks like the following from NVIDIA, except the input which has dimensions 3 @ 70, 320 after pre-processing.

The model includes RELU layers to introduce nonlinearity on all layers except the output layer.

Before entering the core of the model, images are preprocessed through cropping of top and bottom (top 65 and bottom 25 pixels), and normalized using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The NVIDIA model did not mention drop out, but we introduced one at the exit of the convolutional letters, with dropout probability 50% (model.py lines 118). 

We also stopped after 7 epochs of 8,192 images.

After an attempt at creating additional datasets, without joystick, we noticed that the result was usually not better. In the end, we used the dataset provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

The main way to reduce overfitting was to use image augmentation described below.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first the NVIDIA architecture seemed to have too much parameter, so we tried to employ an approach with fewer layers, namely 2 times 2 convolutional layers in a row followed by maxpool, and 2 fully connected layers with dropout.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The car had a strong tendency to go straight (when using all the sample with 0 steering angle) or oscillate (when only part of those images). This implied that the car was overfitting.

To combat this overfitting, we saw that the key was data augmentation. After trying to add new training datasets, we implemented a data augmentation strategy described below, which meant that we could in parallel use a more complex model.

Therefore, we adapted the model architecture from the NVIDIA article to the dimensions of the dataset. We introduced a dropout layer, although the NVIDIA made no mention of it, because the dataset appeared more limited and the image larger.

The final step was to run the simulator to see how well the car was driving around track one with success. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It still has a tendency to confuse shadows (e.g. electric lines and trees) with borders, but it stays on the track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 108-122) consisted of a convolution neural network with the following layers and output layer sizes:
* Preprocessing: cropping of top 65 and bottom 25 pixels # 3 x 70 x 320
* Preprocessing: normalization # 3 x 70 x 320
* Convolution: shape (5x5x24), stride of (2x2), "valid" padding, "relu" activation  # 24 x 33 x 158
* Convolution: shape (5x5x36), stride of (2x2), "valid" padding, "relu" activation  # 36 x 15 x 77
* Convolution: shape (5x5x48), stride of (2x2), "valid" padding, "relu" activation  # 48 x 6 x 37
* Convolution: shape (3x3x64), stride of (1x1), "valid" padding, "relu" activation  # 64 x 4 x 35
* Convolution: shape (3x3x64), stride of (1x1), "valid" padding, "relu" activation  # 64 x 2 x 33
* Flatten and Dropout with 50% probability # 4,224
* Dense: "relu" activation # 100
* Dense: "relu" activation # 50
* Dense: "relu" activation # 10
* Dense: # 1

It looks similar to this:

![alt text][nvidia]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2 = x100]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
