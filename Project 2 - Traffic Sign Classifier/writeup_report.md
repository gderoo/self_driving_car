# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[barchart]: ./images/bar_chart.png "Bar Chart"
[examples]: ./images/sign_examples.png "Sign Examples"
[grey]: ./images/sign_grey.png "Grayscaling"
[tilt]: ./images/sign_tilt.png "Tilting"
[german]: ./images/sign_german.jpg "Internet Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gderoo/self_driving_car/blob/master/Project%202%20-%20Traffic%20Sign%20Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Training = 34799
* Validation = 4410
* Test = 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

During our exploration of the the dataset, we build 2 main visuals:

* Barchart showing the number of examples per class. We see that it varies for 200 to 2000. If we estimate that this is biases compared to reality, we could consider loss functions that give the same weight to each class (vs. each sample)

![alt text][barchart]

* Random examples of each class. We can see that beyond the variation in class, there is variation in "image quality", contrast, etc.

![alt text][examples]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

We performed 4 processing steps:

* Augmenting the data via tilting
* Convert to greyscale
* Perform histogram equalization
* Normalize the data

The code for this step is split in 2:

* In Step 2, the section "Pre-process the Data Set" contains a series of helper functions that were considered, together with an example of the transformation used for the final neural network
* The actual execution on the datasets is done in the section "Train, Validate and Test the Model" of Step 2 for simplicity reasons (avoids having to navigate constantly up the notebook, when changing the number of colours for example)

**Augmenting:** We decided to augment the data, because there is a strong imbalance between classes in the original dataset, which led the model to overfit from the first epoch.

![alt text][tilt]

**Grey:** We originally decided to convert the images to grayscale to reduce the computation, limit overfitting. This could be forgotten later down, since we ended up increase the size of the model.

**Histogram Equalization:** we used histogram equalization described [here](https://en.wikipedia.org/wiki/Histogram_equalization) to correct for the strong variation in contrast.

![alt text][grey]

**Normalization:** We then normalize to improve the convergence of the gradient descent.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The actual execution of preprocessing is done in the section "Train, Validate and Test the Model" of Step 2.

We follow the split of the original dataset, but since we performed an augmentation. The resulting dataset sizes are

* Training = **56337**
* Validation = 4410
* Test = 12630

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After several more complicated trials, including a reproduction of the [LeCun architecture](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) for a similar issue, we fell back on the architecture of the original LeNet homework augmented by convolutional nets.

The code for my final model is located in Step 2, as a combination of the "Net" function described in the "Architecture" section, and the hyperparameters in the first cell of the "Train, Validate and Test the Model" section.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey Image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x12 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x12 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x36 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x36 				|
| Flatten	      	| input 5x5x36 output 900				|
| Fully connected		| outputs 400      									|
| RELU					|												|
| Dropout	| keep_prob = 50%         									|
| Fully connected		| outputs 200       									|
| RELU					|												|
| Dropout	| keep_prob = 50%         									|
| Fully connected		| outputs 43        									|


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the section "Train, Validate and Test the Model" of Step 2. 

To train the model, I used an AWS instance.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.966
* test set accuracy of 0.933

Main steps to find the solution:
* I started with the LeNet architecture, in RGB. However the model was quickly overfitting: validation accuracy was 10% lower than training accuracy from EPOCH 1 (i.e. valid:0.45 vs train:0.5), and the training accuracy did not go above 0.94
* The main adjustements were to
  * complexify the model to allow the model to reach higher accuracies (several convolutions in a row before max pool), as well as increasing the depth of the convolution (6,16 to 12, 36)
  * reduce the overfitting by going to grey scale, introducing a dropout rate, and augmenting the original dataset
* We also reduced the original sigma and the learning rate because the ramp-up of accuracy seemed very high (and still is to some extent)
* The introduction of several convolution in a row before max pooling because it seemed promising while looking at the [literature](https://www.cs.toronto.edu/~frossard/post/vgg16/)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][german]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
