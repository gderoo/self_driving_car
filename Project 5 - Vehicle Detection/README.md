# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[original]: ./images/original.png
[car_features]: ./images/car_features.png
[nocar_features]: ./images/nocar_features.png
[windows]: ./images/windows.png
[heat1]: ./output_images/heatmap_test1.png
[heat2]: ./output_images/heatmap_test2.png
[heat3]: ./output_images/heatmap_test3.png
[heat4]: ./output_images/heatmap_test4.png
[heat5]: ./output_images/heatmap_test5.png
[heat6]: ./output_images/heatmap_test6.png
[class_errors]: ./images/class_errors.png
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Most of the code is in this [notebook](https://github.com/gderoo/self_driving_car/blob/master/Project%205%20-%20Vehicle%20Detection/notebook.ipynb), while the functions developped during the class are directly in this [python file](https://github.com/gderoo/self_driving_car/blob/master/Project%205%20-%20Vehicle%20Detection/lesson_functions.py)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Labeled images were taken from the GTI vehicle image database GTI and the KITTI vision benchmark suite. All images are 64x64 pixels. A third data set released by Udacity was not used here. In total there are 8792 images of vehicles and 8968 images of non vehicles.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][original]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I applied the HOG  algorithm to the different channels to see the result.

Here is an example using the `HLS` color space and the following HOG parameters: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![alt text][car_features]
![alt text][nocar_features]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the ones showed above because:

* HLS color space seemed to have the best results with the L-channel appears to be most important, followed by the S channel. I discarded RGB color space under changing light conditions. YCrCb also provided good results
* Increasing the orientation would have increased the numer of parameters but did not seem to have an impact on the SVM 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

On top of the HOG parameters, we also use the spatial bin and the color histogram (the main feature extraction function is `extract_features()` which can be seen in line 97-116 of the python file, and is called in the "Feature extraction" section of the notebook)

As suggested, I trained a linear SVM (which can be seen in the "Classifier fitting" section of the notebook). The result was a SVM with 99% accuracy on the test set which seemed good enough.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to choose a series of windows that would follow a perspective effect because smaller cars should only be further on the horizon. In addition to this, I decided to put some randomness in the x and y origins in order to help smooth the behaviors of the detector. The main function `slide_window()` was modified from the lesson functions and can be found line 122-166 of `lesson_functions.py`

![alt text][windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the features described above. To further optimize the performance,

* the SVM was trained on features which were scaled across training images
* I also used a GridSearch on the C parameter, with an locally optimal value found for 0.01 (optimal value could be lower, but we were afraid to overfit)

Here are some example of false positives:

![alt text][class_errors]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_processed.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps, togther with the resulting bounding boxes:
![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]
![alt text][heat5]
![alt text][heat6]

We can notice that the 3rd seems to fail to identify, but during the video this does not manifest, because the randomization of window start combined with the averaging over the past 5 images corrects for that mistake.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are a couple of issues we can anticipated:

* Risk associated to generalization of different luminosity context, etc
* We can also notice that the car is not very precisely boxed, with the box sometimes covering only part (despite the smoothing and randomization techniques employed)
* Last, we used a lot of features, so the biggest concern at this stage is the computing speed:

Potential improvements for the computing spee could be:

* Drop the features with least impact to help increase computing speed too
* Implement a faster HOG calculation strategy in which the HOG is calculated once, and not for each window separately
