**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort]: ./images/undistort.png
[chessboard]: ./images/chessboard.png
[binarythreshold]: ./images/binarythreshold.png
[perspective]: ./images/perspective.png
[windowsearch]: ./images/windowsearch.png
[fullprocess]: ./images/fullprocess.png
[curvature]: ./images/curvature.png
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
The whole code is cotained in the notebook names `notebook.ipynb`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Camera Calibration of the IPython notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration image using the `cv2.undistort()` function and obtained this result: 

![alt text][chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image described in the Binary Threshold section. For this I used
* Color transform: we combine the H and S channels and use a min/max threshold
* Gradient transform: we use a min/max threshold on the x gradient and the angle (y gradient and magnitude are coded, but not enforced in the final parameters selected)

Here's an example of my output for this step.  (note: this is not actually from one of the original test images)

![alt text][binarythreshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transf_persp()`, which appears in the Transform Perspective section of the notebook.  The `transf_persp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
dy, dx = img.shape[0], img.shape[1]
# bottom and top left points of the rectangle
mx, my = 600, 447
bx, by = 200, dy
# Transformation
src = np.float32([[bx,by],[mx,my],[dx-mx,my],[dx-bx,by]])
dst = np.float32([[bx+100,by],[bx+100,0],[dx-bx-100,0],[dx-bx-100,by]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720        | 
| 600, 447      | 320, 0      |
| 660, 447     | 980, 0      |
| 1080, 460      | 980, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then, to identify the points belonging to each line, we used a 2 pronged approach:
* Initial lane points is found through the method of sliding windows
* Further lane points are found looking at points in the neighborhood of the previous lane

To summarize the initial lane point finding, shown [here](https://www.youtube.com/watch?v=siAMDK8C_x8)
* the start of the lanes at the bottom of the image is found by looking at peaks in the distribution of non-zero points in the bottom half of the image along the x axis
* then for as many times as we have vertical "windows", we look for points withing a window around the x-axis peaks, and update the value of those peaks using the average of the coordinates of the points found

Once we have found points belonging to the lanes, through either method, we can fit a 2 degree polynomial separately for each line.

The image below shows the impact on the 2 methods below: on the left the original lane initialization, then the search around the previously found lanes

![alt text][windowsearch]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step can be found in the "Curvature and position calculation" section, I used the formula shown below, using a scaling factor to transform the nb of pixel into meters.

![alt text][curvature]

In the processed image, we show the radius of the curvature (i.e. the inverse)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, we drew the zone between the line in green and combined all the step in the "Build pipeline" section.  Here is an example of my result on an image extracted from the video:

![alt text][fullprocess]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are several potential shortcomings to the approach:
* Lane inialization could be done with convolutions
* Polynomial extrapolation could take further advantage of the parallelism of lanes. For example, we could constrain the polynomials to have closer coefficients (except the intercept). A rough attempt at this can be found in the `poly_fit()` function
* More fail safe mechanisms could be build. Currently, there we have a search zone corresponding to an auto-regressive mean. Using a moving window could be more reactive. When averaging, we could weight based on a confidence index related to the number of lane points found 
* Binary image should also be tested in more situations to make sure we have the most robust parameters (e.g. currently not using y gradient or magnitude)
