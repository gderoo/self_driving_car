#**Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps
- image was converted to grayscale
- edges were calculated with the Canny edge method
- picture was masked to focus on a bottom part
- lines were calculated in the Hough Space
- finally, in addition to the course videos, we "averaged" the lines

In order to draw a single line on the left and right lanes, I created new draw_lines() function. In the last version (draw_lines2), we do the following:
- split the Hough Lines, depending on whether the x coordinates cross the middle line (simpler than the splope suggestion)
- doing seperate regression on the line extremities for the left and right panels

###2. Identify potential shortcomings with your current pipeline

There are still several shortcomings:
- despite the tinkering, it still feels that the dotted lines are hard to capture during the HoughLine step
- if the line turns too much and lines cross the middle, the solution will not work

###3. Suggest possible improvements to your pipeline

Possible improvements would be to:
- limit the angles of the Hough Lines
- try to do the regression at the end of the canny edge step directly
- use the continuity of directions to stabilize the algorithm (i.e. reduce variance and introduce some bias):
  - at each image, we calculate the lines, together with a confidence probability (the longer the Hough Lines, the more confidence we have)
  - At time t, the lines are then given by doing a weighted average of the lines of Image t, but also Image t-1 and Image t-2 for instance.
