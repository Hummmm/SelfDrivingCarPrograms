#Advanced Lane Finding Project


----------


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

[image1]: ./output_images/chess_board.png "Chessboard"
[image2]: ./test_images/test4.jpg "Test Road Image"
[image3]: ./output_images/undist.png "Undistorted"
[image4]: ./output_images/combined_binary.png "Binary Example"
[image5]: ./output_images/warped.png "Warp Example"
[image6]: ./output_images/fit.png "Fit Visual"
[image7]: ./output_images/final_result.png "Output"
[image8]: ./output_images/undist_example.png
[image9]: ./output_images/dist_example.png
[video1]: ./project_video_output.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
The distortion-corrected version of above image is shown below:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step.  

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in the third cell of Advanced_Finding_Lanes.ipynb.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
```python
src = np.float32([[575,464],[710,464], [1093,714], [218,714]])
img_size = (img.shape[1], img.shape[0])
dst = np.float32([[300,0],[950,0], [950,img_size[1]], [300,img_size[1]]])
```

This resulted in the following source and destination points:
<center>
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 300, 0        | 
| 710, 464      | 950, 0        |
| 1093, 714     | 950, 720      |
| 218, 714      | 950, 720      |
</center>
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Then I did some other stuff , `fit_binary_warped()` defined in the third cell of Advanced_Finding_Lanes.ipynb, and fit my lane lines with a 2nd order polynomial kinda like this:
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I did this in a funciton `curvature()`  and `distance_from_center()` in Advanced_Finding_Lanes.ipynb. I used the math equaiton for radius of curvature introduced in class and the unit is also transformed in meter. For the second one, the position of the vehicle with respect to center is calculated.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in lines # through # in my code in Advanced_Finding_Lanes.ipynb in the function `drawing()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video is included in the github repo P4 filefolder.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I faced a problem of poor margin parameter of the sliding window. I maginify the margin to absorb more data to improve the correction.

Two factors that will affect the robustness and might make pipeline fail are the quality of road binary image and how you fit the lane lines data. In this case, I combine the x-gradient and HSL color to pick up the lane pixels. It works fine for project_video.mp4, but when I test is on challenge_video.mp4, it just fails to test the correct the line. Apprently, there are other lane-line-like noises in challenge_video.mp4 such as the shadow of middle curb and the marks of road repairing. So we might need to choose better thresholding method and tuning its parameter to pick out the real lane line pixels effectively and also tune the fitting algorithm accordingly. 