# Finding Lane Lines on the Road 

### Author: Jiali Wang

The goal of the project is to identify lane lines on images and on video streams. 
The source code is written in P1.ipyn as Jupyter notebook. 

---
### Program Description
Lane lines detecting pipeline consists of the following steps:
  - convert RGB image to grayscale;
  - apply Gaussian blur filter to the grayscaled image in order to suppress 
    noise and spurious gradients by averaging;
  - apply Canny edge detection to determine edges on the picture;
  - define a region of interest and black out other area;
  - detect lines in the region of interest using Hough transformation;
  - Separate left lanes and right lanes according to x values of line endpoints( in draw_lines())
  - Estimate the slope and intercept of both two lanes using linear regression( in draw_lines())
  - Draw two estimated lines on image to get the final result.

### Reflection
##### Shortcommings &  Improvements
  - The detection is possible affected by light condition; but the light can be adjusted before edge detection. 
  - There is no confidence that it will succeed in cases with a lot of sharp turns; but it should be checked first.
