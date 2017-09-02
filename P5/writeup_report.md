## **Vehicle Detection Project**
### Writeup Report
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

[My writeup report](https://github.com/jinglebot/Vehicle_Detection_and_Tracking/blob/master/writeup_report.md)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle and Non-Vehicle Image Samples][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The code for the features are under the code block entitled *Features*.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Vehicle and Non-Vehicle HOG Image Samples][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and set the classifier up with them and compared the resulting vehicle detection accuracy to see which one works best. I arrived with the final choice combination of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, with `ALL hog channels` used in the computation.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the Spatial Bin, the Color Histogram and the HOG features. I stacked the 2 arrays of vehicle and non-vehicle features and then created a corresponding label list `y`. I randomized the data and split them into two: one for training and one for testing. I then used the scaler and then trained the SVC which took a long time in my old laptop. The code is under the code block entitled *Classifier*.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Based on suggestions in the forum, I tried a systematic approach on which scales to use by testing all the best values: 1, 1.5 and 2. I decided to limit my selection to two scales: 1. and 1.5 so it wouldn't be too hard on the memory. I also adjusted the starting and ending height(y) range that the sliding window would search since the small scales for the small vehicles would most likely be around the middle part of the image only and the big scales for the bigger vehicles would take more space in the lower part of the image. I used a `cell_per_step=2` which is equivalent to an overlap of 75% because it gives a better detection result.

![Image with the window boxes][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I added the use of heatmaps to heat up the detected areas. This resulted in a lot of overlapping window boxes. To merge them into one box for a detected vehicle, I used a combination of scipy's labels and threshold of the heatmap. Here are some example images:

![Test Images Optimized with Heatmap, Labels and Threshold][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Finally, I tweaked with the threshold to remove the false positives and I took the average of the heatmaps and used it to smoothen the boxes in the video.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have had a lot of problem with the projects and my laptop since it did not have a GPU. This project, like the others, requires a laptop with a higher capacity for speed and memory. Mine took most of the whole day training the classifier alone. I couldn't have finished it without the AWS. I'm not sure if the results were affected but the application of this project in realtime will not be possible on a turtle laptop.

The pipeline will most likely fail on the extraction function. It will show error if the hog channel is set to `ALL` and and visualization is set to `True` since the visualization has this choice of True or False. I would remove it if I were going to pursue this project further.

I would also try for further pursuit, as suggested in the forums, the use of Deep Learning YOLO and SSD. It looks easy or maybe it's deceiving, but I might try rewriting my projects in other versions when I have the time

Addendum:
As suggested, I made the following revisions to optimize the detection process:
- I decided to add a third scale option for  the windows size, `scale = 2`. This is equivalent to `128x128` window.
- I used `GridSearchCV` to find the best parameter, C.
- I lowered the `cell_per_step` to 1. This is equivalent to an overlap of 87.5%.
- I removed the `Heatmaps()` class statement for averaging the heatmap deque and instead used `sum`.
- I raised the Heatmaps deque maximum length, `maxlen = 25`.
- I raised the windows `threshold` to 20.
