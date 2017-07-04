**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./bar.png "Dataset Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/1.jpg "Traffic Sign 1"
[image5]: ./test_images/2.jpg "Traffic Sign 2"
[image6]: ./test_images/3.jpg "Traffic Sign 3"
[image7]: ./test_images/4.jpg "Traffic Sign 4"
[image8]: ./test_images/5.jpg "Traffic Sign 5"
[image9]: ./test_images/grayscale.jpg "grayscale image"
[image10]: ./test_images/normalized.jpg "normalized image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/VikasPatil-GitHub/CarND-Traffic-Sign-Classifier.git)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Further exploratory analysis using mean, standard deviation and a histogram of the label distribution in training data showed that the labels were not uniformly distributed. Many classes had fairly small number of training examples, others had very large.

| Training Label mean   | 809.279 |
|-----------------------|---------|
| Training Label stddev | 619.420 |

Here is a bar chart showing number of training samples for each traffic sign class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

Data pre-processing:
Several pre-processing techniques have been considered for this project and almost all of them have been tried and tested. A few of them are:
1.	What: Converting the images to different grayscale. Why:prediction accuracy is better on grayscale format than on RGB format
![alt text][image9]
2.	Image normalization. Why:A well conditioned data make the optimization process easier and decrease optimization time.
![alt text][image10]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started out with lenet as a baseline and started experimenting with some additional layers to it. After a few trial and error runs involving more fully connected and dropout layers, I arrived at my final model.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 data |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6				|
| Fully connected		| Input = 400 Output = 120	|
| Dropout		| Probability = 0.75	|
| Fully connected		| Input = 120 Output = 84	|
| Dropout		| Probability = 0.75	|
| Fully connected		| Input = 84 Output = 43	|
|						|												|
|						|												|
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

batch size: 128
epochs: 25
learning rate: 0.001
probability for the dropout layer: 0.75 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The dataset is divided into 3 portions, training, validation & test. I train the model on the training test, use the validation to check for error rate and adjust until satisfied. Finally, the model’s accuracy is tested against the test set which the model hasn’t seen before.

| training set accuracy   | 99.4 % |
|-----------------------|---------|
| validation set accuracy | 93.7 %  |
|-----------------------|---------|
| test set accuracy | 91.4 %  |


If a well known architecture was chosen:
* What architecture was chosen?

LeNet architecture

* Why did you believe it would be relevant to the traffic sign application?

Using grayscale images, the prdiction accuracy of 98%, proving it's a relevant to the traffic sign application.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

| training set accuracy   | 99.6% |
|-----------------------|---------|
| validation set accuracy | 93.2%   |
|-----------------------|---------|
| test set accuracy | 92.2%  |

 The result is pretty high.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The model was run on these images and the results are presented below.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set .

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      		| Double curve   									| 
| Keep left     			| Keep left 										|
| Roundabout mandatory					| Roundabout mandatory											|
| Children crossing	      		| Children crossing					 				|
| Speed limit (20km/h)			| Speed limit (30km/h)      							|

Given 5 test images, 4 is predicted correct with 1 wrong, so the accuray is 80%.

Based on a quick look at the images itself reveals that the accurately predicted ones have distinct features that are apparent to human eye  wheres as the images that could not be determined accurately have quite a blurry part on the number,even human can not figure out clearly.

Comparing selective images from the internet against a standard set( test data set in this case) might not be a very good indicative of the model's accuracy. While looking at different such non-standard images, it seems the model might not ve very well-adpated to scenes with perspective change, multiple signs in the image and any damage to the sign( dirt, paper stuck to the sign, etc). 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The model was able to predict 4 out of the 5 signs giving a accuracy of 80%. This accuracy is achieved by using only two data preprocessing techniques viz. grayscale conversion and normalization.

For the first image, the model is relatively sure that this is a double curve (probability of 0.6).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59         			| Double curve   									| 
| 1.0     				| Keep left 										|
| .96					| Roundabout mandatory											|
| .84	      			| Children crossing					 				|
| .000022, 0.99			    | Speed limit (20km/h), Speed limit (30km/h)      							|


For the 2,3,4 images the probabilities were close to 1 indicating accurate prediction. It wasn't expected but that's what the model determined..Whereas for the fifth image the model was leaning towards 30km/h rather than 20km/h. The reason for this migth be the image part is blurry. If the this sign for tranning is more clear and larger, the acuuracy  will increase.



