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

[image1]: ./writeup_images/unseen_images.png "Unseen Images"
[image6]: ./writeup_images/train_hist.png "Training data distribution"
[image7]: ./writeup_images/label_dist_after.png "Label distribution after augmentation"
[image8]: ./writeup_images/translate_transform.png "Translate Transform"
[image9]: ./writeup_images/gray_hist.png "Grayscale and Hist Equalizer"
[image15]: ./writeup_images/unseen_1.png "Unseen hist 1"
[image16]: ./writeup_images/unseen_2.png "Unseen hist 2"
[image17]: ./writeup_images/unseen_3.png "Unseen hist 3"
[image18]: ./writeup_images/unseen_4.png "Unseen hist 4"
[image19]: ./writeup_images/unseen_5.png "Unseen hist 5"


## Dataset Exploration

### Dataset Summary

The first step I did was to use the existing code to get a sense of the data
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory Visualization
Further exploratory analysis using mean, standard deviation and a histogram of the label distribution in training data showed that the labels were not uniformly distributed. Many classes had fairly small number of training examples, others had very large.

| Training Label mean   | 809.279 |
|-----------------------|---------|
| Training Label stddev | 619.420 |

![Training data distribution][image6]


## Design and Test a Model Architecture

### 1. Data preprocessing and Enrichment
Since the number of training examples for the label classes were so highly skewed, I decided to augment the data. To do this I selected multiple transformations available in the OpenCV library and generated a list of possible transforms to apply to generate new data. Using a very naive approach, I randomly selected transforms to apply and generate new training data from data which had low frequency. 

#### Transforms used to augment the data
* [Gaussian Blur](http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html)
* [Rescaling](http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html)
* [Translating](http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html)
* [Morphological](http://docs.opencv.org/3.1.0/d9/d61/tutorial_py_morphological_ops.html)
* [Rotation](http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html)

I chose the above transforms since they modify a training data just enough to be different without changing any features of interest. 

Eventually, I managed to remove the skew by adding synthesized training data and get the number of training examples per class upto **~1400** (mean + std_dev).

![Label distribution after augmentation][image7]

#### Example: Translate transform

![Translate Transform][image8]

The first model testing I did was using colored data with all R,G,B channels. Compared to later iterations with grayscale data, the colored data accuracy was only slightly less. As such, before making any further modeling iterations, I converted the data to grayscale and applied a [histogram equalizer](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) to enhance the features of the training images which were of interest. As a final step, I normalized the data.

![Grayscale and Hist Equalizer][image9]


#### 2. Model architecture
I started the training using the LeNet architecture from a previous lab. Using the model as is, did not provide good accuracy. Looking for better architectures, I found a CNN on the course page of Stanford's [CS231](http://cs231n.github.io/convolutional-networks/) class.

The final model consisted of the following layers:

| Layer					|		Description								|
|:---------------------:|:---------------------------------------------:|
| Input					| 32x32x1 Grayscale image						|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 30x30x6 	|
| RELU					|												|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 14x14x12 	|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 12x12x18 	|
| RELU					|												|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 5x5x24 	|
| Fully Connected		| outputs 120									|
| RELU					| 												|
| Dropout				| Probability 0.75								|
| Fully Connected		| outputs 84									|
| RELU					| 												|
| Dropout				| Probability 0.75								|
| Fully Connected		| outputs 43									|
 


#### 3. Training the model
Starting with the LeNet architecture, the best accuracy that I managed to get was ~0.85 by changing the filter from 5x5 to 3x3. Given the size of the images were 32x32 it made sense to use a smaller filter to extract features. However, no modifications to the hyperparameters did not result in substantial increase in accuracy on the validation set and resulted in overfitting.

To enhance the model further, I made it deeper and added a dropout to prevent overfitting. With these enhancements the model accuracy on validation set approached 0.90. However, multiple iterations later and training for 50 epochs did not result in model accuracy above 0.93.

Implementing a model similar to the architecture mentioned on the CS231 course site, combined with max pooling and dropout layers (probability = 0.5) I managed to get an accuracy just above 0.90. Compared to the LeNet architecture it had more convolutions and 3 fully connected layers. Experimenting further with various hyperparameters of this network, I noticed that applying dropout before pooling gave a better accuracy than applying after. Increasing dropout probability from 0.5 to 0.75 and training for 20 Epochs, with a learning rate of 0.001 and batch size of 128, resulted in ~0.95 accuracy on the validation set. Removing dropout completely, resulted in overfitting with considerably high accuracy on validation set but 0.4 accuracy on images downloaded from the internet. As such I decided to stick with a 0.75 dropout.

The final model results were:
* validation set accuracy of 0.954 
* test set accuracy of 0.930
 

### Testing the Model on New Images

#### 1. Choosing five German traffic signs found on the web

Here are five German traffic signs that I found on the web and their corresponding cropped and scaled versions to conform to the model input:
*[Full scale version](test_model_images/original/)*

![Unseen Imgages][image1]

I purposefully chose some images which had a watermark on them, to see how well the model is able to identify the traffic signs. This would be similar to situations like snow stuck on a traffic sign or dust on the camera capturing the traffic signs. In addition the 2nd image has a different perspective.

#### 2. Model's predictions

Here are the results of the prediction:

| Image								|		Prediction								| 
|:---------------------------------:|:---------------------------------------------:| 
| Dangerous curve to the right		| Dangerous curve to the right					| 
| Slippery Road	 					| Slippery Road 								|
| Dangerous curve to the right		| Children crossing								|
| Turn right ahead					| Keep left						 				|
| Speed limit (30km/h)				| Speed limit (30km/h)							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 0.6. Given 

#### 3. Softmax probabilities for each prediction
![Unseen hist 1][image15] ![Unseen hist 2][image16] ![Unseen hist 3][image17] 
![Unseen hist 4][image18] ![Unseen hist 5][image19]

The 1st and 2nd images which were correctly predicted by the model have as expected very high probability for the correct class and almost negligible for others. 

Surprisingly, the 3rd image, which is incorrectly predicted, also has a almost certain class prediction. It may be possible that the watermark on the image may be throwing the model off. More investigation is required to account for this behaviour. Although, the second prediction is the correct one, its probability is way too low

The perspective distortion in the 4th image may be th reason why none of the top 5 classes do not correspond to the correct prediction. Augmenting some of the data by applying similar distortion might lead to a better accuracy.

The final, 5th image, has a fairly certain correct prediction of 30 km/h speed limit and a not so close 20 km/h at the second spot.
