# **Traffic Sign Classifier** 

---

**Dataset Exploraton**
* The dataset used for training, validation and testing is the one provided by the zipped file provided by Udacity: traffic-signs-data.zip
* All sub-sets include small imagelets containing a single traffic sign 28x28 image.
* Exploratory Visualization -> See Traffic_Sign_Classifier.ipynb, "Include an exploratory visualization of the dataset" section

**Design and Test a Model Architecture**
* The preprocessing consists of image normalization, i.e. with mean of zero and unit norm => pixel values ranging between -1.0 and +1.0. This technique is useful to help Tensorflow converge faster.
* The Model Architecture is similar to LeNet architecture. The modifications I brought to this model are<br>
&nbsp;&nbsp;&nbsp;&nbsp;1- Doubling the number of features calculated at every convolutional layer. Therefore the first convolutional layer uses a [5x5] kernel with a depth of 3 as input and a depth of 12 in output. Same goes for the second convolutional layer which uses a [5x5] kernel with a depth of 12 in input and 32 in output.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2- Expansion of the fully connected layers with the first layer having 800 inputs, 240 outputs and the second layer having 240 inputs and 168 outputs.<br>
&nbsp;&nbsp;&nbsp;&nbsp;3- Perform dropout at the fully connected layer, using 0.5 as keep probability during training and 1.0 during evaluation
* Model Training<br>
&nbsp;&nbsp;&nbsp;&nbsp;1- Optimizer used: The original AdamOptimizer is used, since training converges relatively quickly and using a different optimizer didn't seem to change much in my case anyway, I left it as is.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2- Batch size: Depending on my tests, I tested with batch sizes of 128 and 256. Finally a batch of 128 was optimum in my case.<br>
&nbsp;&nbsp;&nbsp;&nbsp;3- Number of Epoch: I found that 10 epochs was a bit small in some cases, especially when making the learning rate smaller. But for my current setup, I found that 15 epochs were still enough to see the training converge.<br>
&nbsp;&nbsp;&nbsp;&nbsp;4- Other hyperparameters: tf.truncated_normal uses a mu of 0 and sigma of 0.1 and learning rate is left at 0.001
* The approach I used to obtain the 0.93 or greater was simply to test methods that we learned from previous lessons. The first method I used was to add dropout to both FC layers, but was not enough. The second method I used is the one described in the model architecture (this document) which is to expand the different layers. That was enough to reach that score.

**Test a Model on New Images**
* The new images found on the Internet are shown in the .ipynb file, at the "Load and Output the Images" section. The images are not actually particularly difficult except maybe for the roundabout sign that is tilted and had some website info on top of it.
* The final result is 80%, meaning that all but one signs were classified correctly. The test set had a performance of 94.9%.
* The probalities calculated for each class appears quite strong for "Road work", "general caution" and "no entry" signs. Their respective highest scores are almost all twice the second best score. The "roundabout mandatory" and "keep right" signs however have less of a distance to their second best score which makes them suceptible to failures and the former indeed failed.

**Neural Network Layers Visualization**
* Images from convolutiona layers are displayed in the last section. While the images for the first convolutional layer look more or less like the sample given (road work), it is difficult to find any pattern in the images for the second layer.
