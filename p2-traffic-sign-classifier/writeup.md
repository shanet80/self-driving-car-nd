#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization1.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/test1.png "Traffic Sign 1"
[image5]: ./test_images/test2.png  "Traffic Sign 2"
[image6]: ./test_images/test3.png  "Traffic Sign 3"
[image7]: ./test_images/test4.png  "Traffic Sign 4"
[image8]: ./test_images/test5.png  "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because to attempt to remove any inconsistencies in the images which may be picked up by the network as features.

I decided to generate additional data because in general more data will mean more significant features are identified by the model.

To add more data to the the data set, I split the data into a train test split using sklearn and iteratively made small random changes to the images rotation, zoom, or shifted height and/or width.

The difference between the original data set and the augmented data set is the following the augmented set is larger and more varied due to the variation introduced during data generation. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 128 image batch size and ran 5000 batches per epoch. I then ran through 30 epochs. I used the tensorflow optimizer to minimize the sparse softmax cross entropy with logits function.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8
* validation set accuracy of 99.4
* test set accuracy of 94.6

I first attempted to use a smaller batch size and a decaying learning rate but the results I was getting were high 80s to low 90s and would frequently decrease before the rate would decay further. Changing to the batches within epochs greatly increased the training time of the model but also resulted in much higher accuracy scores.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it was streched during resizing and also has some watermarking over the image.

The second image might be difficult to classify because it is somewhat farther back in the image.

The third image seems like it would be the easiest to classify. It was not changed much at all by resizing and had no background. The softmax functions showed a 72% confidence in the correct prediction.

The fourth image had some watermarking and was slightly misshapen during resizing.

The final image was also misshapen during resizing.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			                |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Speed limit (60km/h) 	        | Speed limit (100km/h)							| 
| Speed limit (30km/h) 	        | Speed limit (100km/h)							|
| Dangerous curve to the right	| Dangerous curve to the right					|
| Go straight or right     		| Speed limit (70km/h)			 				|
| Pedestrians       			| Speed limit (120km/h) 						|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set of 94.6%. Some reasons for this could be variations in the data particularly deformations while resizing, as well as a small sample size.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 100km/h sign (probability of 0.6), while the image does contain a speed limit sign it is 60km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58          			| 100km/h     									| 
| .26     				| 80km/h										|
| .07					| 120km/h										|
| .05	      			| 50km/h    					 				|
| .03				    | 60km/h              							|


For the second image the model is a little less confident in exactly which speed limit sign was present making a prediction of 100km/h with a probability of 0.49. All of the predictions were for a speed limit sign but the speed was incorrect. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49          			| 100km/h     									| 
| .17     				| 60km/h										|
| .16					| 120km/h										|
| .09	      			| 50km/h    					 				|
| .03				    | 80km/h              							|

For the third image the model was highly confident in a prediction of Dangerous curve to the right with a 72% probability. This was the sign contained in the image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .72          			| Dangerous curve to the right					| 
| .19     				| Right-of-way at the next intersection 		|
| .04					| Dangerous curve to the left					|
| .02	      			| Slippery road 				 				|
| .01				    | Children crossing   							|

For the fourth image the model prediction was 70km/h with a 0.42 probability. The image contained a Go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42          			| 70km/h					                    | 
| .22     				| Dangerous curve to the right          		|
| .17					| General caution           					|
| .08	      			| Road work      				 				|
| .06				    | Go straight or right 							|

For the final image the model predicted a 120km/h with a low probability of 0.28. The image contained a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .28          			| 120km/h					                    | 
| .24     				| 50km/h                                  		|
| .15					| Keep right                   					|
| .10	      			| 20km/h          				 				|
| .08				    | 100km/h            							|