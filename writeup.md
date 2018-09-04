# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/1_v1.png "Visualization"
[image2]: ./writeup/2_v2.png "Visualization" 
[image3]: ./writeup/3_gray.png "Grayscaling"
[image4]: ./writeup/4_norm.png "Normalization"
[image5]: ./writeup/5_new.png "Traffic Signs"
[image6]: ./writeup/6_soft.png "Softmax"
[image7]: ./writeup/7_soft.png "Softmax"

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/AhHosny/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an image of 10 random data points

![alt text][image1]

And here is a histogram of label frequency

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Data preprocessing

My dataset preprocessing consisted of:

1. Converting to grayscale - This worked well for Sermanet and LeCun as described in their traffic sign classification article. It also helps to reduce training time, which was nice when a GPU wasn't available.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

2. Normalizing the data to the range (-1,1) - This was done using the line of code X_train_normalized = (X_train - 128)/128. The resulting dataset mean wasn't exactly zero, but it was reduced from around 82 to roughly -0.35. I chose to do this mostly because it was suggested in the lessons and it was fairly easy to do.

Here is an example of traffic sign image before and after Normalization:

![alt text][image4]


#### 2. Final Model Description

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 1x1 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 1x1 stride,  outputs 5x5x16 					|
| Fully connected		| inputs 400 outputs 120  						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 120 outputs 84  						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 84 outputs 43  						|
|						|												|
 


#### 3. Training the model

I used the Adam optimizer (already implemented in the LeNet lab). The final settings used were:

1. batch size: 100
2. epochs: 75
3. learning rate: 0.001
4. mu: 0
5. sigma: 0.1
6. dropout keep probability: 0.5

#### 4. Results

My final model results were:
* validation set accuracy of 96.5%
* test set accuracy of 95%

I implemented the same architecture from the LeNet Lab, with a little changes since my dataset is in grayscale. This model worked quite well (~96.5% validation accuracy)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

#### 2.  Model's predictions on these new traffic signs 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Even better than the 96.5% validation accuracy and the 95% test accuracy. This is a good sign that the model performs well on real-world data. And while it's reasonable to assume that the accuracy would not remain so high given more data points.

#### 3.  How certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

![alt text][image6]

![alt text][image7]
