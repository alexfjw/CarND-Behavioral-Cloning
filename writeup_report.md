# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnnarch.png "Model Visualization"
[centerlane]: ./examples/centerlane.jpg "center lane example"
[normal]: ./examples/normal.jpg "Normal Image"
[flipped]: ./examples/flipped.jpg "Flipped Image" 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* project_nb.ipynb containing the code to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4, a recording of the car simluation running autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The project_nb.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used was a model from Nvidia's End-to-End Deep Learning for Self-Driving Cars paper, with summary [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

Images from a front facing car camera were used to predict the steering angle that the car should follow. Mean squared error was used as the loss function.

First, we shift the pixel values to be between -0.5 & 0.5.
Then, we crop out unwanted sections of the image (such as the sky, and the car hood)
Finally, we follow the Nvidia architecture by the word, using 3 5x5 convolutional layers, 2 3x3 convolutional layers, and 3 fully connected layers. One additional, final layer (not present in Nvidia paper) was added after that to predict the steering angle.

The network architecture is as follows: 

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0

```

#### 2. Attempts to reduce overfitting in the model

Similar to how deep learning models in NLP reduce overfitting, I trained the model on only a small number of epochs, 1 epoch, in this case. This means that the model looks at new data at every point of training.

The model was trained and validated on different data sets to ensure that the model was not overfitting. This can be seen in code cell 8 of the IPython notebook. No test set was created as the model can be tested by running through the simulator and ensuring that the vehicle stays on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code cell 10).

#### 4. Appropriate training data

The data was made up of 2 laps of the first track going forward, and 2 laps around the first track going in the opposite direction. I only used center lane driving. However, data augmentation was used to increase the amount of data for training.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing architecutre that was proven to work well with self driving cars. 

My first step was to use a convolution neural network model similar to ones used in production. This turned out to be the model from Nvidia's End-to-End Deep Learning for Self-Driving Cars paper, [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
This model is likely to be appropriate as it has been experimented on for real self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had quite a close loss value on both the training set and validation set, training loss: 0.0079 & val_loss: 0.0112. This implied that minimal overfitting was present.

This is likely due to the fact that only 1 epoch was used, meaning the model was never trained on repeated data.

The final step was to run the simulator to see how well the car was driving around track one. The car did not fall off the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

As mentioned above, the following is the architecture used.

First, we shift the pixel values to be between -0.5 & 0.5.
Then, we crop out unwanted sections of the image (such as the sky, and the car hood)
Finally, we follow the Nvidia architecture by the word, using 3 5x5 convolutional layers, 2 3x3 convolutional layers, and 3 fully connected layers. One additional, final layer (not present in Nvidia paper) was added after that to predict the steering angle.

The network architecture is as follows: 

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0

```

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded laps on track one using center lane driving. 

Here is an example:

![alt text][centerlane]

The first track was driven forward for 2 laps, and driven in the opposite direction for another 2 laps.

To augment the data sat, I also flipped images and angles thinking that this would add additional valualbe data for training.

Normal Image:  
![alt text][normal]

Flipped Image:  
![alt text][flipped]

After the collection process, I had 37758 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the car successfully driving through the lap without falling deviating from the track at all. I used an adam optimizer so that manually training the learning rate wasn't necessary.
