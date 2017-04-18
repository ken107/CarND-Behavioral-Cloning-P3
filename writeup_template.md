# **Behavioral Cloning Project**

## Writeup

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/augment.PNG "Augmented samples"
[image2]: ./examples/hidden_1.PNG "Conv 1 Output"
[image3]: ./examples/hidden_2.PNG "Conv 2 Output"
[image4]: ./examples/model_summary.PNG "Model Architecture"
[image5]: ./examples/training_result.PNG "Training Result"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* _train.py_ is the main training script
* _data.py_ containing scripts to load data and create generators
* _model.py_ containing the script to create the model
* _visualize.py_ containing script to view/debug hidden layer outputs
* _drive.py_ for driving the car in autonomous mode (only modified speed setting)
* _model.h5_ containing a trained convolution neural network
* _video.mp4_ showing autonomous driving on both track 1 and 2
* _writeup_report.md_ summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

My model can drive track 1 safely at 25 mph.  It can drive track 2 safely at 13 mph.  Please modify _drive.py_ correspondingly before testing.

#### 3. Submission code is usable and readable

The _train.py_ file contains the code for training and saving the convolution neural network. It imports _data.py_ and _model.py_ for managing data and creating the model. The code in these 3 files are fairly straightforward and requires few comments.

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After much experimentation (see explanation in next section), I choose to use the NVIDIA architecture.

#### 2. Attempts to reduce overfitting in the model

The model contains Dropout layers in order to reduce overfitting (_model.py_ line 35).

Also, by training on both tracks and augmenting the dataset by flipping & using left/right images, there are sufficiently large number of samples to minimize overfitting (_data.py_ line 23).

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (_train.py_ line 20).  I use a batch_size of 32, dropout rate of .5.

#### 4. Appropriate training data

For track 1, I use the provided sample training data.  For track 2, I drove total 8 laps, forward & backward, to generate training data.

To augment the data, I use the images from the left and right cameras with a correction factor of 10%.  In other words, if the steering angle is `a` then the left-corrected steering angle is `a+.1*abs(a)` and the right-corrected steering angle is `a-.1*abs(a)` (_data.py_ line 24).  In addition, I flip every image horizontally and add it to the training set.

For details about how I created the training data, see the next section.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially I attempted to experiment with various architectures, starting with LeNet and then adding/removing layers, tweaking parameters.  I was able get the car to drive most of track 1 using an architecture similar to LeNet.  But I could not, in any concrete way, incrementally improve the architecture.

For example, I would tweak a parameter and rerun the training, but the loss would be mostly the same.  I would add an additional FC layer with activation, but I couldn't tell the car is driving any better.  I did try to visualize the feature maps of the first few Conv layers, but did not glean meaningful indicators as to how to improve the architecture.

*Feature map of 1st Convolution layer*
![output of 1st Conv layer][image2]

*Feature map of 2nd Convolution layer (same image)*
![output of 2nd Conv layer][image3]

In some cases, however, I did discover what _not_ to do, such as using `tanh` for a Conv layer.

Nonetheless I had general ideas about the purpose of the various layers.  The convolution layers are intended to extract features in the image, specifically the lane lines, and allow the following DNN to make decision based on those features.  Activation is an important part of convolution layers.  It's not clear to me how to pick kernel size, but I follow the commonly-used 3 or 5.  Using larger kernel sizes seemed to have a negative impact on the result.

It was also difficult to decide how many fully-connected layers to use.  I know I needed at least two to create a deep NN, with activation for nonlinearity.  But how many layers, or how many neurons per layer, was unclear.

#### 2. Final Model Architecture

In the end, I decided to just implement NVIDIA's architecture (_model.py_ line 22).  The NVIDIA paper doesn't mention what activation they use, so I chose "relu".  I added a Cropping layer to remove scenery, and a Lambda layer that resizes and normalizes the image.  Here is the model's summary:

![model architecture][image4]

#### 3. Creation of the Training Set & Training Process

I notice that the provided sample training data for track 1 was quite extensive, over 20 laps forward and backwards, so I decided to use it without recording my own.  I found that the data was more than sufficient.  To augment the dataset, I use the left and right camera images, and also flip every image horizontally, resulting in 6 times increase in the number of samples (around 48,000).

![augmented samples][image1]
*center/left/right image and their flipped versions*

With the right setting of the correction factor, I found a sweet spot and the car can drive track 1 smoothly and expertly at 30mph, possibly even faster.  However, after adding training data for track 2, it no longer drives smoothly on track 1, and could only do max 25 mph.

Track 2 is difficult to drive manually, even with a calibrated joystick.  I drove total 8 laps forward and backward.  It was hard to keep to the center, the whole drive was recovery driving.  The number of track 2 samples after augmentation is about 60,000.  After training, the model can drive track 2 safely at 13 mph.

For validation, instead of splitting the training data, I did some manual driving on both track 1 and 2 and use that as validation data.

**Preprocessing**

First, a Keras Cropping2D layer clips the top 60 and bottom 20 pixels, removing scenery and hood (_model.py_ line 25).  Removing scenery helps the network focus on the road only.  Removing the hood is necessary since the hood looks different in the left and right images and we don't want the network to train on that.

Secondly, a Lambda layer (_model.py_ line 26) resizes the image to 66x200 (the input resolution of the NVIDIA architecture), and then normalizes the pixel values to (-1, 1).

**Training results**

![training results][image5]

The training loss is slightly larger than the validation loss, indicating slight underfitting but no overfitting.  I used 3 epochs, additional epochs did not seem to improve result.  I used the Adam optimizer, so the learning rate was not tuned manually.

## Conclusion
Even though I did not use my own architecture, I experimented quite a bit with various configurations, and have gained better understanding of the different types of layers and their purposes.  Most importantly I learned how to get my hands dirty with Keras, surf its and TensorFlow's documentation, examine outputs of intermediate layers to gauge how well the network is learning, and appreciate the importance of having lots of data, and how to achieve that through simple augmentation.

In the end, it still surprises me how simply by watching me drive, this network of numbers learned to do the same, perhaps even better.  As one SDC student said, it's like "black magic".
