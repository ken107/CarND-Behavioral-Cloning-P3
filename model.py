### LIST OF FILES
# train.py: the main training script
# data.py: manage data
# model.py: defines the model
# visualize.py: for debugging hidden layer outputs

from keras.models import Model
from keras.layers import *
from keras import backend as K


def Normalize(img):
    #img = K.tf.image.rgb_to_hsv(img)
    img = K.tf.image.resize_images(img, (66,200), method=K.tf.image.ResizeMethod.AREA)
    return img / 127.5 - 1

def Concatenate(values):
    return K.tf.concat(1, values)

# the from_model parameter is used in visualize.py to create a truncated model for debugging hidden layers' outputs
# NVIDIA architecture
def model_create(from_model=None):
    image_input = Input(shape=(160,320,3))
    x = image_input
    x = Cropping2D(cropping=((60,20), (0,0)))(x)
    x = Lambda(function=Normalize)(x)
    if from_model is not None:
        x = Convolution2D(24, 5, 5, activation="relu", subsample=(2,2), weights=from_model.layers[3].get_weights())(x)
        x = Convolution2D(36, 5, 5, activation="relu", subsample=(2,2), weights=from_model.layers[4].get_weights())(x)
        return Model(image_input, x)
    x = Convolution2D(24, 5, 5, activation="relu", subsample=(2,2))(x)
    x = Convolution2D(36, 5, 5, activation="relu", subsample=(2,2))(x)
    x = Convolution2D(48, 5, 5, activation="relu", subsample=(2,2))(x)
    x = Convolution2D(64, 3, 3, activation="relu")(x)
    x = Convolution2D(64, 3, 3, activation="relu")(x)
    x = Dropout(.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    x = Dense(50, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    x = Dense(1)(x)
    return Model(image_input, x)
