from keras.models import Model
from keras.layers import *
from keras import backend as K


def Normalize(img):
    #img = K.tf.image.resize_images(img, size, K.tf.image.ResizeMethod.AREA)
    img = K.tf.image.rgb_to_grayscale(img)
    return img / 127.5 - 1

def Concatenate(values):
    return K.tf.concat(1, values)

def model_create(from_model=None):
    image_input = Input(shape=(160,320,3))
    x = image_input
    x = Cropping2D(cropping=((60,10), (0,0)))(x)
    x = Lambda(function=Normalize)(x)
    x = Convolution2D(24, 3, 3, activation="relu")(x)
    x = MaxPooling2D((3,3))(x)
    x = Dropout(.75)(x)
    #if from_model is not None:
    #    x = Convolution2D(48, 3, 3, activation="relu", weights=from_model.layers[6].get_weights())(x)
    #    return Model(image_input, x)
    #x = Convolution2D(32, 3, 3, activation="relu")(x)
    #x = MaxPooling2D((2,2))(x)
    #x = Dropout(.75)(x)
    x = Flatten()(x)
    x = Dense(1, activation="tanh")(x)
    return Model(image_input, x)
