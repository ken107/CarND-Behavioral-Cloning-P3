from keras.models import Model
from keras.layers import *
from keras import backend as K


def Resize(img, size, method):
    return K.tf.image.resize_images(img, size, method)

def Normalize(img):
    return img / 255.0

def Concatenate(values):
    return K.tf.concat(1, values)

def model_create(from_model=None):
    image_input = Input(shape=(160,320,1))
    x = image_input
    #x = Cropping2D(cropping=((60,10), (0,0)))(x)
    #x = Lambda(function=Resize, arguments={"size": (32,32), "method": K.tf.image.ResizeMethod.AREA})(x)
    #x = Lambda(function=Normalize)(x)
    if from_model is not None:
        return Model(image_input, x)
    x = Convolution2D(16, 5, 5)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(.5)(x)
    if from_model is not None:
        x = Convolution2D(24, 3, 3, weights=from_model.layers[7].get_weights())(x)
        x = Activation("relu")(x)
        return Model(image_input, x)
    x = Convolution2D(24, 3, 3)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(.5)(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    return Model(image_input, x)

