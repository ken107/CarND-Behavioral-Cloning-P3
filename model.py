import re
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Cropping2D, Convolution2D, MaxPooling2D
from keras import backend as K


### data
def samples_get(path):
    samples = []
    with open(path + "driving_log.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            if line[0] != 'center':
                samples.append({
                    "path": path + "IMG/" + re.split(r"[\\/]", line[0])[-1],
                    "speed": float(line[6]),
                    "steer": float(line[3])
                })
    return samples

def batch_gen(samples, batch_size):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_x1, batch_x2, batch_y = [], [], []
            for batch_sample in batch_samples:
                batch_x1.append(cv2.imread(batch_sample["path"]))
                batch_x2.append(batch_sample["speed"])
                batch_y.append(batch_sample["steer"])
            yield [np.array(batch_x1), np.array(batch_x2)], np.array(batch_y)


### model
def Resize(img, size, method):
    return K.tf.image.resize_images(img, size, method)

def Normalize(img):
    return img / 255.0

def Concatenate(values):
    return K.tf.concat(1, values)

def model_create():
    image_input = Input(shape=(160,320,3))
    speed_input = Input(shape=(1,))
    x = image_input
    #x = Cropping2D(cropping=((70,10), (0,0)))(x)
    x = Lambda(function=Resize, arguments={"size": (32,32), "method": K.tf.image.ResizeMethod.AREA})(x)
    x = Lambda(function=Normalize)(x)
    x = Flatten()(x)
    x = Lambda(function=Concatenate)([x, speed_input])
    x = Dense(200, activation="relu")(x)
    x = Dense(120, activation="relu")(x)
    x = Dense(84, activation="relu")(x)
    x = Dense(1)(x)
    return Model([image_input, speed_input], x)


### training
K.tf.app.flags.DEFINE_integer('epochs', 3, 'Number of epochs to train')
K.tf.app.flags.DEFINE_integer('batch_size', 32, 'Training batch size')
K.tf.app.flags.DEFINE_string('training', 'data/training/', 'Path containing training CSV & images')
K.tf.app.flags.DEFINE_string('validation', 'data/validation/', 'Path containing validation CSV & images')
FLAGS = K.tf.app.flags.FLAGS

train_path = FLAGS.training
valid_path = FLAGS.validation
train_samples = samples_get(train_path)
valid_samples = samples_get(valid_path)
train_gen = batch_gen(train_samples, FLAGS.batch_size)
valid_gen = batch_gen(valid_samples, FLAGS.batch_size)

model = model_create()
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
#batch_X, batch_y = next(train_gen)
#output = model.predict(batch_X[0:1], batch_size=1)
#cv2.imshow("", output[0])
#cv2.waitKey(0)
#exit()
model.fit_generator(
    train_gen,
    samples_per_epoch=len(train_samples),
    nb_epoch=FLAGS.epochs,
    validation_data=valid_gen,
    nb_val_samples=len(valid_samples))

model.save("model.h5")
print("Model saved")
