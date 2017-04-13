import sys
import csv
import cv2
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D
from keras import backend as K


samples = []
with open("R:/data/driving_log.csv") as f:
    lines = csv.reader(f)
    next(lines, None)
    for line in lines:
        samples.append({
            "path": "R:/data/" + line[0].split("\\")[-1],
            "steer": float(line[3])
        })
train_samples, validation_samples = train_test_split(samples, test_size=.2)


def batch_gen(samples, batch_size):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_X, batch_y = [], []
            for batch_sample in batch_samples:
                 batch_X.append(cv2.imread(batch_sample["path"]))
                 batch_y.append(batch_sample["steer"])
            yield np.array(batch_X), np.array(batch_y)

def Resize(img, size, method):
    return K.tf.image.resize_images(img, size, method)

def Normalize(img):
    return (img / 255.0) - .5

model = Sequential()
#model.add(Cropping2D(cropping=((70, 10), (0, 0)), dim_ordering="tf"))
#model.add(Lambda(function=Resize, arguments={"size": (40, 40), "method": tf.image.ResizeMethod.AREA}))
model.add(Lambda(function=Normalize, input_shape=(160, 320, 3)))
#model.add(Convolution2D(6, 3, 3))
model.add(Flatten())
model.add(Dense(1))

#img = cv2.imread("samples/IMG/center_2017_04_12_11_41_09_070.jpg")
#output = model.predict(x=np.array(img[None, ...]), batch_size=1)
#cv2.imshow("im", output.squeeze())
#cv2.waitKey(0)
#exit()

model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
history = model.fit_generator(
    batch_gen(train_samples, 32),
    samples_per_epoch=len(train_samples),
    nb_epoch=5,
    validation_data=batch_gen(validation_samples, 32),
    nb_val_samples=len(validation_samples))

model.save("model.h5")
print("Model saved")
