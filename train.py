from keras import backend as K
from model import *
from data import *

K.tf.app.flags.DEFINE_integer('epochs', 2, 'Number of epochs to train')
K.tf.app.flags.DEFINE_integer('batch_size', 32, 'Training batch size')
FLAGS = K.tf.app.flags.FLAGS

# training and training2 contain training data for track 1 and 2 respectively.
train_samples = samples_get("data/training/") + samples_get("data/training2/")
# I record some manual driving and use as validation data, rather than splitting the training data
valid_samples = samples_get("data/validation/") + samples_get("data/validation2/")
train_gen = batch_gen(train_samples, FLAGS.batch_size)
valid_gen = batch_gen(valid_samples, FLAGS.batch_size)
print(len(train_samples), len(valid_samples))


model = model_create()
model.summary()
model.compile(optimizer="adam", loss="mse")
model.fit_generator(
    train_gen,
    samples_per_epoch=len(train_samples),
    nb_epoch=FLAGS.epochs,
    validation_data=valid_gen,
    nb_val_samples=len(valid_samples))

model.save("model.h5")
print("Model saved")
