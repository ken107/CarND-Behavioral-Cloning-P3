from keras import backend as K
from model import *
from data import *

K.tf.app.flags.DEFINE_integer('epochs', 2, 'Number of epochs to train')
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
