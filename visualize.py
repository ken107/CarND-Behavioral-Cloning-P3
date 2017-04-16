from data import *
from model import *
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model("model.h5")
model2 = model_create(model)

test_samples = samples_get('data/validation/')
test_gen = batch_gen(test_samples, 32)
batch_X, batch_y = next(test_gen)

output = model2.predict(batch_X[0:1], batch_size=1)
for i in range(0, output.shape[3]):
    plt.subplot(1, 1, i+1)
    plt.imshow(output[0,:,:,i], cmap='gray')
plt.show()
