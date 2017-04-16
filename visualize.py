from data import *
from model import *
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model("model.h5")
model2 = model_create(model)

test_samples = samples_get('R:/validation/')
test_gen = batch_gen(test_samples, 32)
batch_X, batch_y = next(test_gen)

output = model2.predict(batch_X[0:1], batch_size=1)
print(output.shape, output[0,0,0])
plt.figure(figsize=(12,6))
for i in range(0, min(25, output.shape[3])):
    plt.subplot(5, 5, i+1)
    plt.imshow(output[0,:,:,i], cmap='gray')
plt.show()
