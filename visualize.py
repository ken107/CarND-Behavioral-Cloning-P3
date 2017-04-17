from data import *
from model import *
from keras.models import load_model
import matplotlib.pyplot as plt


plt.figure(figsize=(12,6))

test_samples = samples_get('R:/validation/')
test_gen = batch_gen(test_samples, 32)
batch_X, batch_y = next(test_gen)
print(batch_X.shape, batch_y.shape)


model = load_model("model.h5")


### show layer output
#model2 = model_create(model)
#output = model2.predict(batch_X, batch_size=1)
#plt.subplot(211); plt.imshow(batch_X[2])
#plt.subplot(212); plt.imshow(output[2])


### show batch image
for i in range(0, 6):
    plt.subplot(3, 2, i+1); plt.imshow(batch_X[18+i])
    print(batch_y[18+i])

plt.show()
