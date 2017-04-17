import csv
import re
import cv2
import numpy as np
from sklearn.utils import shuffle

def sample_add(samples, path, steer):
    samples.append({"path": path, "steer": steer, "flip": False})
    samples.append({"path": path, "steer": steer, "flip": True})

def samples_get(path):
    samples = []
    with open(path + "driving_log.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            if line[0] != 'center':
                steer = float(line[3])
                sample_add(samples, path + "IMG/" + re.split(r"[\\/]", line[0])[-1], steer)
                sample_add(samples, path + "IMG/" + re.split(r"[\\/]", line[1])[-1], steer+.1*abs(steer))
                sample_add(samples, path + "IMG/" + re.split(r"[\\/]", line[2])[-1], steer-.1*abs(steer))
    return samples

def batch_gen(samples, batch_size):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_x, batch_y = [], []
            for batch_sample in batch_samples:
                steer = batch_sample["steer"]
                img = cv2.imread(batch_sample["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if batch_sample["flip"]:
                    img = cv2.flip(img, 1)
                    steer = -steer
                batch_x.append(img)
                batch_y.append(steer)
            yield np.array(batch_x), np.array(batch_y)
