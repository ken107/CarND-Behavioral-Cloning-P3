import csv
import re
import cv2
import numpy as np
from sklearn.utils import shuffle

def samples_get(path):
    samples = []
    with open(path + "driving_log.csv") as f:
        lines = csv.reader(f)
        for line in lines:
            if line[0] != 'center':
                samples.append({
                    "path": path + "IMG/" + re.split(r"[\\/]", line[0])[-1],
                    "steer": float(line[3])
                })
    return samples

def batch_gen(samples, batch_size):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_x, batch_y = [], []
            for batch_sample in batch_samples:
                batch_x.append(cv2.imread(batch_sample["path"]))
                batch_y.append(batch_sample["steer"])
            yield np.array(batch_x), np.array(batch_y)
