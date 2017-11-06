import csv
import cv2
import numpy as np

lines = []
with open('./p3-behavioral-cloning/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    first = True
    for line in reader:
        if first is True:
            first = False
        else:
            lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = lines[0][0]
    filename = source_path.split('/')[-1]
    current_path = './p3-behavioral-cloning/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('./p3-behavioral-cloning/model.h5')