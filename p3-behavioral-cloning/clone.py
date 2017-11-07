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
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './p3-behavioral-cloning/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(36,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('./p3-behavioral-cloning/model.h5')