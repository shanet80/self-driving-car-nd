import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import warp, resize
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

def flip(image, angle):
    if np.random.randint(0,2) == 1:
        image = image[:, ::-1, :]
        angle = -1 * angle
    return image, angle

def shift(image, angle):
    dx = 40 * (np.random.rand() - 0.5)
    dy = 20 * (np.random.rand() - 0.5)
    trans_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    image = warp(image, trans_matrix)
    angle += -dx * 0.005
    return image, angle

def generator(samples, batch_size=500, train=False):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './p3-behavioral-cloning/data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle = angle + 0.1 # left
                    elif i == 2:
                        angle = angle - 0.1 # right
                    if train == True:
                        image, angle = flip(image, angle)
                        image, angle = shift(image, angle)
                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

samples = []
with open('./p3-behavioral-cloning/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    first = True
    for line in reader:
        if first is True:
            first = False
        else:
            samples.append(line)


print('Total Samples: {}'.format(len(samples)))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32, train=True)
validation_generator = generator(validation_samples, batch_size=32, train=False)

#Create Model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,(5,5), activation='relu', strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5), activation='relu', strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
checkpoint = ModelCheckpoint('./p3-behavioral-cloning/model-{epoch:02d}.h5', monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit_generator(train_generator, steps_per_epoch=(20000/32),
                           validation_data=validation_generator, validation_steps=(4480/32),
                           epochs=10, callbacks=[early_stop, checkpoint])

model.save('./p3-behavioral-cloning/model.h5')