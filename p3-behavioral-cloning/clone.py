import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import warp, resize
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.callbacks import ModelCheckpoint


def flip(image, angle):
    if np.random.randint(0,2) == 1:
        image = image[:, ::-1, :]
        angle = -angle
    return image, angle

def brightness_adjustment(image):
    # convert to HSV so that its easy to adjust brightness
    new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    random_bright = .25 + np.random.uniform()

    # Apply the brightness reduction to the V channel
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright

    # convert to RGB again
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def generator(samples, batch_size=32, train=False):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                index = np.random.randint(0,3)
                name = './p3-behavioral-cloning/data/IMG/'+batch_sample[index].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[3])
                if index == 1:
                    angle = angle + 0.1 # left
                elif index == 2:
                    angle = angle - 0.1 # right
                if train == True:
                    image, angle = flip(image, angle)
                    image = brightness_adjustment(image)
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

train_samples, validation_samples = train_test_split(samples, test_size=0.20)

print('Training Samples: {}'.format(len(train_samples)))
print('Validation Samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32, train=True)
validation_generator = generator(validation_samples, batch_size=32, train=False)

#Create Model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,(5,5), activation='elu', strides=(2,2)))
model.add(Conv2D(36,(5,5), activation='elu', strides=(2,2)))
model.add(Conv2D(48,(5,5), activation='elu', strides=(2,2)))
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='elu'))
model.summary()

model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('./p3-behavioral-cloning/model-{epoch:02d}.h5', monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit_generator(train_generator, steps_per_epoch=(500),
                           validation_data=validation_generator, validation_steps=(100),
                           epochs=3, callbacks=[checkpoint])

model.save('./p3-behavioral-cloning/model.h5')