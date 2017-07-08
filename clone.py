import os
import csv
import numpy as np
import cv2
import numpy as np
import sklearn
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential

samples = []
with open('./data-2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data-2/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './data-2/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './data-2/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                
                correction = 0.21
                center_angle = float(batch_sample[3])
                left_angle =  center_angle + correction
                right_angle = center_angle - correction

                center_image_flipped = np.fliplr(center_image)
                center_measurement_flipped = -center_angle

                left_image_flipped = np.fliplr(left_image)
                right_image_flipped = np.fliplr(right_image)

                left_angle_flipped = -left_angle
                right_angle_flipped = -right_angle
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(center_image_flipped)
                images.append(left_image_flipped)
                images.append(right_image_flipped)

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                angles.append(center_measurement_flipped)
                angles.append(left_angle_flipped)
                angles.append(right_angle_flipped)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format


nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

nb_train = len(train_samples) * 6
nb_valid = len(validation_samples) * 6

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], border_mode='valid', batch_input_shape=(80, 320, 16, 3)))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
# Apply dropout of 25%
model.add(Dropout(0.25))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=nb_train, validation_data=validation_generator, nb_val_samples=nb_valid, nb_epoch=3)
model.save('./model.h5')
import gc
gc.collect()



