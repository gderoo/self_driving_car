import os, csv, cv2, sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import scipy.ndimage

samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0], line[1] = line[1], line[0] # Correcting the order to make later code cleaner
        samples.append(line)


train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2) # excluding the header line

def random_modif(img, ang):
    
    image, angle = img, ang
    
    # Random flipping
    if (np.random.uniform()<0.5):
        image = cv2.flip(image,1)
        angle = angle*-1.0
                
    # Random brightness
    factor = np.random.uniform(0.5, 1.2)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * factor
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Smoothing for position in the lane
    s = np.random.uniform(-10, 10)
    image = scipy.ndimage.interpolation.shift(image,[0,s,0], mode = 'nearest')
    
    # Angle smoothing
    angle = angle + np.random.uniform(-0.1, 0.1)

    return image, angle


def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        i_batch = 0
        n_batch = 0
        images = []
        angles = []

        while n_batch < batch_size:

            sample = samples[i_batch]
            i_batch += 1
            
            # Random selection of the image
            n = np.random.choice([0,1,2])
            name = './data/IMG/'+sample[n].split('/')[-1]
            image = cv2.imread(name)
            angle = float(sample[3])+0.25-0.25*n
            
            # Random data preprocessing
            new_image, new_angle = random_modif(image, angle)
            
            # Under sampling of small angles
            if (np.random.uniform()<0.4) or (new_angle > 0.1):
                images.append(new_image)
                angles.append(new_angle)
                n_batch +=1

        X_train = np.array(images)
        y_train = np.array(angles)
        yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

row, col, ch = 160, 320, 3  # Regular image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.)) # Normalization
model.add(Convolution2D(24,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 256*(2**5), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

model.save('model.h5')


