import os, csv, cv2, sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import scipy.ndimage

# Aggregating the samples from the different sample sources
data_list = ['./data/', './data_manual/', './data_manual_counter/']
#data_list = ['./data/']
samples = []
for source in data_list:
    with open(source+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0], line[1] = line[1], line[0] # Correcting the order to make later code cleaner
            for i in range(3):
                line[i] = source+'IMG/'+line[i].split('/')[-1] # Already correcting image paths
            samples.append(line)

# Data split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2) # excluding the header line

# Helper function to preprocess image for the generator
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
    s = np.random.uniform(-20, 20)
    image = scipy.ndimage.interpolation.shift(image,[0,s,0])+0.003*s
    # Angle smoothing
    angle = angle*(1+0.04*np.random.uniform(-1, 1))

    return image, angle

# Image generator
def generator_new(samples, batch_size=32):
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        i_batch, n_batch = 0, 0
        images, angles = [], []

        while n_batch < batch_size:

            sample = samples[i_batch]
            i_batch += 1
            
            # Random selection between left/center/right image
            n = np.random.choice([0,1,2])
            image = cv2.imread(sample[n])
            angle = float(sample[3])+0.25*(1-n) # correcting angle
            
            # Random data preprocessing
            new_image, new_angle = random_modif(image, angle)
            
            # Under sampling of small angles
            if (np.random.uniform()<0.5) or (abs(new_angle)>0.1):
                images.append(new_image)
                angles.append(new_angle)
                n_batch +=1

        X_train = np.array(images)
        y_train = np.array(angles)
        yield shuffle(X_train, y_train)

# Original image generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator_new(train_samples, batch_size=256)
validation_generator = generator_new(validation_samples, batch_size=256)

row, col, ch = 160, 320, 3  # Regular image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.)) # Normalization
model.add(Convolution2D(24,5,5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,3,3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,3,3, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 256*(2**6), validation_data=validation_generator, nb_val_samples=256*(2**2), nb_epoch=3)

model.save('model.h5')


