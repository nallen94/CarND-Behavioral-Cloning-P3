
# coding: utf-8

# In[3]:


#Importing and seggregating the images
import csv
import cv2
lines=[]
images=[]

with open('../Data/driving_log.csv') as csvfile: #Reading the csv file for the name and location of images
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurement=[]
steer=[]
for i in range(16000):
    source_path=lines[i][0] #Storing all the center images
    filename=source_path.split('\\')[-1]
    current_path='../Data/Image/'+filename
    image=cv2.imread(current_path)
    images.append(image)
    measurement=float(lines[i][3]) #Storing all the steer measurements related to center images
    steer.append(measurement)

# Adding right and left images and steering measurements to training data
measurement_left = [0.25 + x for x in steer] #Including the offset steering values for left image
measurement_right = [-0.25 + x for x in steer] #Including the offset steering values for left image
steer = steer + measurement_left #Adding these values to the steer measurement array
print(len(steer))
steer = steer + measurement_right #Adding these values to the steer measurement array
print(len(steer))
# steer.extend(measurement_right)
for i in range(16000):
    source_path = lines[i][1]
    #     print(source_path)
    filename = source_path.split('\\')[-1]
    current_path = '../Data/Image/' + filename
    image = cv2.imread(current_path) #Appending left images
    images.append(image)

for i in range(16000):
    source_path = lines[i][2]
    #     print(source_path)
    filename = source_path.split('\\')[-1]
    current_path = '../Data/Image/' + filename
    image = cv2.imread(current_path) #Appending right images
    images.append(image)

import numpy as np
X_train=np.array(images) #Converting the list into array
y_train=np.array(steer) #Converting the list into array

#Creating and training neural net with keras (NVIDIA Network)
from keras.models import Sequential,load_model
from keras.layers import Flatten,Dense,Convolution2D,Lambda,MaxPooling2D,Cropping2D,Dropout

model=Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3))) #Lambda layer to normalize the image data
model.add(Cropping2D(cropping=((50,20), (0,0)))) #Cropping the images to eliminate the unrequired part
model.add(Convolution2D(24,5,5,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train ,y_train , validation_split=0.2, shuffle=True, nb_epoch=8) #Training the model for 8 epochs

model.save('model.h5') #Storing the model

