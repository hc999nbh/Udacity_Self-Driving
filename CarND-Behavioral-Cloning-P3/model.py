import csv
import cv2
import numpy as np
from PIL import Image

lines = []
images = []
measurements = []

# 导入第一组训练数据
with open('..\\train_data\\driving_log.csv') as csvfile:
	reader =  csv.reader(csvfile)
	for line in reader:
		lines.append(line)


# 导入第八组训练数据
with open('..\\train_data_ninth\\driving_log.csv') as csvfile:
	reader =  csv.reader(csvfile)
	for line in reader:
		lines.append(line)	


# 导入第九组训练数据
with open('..\\train_data_tenth\\driving_log.csv') as csvfile:
	reader =  csv.reader(csvfile)
	for line in reader:
		lines.append(line)	


for line in lines:
	#center_image = Image.open(line[0])
	#left_image = Image.open(line[1])
	#right_image = Image.open(line[2])

	center_image = np.array(Image.open(line[0]))
	left_image = np.array(Image.open(line[1]))
	right_image = np.array(Image.open(line[2]))

	correction = 0.2
	center_measurement = float(line[3])
	left_measurement = center_measurement + correction
	right_measurement = center_measurement - correction

	images.append(center_image)
	images.append(left_image)
	images.append(right_image)

	#measurements.extend(center_measurement , left_measurement , right_measurement)
	measurements.append(center_measurement)
	measurements.append(left_measurement)
	measurements.append(center_measurement)


#print(current_path)
augmented_images = []
augmented_measurements = []
for image , measurement in zip(images , measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image , 1))
	augmented_measurements.append(measurement * (-1.0))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)
print(y_train.shape)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


model = Sequential()
model.add(Lambda(lambda x : x / 255.0 -0.5 , input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping = ((70 , 25),(0 , 0))))

model.add(Conv2D(24, (5, 5), subsample = (2,2) , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Conv2D(36, (5, 5), subsample = (2,2) , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Conv2D(48, (5, 5), subsample = (2,2) , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Dropout(0.5))

model.add(Flatten())
#model.add(Dense(500 , activation = 'relu'))
model.add(Dense(100 , activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss = 'mse' , optimizer = 'adam')
model.fit(X_train , y_train , validation_split = 0.2 , shuffle = True , nb_epoch = 2)

model.save('model.h5')



