import os
import cv2
import pickle
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#from keras.layers import Sequential
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters 

DIRECTORY = 'C:\\Users\\SHERIF\\OneDrive\\Documents\\Dataset\\Dogs&Cats'
CATEGORIES = ['CAT', 'DOG']

IMG_SIZE = 100
data = []

for category in CATEGORIES:
	folder = os.path.join(DIRECTORY, category)
	label = CATEGORIES.index(category)
	for img in os.listdir(folder):
		img_path = os.path.join(folder, img)
		img_arr = cv2.imread(img_path)
		img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
		data.append([img_arr, label])

print(len(data))

random.shuffle(data)

X = []
y = []

for features, label in data:
	X.append(features)
	y.append(label)

X = np.array(X)
y = np.array(y)

print(len(X))
print(len(y))

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))